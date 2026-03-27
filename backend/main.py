import torch
import shutil
import subprocess
import os
from pathlib import Path
import threading
import cv2
import numpy as np
import sys
from functools import cached_property

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from backend.tools.common_tools import is_video_or_image, is_image_file
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.inpaint.sttn_inpaint import STTNInpaint, STTNVideoInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.video_inpaint import VideoInpaint
from backend.inpaint.e2fgvi_inpaint import E2FGVIInpaint
from backend.tools.inpaint_tools import create_mask, batch_generator
import importlib
import platform
import tempfile
import multiprocessing
from shapely.geometry import Polygon
import time
import json
import glob
import pickle
import hashlib
from tqdm import tqdm

# Saved by the GUI (Advanced Settings); merged after each importlib.reload(config).
_VSR_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI_USER_SETTINGS_PATH = os.path.join(_VSR_APP_ROOT, 'gui_user_settings.json')
GUI_USER_SETTINGS_KEYS = frozenset({
    'ANTIFLICKER_ALPHA', 'ANTIFLICKER_MOTION_THRESHOLD',
    'E2FGVI_NEIGHBOR_LENGTH', 'E2FGVI_REF_STRIDE',
    'MASK_FEATHER_RADIUS', 'SUBTITLE_AREA_DEVIATION_PIXEL',
    'E2FGVI_MAX_LOAD_NUM', 'E2FGVI_STREAM_MAX_LOAD',
    'E2FGVI_CROP_MARGIN', 'E2FGVI_MAX_CROP_SIDE',
    'MANUAL_BOXES_ONLY', 'FORCE_INPAINT_SELECTED_AREAS',
    'E2FGVI_FORCE_SCENE_DETECT', 'ENABLE_POST_DEFLICKER',
})


def apply_gui_user_settings_overlay():
    """Re-apply gui_user_settings.json onto config after importlib.reload(config)."""
    if not os.path.isfile(GUI_USER_SETTINGS_PATH):
        return
    try:
        with open(GUI_USER_SETTINGS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for k, v in data.items():
        if k not in GUI_USER_SETTINGS_KEYS:
            continue
        if not hasattr(config, k):
            continue
        try:
            cur = getattr(config, k)
            if isinstance(cur, bool):
                setattr(config, k, bool(v))
            elif isinstance(cur, int):
                setattr(config, k, int(v))
            elif isinstance(cur, float):
                setattr(config, k, float(v))
            else:
                setattr(config, k, v)
        except Exception:
            pass


def _opencv_image_ok(img):
    """True if ndarray is non-empty and has valid H×W for cv2.cvtColor etc."""
    try:
        return (
            img is not None
            and isinstance(img, np.ndarray)
            and img.size > 0
            and img.ndim >= 2
            and int(img.shape[0]) > 0
            and int(img.shape[1]) > 0
        )
    except Exception:
        return False


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self, video_path, sub_area=None):
        self.video_path = video_path
        self.sub_area = sub_area

    @staticmethod
    def _normalize_sub_areas(sub_area):
        """
        Normalize user-specified area(s) into a list of (ymin, ymax, xmin, xmax).
        Accepts:
          - None
          - a single tuple/list: (ymin, ymax, xmin, xmax)
          - a list/tuple of tuples/lists
        """
        if sub_area is None:
            return []
        if isinstance(sub_area, (list, tuple)):
            if len(sub_area) == 4 and all(isinstance(v, (int, float)) for v in sub_area):
                return [tuple(int(v) for v in sub_area)]
            normalized = []
            for item in sub_area:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    normalized.append(tuple(int(v) for v in item))
            return normalized
        return []

    @cached_property
    def text_detector(self):
        import paddle
        paddle.disable_signal_handler()
        from paddleocr.tools.infer import utility
        from paddleocr.tools.infer.predict_det import TextDetector
        importlib.reload(config)
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = self.convertToOnnxModelIfNeeded(config.DET_MODEL_PATH)
        args.use_onnx = len(config.ONNX_PROVIDERS) > 0
        args.onnx_providers = config.ONNX_PROVIDERS
        # Lower thresholds to detect any-color text including colored/low-contrast subtitles.
        # Default DB thresholds (0.3 / 0.6) miss colored text because binarization scores
        # for yellow/red/cyan text are typically lower than for white text.
        args.det_db_thresh = 0.2           # pixel-level binarization threshold (was 0.3)
        args.det_db_box_thresh = 0.45      # box-level confidence threshold (was 0.6)
        args.det_db_unclip_ratio = 2.0     # expand detected boxes outward (covers color bg boxes)
        return TextDetector(args)

    @staticmethod
    def _clahe_enhance(img):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) per-channel.
        Boosts local contrast so text is detectable on dark/night/low-light frames.
        Only used for detection — does not affect the inpainted video output.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def detect_subtitle(self, img):
        """
        Multi-pass detection for any-color text including night/dark scenes.

        CLAHE pre-processing boosts local contrast before detection so text on
        dark interrogation-room or night-scene frames is reliably found.
        Pass 1: CLAHE-enhanced original  → catches white/bright text
        Pass 2: CLAHE-enhanced inverted  → catches dark/colored text on light bg
        Deduplication by center-proximity removes boxes found by both passes.
        """
        all_boxes = []

        enhanced = self._clahe_enhance(img)

        # Pass 1: contrast-enhanced original
        boxes1, elapse = self.text_detector(enhanced)
        if boxes1 is not None and len(boxes1) > 0:
            all_boxes.append(boxes1)

        # Pass 2: contrast-enhanced inverted — flips polarity for any-color text
        inverted = cv2.bitwise_not(enhanced)
        boxes2, _ = self.text_detector(inverted)
        if boxes2 is not None and len(boxes2) > 0:
            all_boxes.append(boxes2)

        if not all_boxes:
            return np.array([]), elapse

        combined = np.concatenate(all_boxes, axis=0)

        # Deduplicate: drop boxes whose center is within 20×10 px of an already-kept box.
        # Both passes often detect the same text; keeping duplicates would double the mask area.
        kept = []
        for box in combined:
            cx = float(np.mean(box[:, 0]))
            cy = float(np.mean(box[:, 1]))
            duplicate = False
            for k in kept:
                kx = float(np.mean(k[:, 0]))
                ky = float(np.mean(k[:, 1]))
                if abs(cx - kx) < 20 and abs(cy - ky) < 10:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)

        return (np.array(kept) if kept else np.array([])), elapse

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = min(x1, x2, x3, x4)
                xmax = max(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                ymax = max(y1, y2, y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    def find_subtitle_frame_no(self, sub_remover=None):
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(
            total=int(frame_count),
            unit='frame',
            position=0,
            file=sys.__stdout__,
            desc='Subtitle Finding',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
        )
        current_frame_no = 0
        subtitle_frame_no_box_dict = {}
        print('[Processing] start finding subtitles...')
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            dt_boxes, elapse = self.detect_subtitle(frame)
            coordinate_list = self.get_coordinates(dt_boxes.tolist())
            if coordinate_list:
                temp_list = []
                normalized_sub_areas = self._normalize_sub_areas(self.sub_area)
                for coordinate in coordinate_list:
                    xmin, xmax, ymin, ymax = coordinate
                    if normalized_sub_areas:
                        matched = False
                        cx = (xmin + xmax) // 2
                        cy = (ymin + ymax) // 2
                        for s_ymin, s_ymax, s_xmin, s_xmax in normalized_sub_areas:
                            # Use center-point check so detections that slightly
                            # overflow the user box edge are not silently dropped.
                            if s_xmin <= cx <= s_xmax and s_ymin <= cy <= s_ymax:
                                matched = True
                                break
                            # Fallback: containment with 8px tolerance on every side
                            if (s_xmin - 8 <= xmin and xmax <= s_xmax + 8
                                    and s_ymin - 8 <= ymin
                                    and ymax <= s_ymax + 8):
                                matched = True
                                break
                        if matched:
                            temp_list.append((xmin, xmax, ymin, ymax))
                    else:
                        temp_list.append((xmin, xmax, ymin, ymax))
                if len(temp_list) > 0:
                    subtitle_frame_no_box_dict[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
                sub_remover._refresh_eta_from_total_progress(stage='finding')
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        # if config.UNITE_COORDINATES:
        #     subtitle_frame_no_box_dict = self.get_subtitle_frame_no_box_dict_with_united_coordinates(subtitle_frame_no_box_dict)
        #     if sub_remover is not None:
        #         try:
        #             # 当帧数大于1时，说明并非图片或单帧
        #             if sub_remover.frame_count > 1:
        #                 subtitle_frame_no_box_dict = self.filter_mistake_sub_area(subtitle_frame_no_box_dict,
        #                                                                           sub_remover.fps)
        #         except Exception:
        #             pass
        #     subtitle_frame_no_box_dict = self.prevent_missed_detection(subtitle_frame_no_box_dict)
        print('[Finished] Finished finding subtitles...')
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    def convertToOnnxModelIfNeeded(self, model_dir, model_filename="inference.pdmodel", params_filename="inference.pdiparams", opset_version=14):
        """Converts a Paddle model to ONNX if ONNX providers are available and the model does not already exist."""
        
        if not config.ONNX_PROVIDERS:
            return model_dir
        
        onnx_model_path = os.path.join(model_dir, "model.onnx")

        if os.path.exists(onnx_model_path):
            print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")
            return onnx_model_path
        
        print(f"Converting Paddle model {model_dir} to ONNX...")
        model_file = os.path.join(model_dir, model_filename)
        params_file = os.path.join(model_dir, params_filename) if params_filename else ""

        try:
            import paddle2onnx
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)

            # Convert and save the model
            onnx_model = paddle2onnx.export(
                model_filename=model_file,
                params_filename=params_file,
                save_file=onnx_model_path,
                opset_version=opset_version,
                auto_upgrade_opset=True,
                verbose=True,
                enable_onnx_checker=True,
                enable_experimental_op=True,
                enable_optimize=True,
                custom_op_info={},
                deploy_backend="onnxruntime",
                calibration_file="calibration.cache",
                external_file=os.path.join(model_dir, "external_data"),
                export_fp16_model=False,
            )

            print(f"Conversion successful. ONNX model saved to: {onnx_model_path}")
            return onnx_model_path
        except Exception as e:
            print(f"Error during conversion: {e}")
            return model_dir


    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    @staticmethod
    def get_scene_div_frame_no(v_path):
        """
        获取发生场景切换的帧号
        """
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        return scene_div_frame_no_list

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.PIXEL_TOLERANCE_X and abs(xmax1 - xmax2) <= config.PIXEL_TOLERANCE_X and \
            abs(ymin1 - ymin2) <= config.PIXEL_TOLERANCE_Y and abs(ymax1 - ymax2) <= config.PIXEL_TOLERANCE_Y

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                for idx, region in enumerate(current_regions):
                    last_standard_region = unify_value_map[last_key][idx] if idx < len(unify_value_map[last_key]) else None

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and self.are_similar(region, last_standard_region):
                        new_unify_values.append(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def sub_area_to_polygon(sub_area):
        """
        xmin, xmax, ymin, ymax = sub_area
        """
        s_xmin = sub_area[0]
        s_xmax = sub_area[1]
        s_ymin = sub_area[2]
        s_ymax = sub_area[3]
        return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])

    @staticmethod
    def expand_and_merge_intervals(intervals, expand_size=config.STTN_NEIGHBOR_STRIDE*config.STTN_REFERENCE_LENGTH, max_length=config.STTN_MAX_LOAD_NUM):
        # 初始化输出区间列表
        expanded_intervals = []

        # 对每个原始区间进行扩展
        for interval in intervals:
            start, end = interval

            # 扩展至至少 'expand_size' 个单位，但不超过 'max_length' 个单位
            expansion_amount = max(expand_size - (end - start + 1), 0)

            # 在保证包含原区间的前提下尽可能平分前后扩展量
            expand_start = max(start - expansion_amount // 2, 1)  # 确保起始点不小于1
            expand_end = end + expansion_amount // 2

            # 如果扩展后的区间超出了最大长度，进行调整
            if (expand_end - expand_start + 1) > max_length:
                expand_end = expand_start + max_length - 1

            # 对于单点的处理，需额外保证有至少 'expand_size' 长度
            if start == end:
                if expand_end - expand_start + 1 < expand_size:
                    expand_end = expand_start + expand_size - 1

            # 检查与前一个区间是否有重叠并进行相应的合并
            if expanded_intervals and expand_start <= expanded_intervals[-1][1]:
                previous_start, previous_end = expanded_intervals.pop()
                expand_start = previous_start
                expand_end = max(expand_end, previous_end)

            # 添加扩展后的区间至结果列表
            expanded_intervals.append((expand_start, expand_end))

        return expanded_intervals

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length=config.STTN_REFERENCE_LENGTH):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        """
        expanded = []
        # 首先单独处理单点区间以扩展它们
        for start, end in intervals:
            if start == end:  # 单点区间
                # 扩展到接近的目标长度，但保证前后不重叠
                prev_end = expanded[-1][1] if expanded else float('-inf')
                next_start = float('inf')
                # 查找下一个区间的起始点
                for ns, ne in intervals:
                    if ns > end:
                        next_start = ns
                        break
                # 确定新的扩展起点和终点
                new_start = max(start - (target_length - 1) // 2, prev_end + 1)
                new_end = min(start + (target_length - 1) // 2, next_start - 1)
                # 如果新的扩展终点在起点前面，说明没有足够空间来进行扩展
                if new_end < new_start:
                    new_start, new_end = start, start  # 保持原样
                expanded.append((new_start, new_end))
            else:
                # 非单点区间直接保留，稍后处理任何可能的重叠
                expanded.append((start, end))
        # 排序以合并那些因扩展导致重叠的区间
        expanded.sort(key=lambda x: x[0])
        # 合并重叠的区间，但仅当它们之间真正重叠且小于目标长度时
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            # 检查是否重叠
            if start <= last_end and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 需要合并
                merged[-1] = (last_start, max(last_end, end))  # 合并区间
            elif start == last_end + 1 and (end - last_start + 1 < target_length or last_end - last_start + 1 < target_length):
                # 相邻区间也需要合并的场景
                merged[-1] = (last_start, end)
            else:
                # 如果没有重叠且都大于目标长度，则直接保留
                merged.append((start, end))
        return merged

    def compute_iou(self, box1, box2):
        box1_polygon = self.sub_area_to_polygon(box1)
        box2_polygon = self.sub_area_to_polygon(box2)
        intersection = box1_polygon.intersection(box2_polygon)
        if intersection.is_empty:
            return -1
        else:
            union_area = (box1_polygon.area + box2_polygon.area - intersection.area)
            if union_area > 0:
                intersection_area_rate = intersection.area / union_area
            else:
                intersection_area_rate = 0
            return intersection_area_rate

    def get_area_max_box_dict(self, sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        _area_max_box_dict = dict()
        for start_no, end_no in sub_frame_no_list_continuous:
            # 寻找面积最大文本框
            current_no = start_no
            # 查找当前区间矩形框最大面积
            area_max_box_list = []
            while current_no <= end_no:
                for coord in subtitle_frame_no_box_dict[current_no]:
                    # 取出每一个文本框坐标
                    xmin, xmax, ymin, ymax = coord
                    # 计算当前文本框坐标面积
                    current_area = abs(xmax - xmin) * abs(ymax - ymin)
                    # 如果区间最大框列表为空，则当前面积为区间最大面积
                    if len(area_max_box_list) < 1:
                        area_max_box_list.append({
                            'area': current_area,
                            'xmin': xmin,
                            'xmax': xmax,
                            'ymin': ymin,
                            'ymax': ymax
                        })
                    # 如果列表非空，判断当前文本框是与区间最大文本框在同一区域
                    else:
                        has_same_position = False
                        # 遍历每个区间最大文本框，判断当前文本框位置是否与区间最大文本框列表的某个文本框位于同一行且交叉
                        for area_max_box in area_max_box_list:
                            if (area_max_box['ymin'] - config.THRESHOLD_HEIGHT_DIFFERENCE <= ymin
                                    and ymax <= area_max_box['ymax'] + config.THRESHOLD_HEIGHT_DIFFERENCE):
                                if self.compute_iou((xmin, xmax, ymin, ymax), (
                                        area_max_box['xmin'], area_max_box['xmax'], area_max_box['ymin'],
                                        area_max_box['ymax'])) != -1:
                                    # 如果高度差异不一样
                                    if abs(abs(area_max_box['ymax'] - area_max_box['ymin']) - abs(
                                            ymax - ymin)) < config.THRESHOLD_HEIGHT_DIFFERENCE:
                                        has_same_position = True
                                    # 如果在同一行，则计算当前面积是不是最大
                                    # 判断面积大小，若当前面积更大，则将当前行的最大区域坐标点更新
                                    if has_same_position and current_area > area_max_box['area']:
                                        area_max_box['area'] = current_area
                                        area_max_box['xmin'] = xmin
                                        area_max_box['xmax'] = xmax
                                        area_max_box['ymin'] = ymin
                                        area_max_box['ymax'] = ymax
                        # 如果遍历了所有的区间最大文本框列表，发现是新的一行，则直接添加
                        if not has_same_position:
                            new_large_area = {
                                'area': current_area,
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            }
                            if new_large_area not in area_max_box_list:
                                area_max_box_list.append(new_large_area)
                                break
                current_no += 1
            _area_max_box_list = list()
            for area_max_box in area_max_box_list:
                if area_max_box not in _area_max_box_list:
                    _area_max_box_list.append(area_max_box)
            _area_max_box_dict[f'{start_no}->{end_no}'] = _area_max_box_list
        return _area_max_box_dict

    def get_subtitle_frame_no_box_dict_with_united_coordinates(self, subtitle_frame_no_box_dict):
        """
        将多个视频帧的文本区域坐标统一
        """
        subtitle_frame_no_box_dict_with_united_coordinates = dict()
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        area_max_box_dict = self.get_area_max_box_dict(frame_no_list, subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                area_max_box_list = area_max_box_dict[f'{start_no}->{end_no}']
                current_boxes = subtitle_frame_no_box_dict[current_no]
                new_subtitle_frame_no_box_list = []
                for current_box in current_boxes:
                    current_xmin, current_xmax, current_ymin, current_ymax = current_box
                    for max_box in area_max_box_list:
                        large_xmin = max_box['xmin']
                        large_xmax = max_box['xmax']
                        large_ymin = max_box['ymin']
                        large_ymax = max_box['ymax']
                        box1 = (current_xmin, current_xmax, current_ymin, current_ymax)
                        box2 = (large_xmin, large_xmax, large_ymin, large_ymax)
                        res = self.compute_iou(box1, box2)
                        if res != -1:
                            new_subtitle_frame_no_box = (large_xmin, large_xmax, large_ymin, large_ymax)
                            if new_subtitle_frame_no_box not in new_subtitle_frame_no_box_list:
                                new_subtitle_frame_no_box_list.append(new_subtitle_frame_no_box)
                subtitle_frame_no_box_dict_with_united_coordinates[current_no] = new_subtitle_frame_no_box_list
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict_with_united_coordinates

    def prevent_missed_detection(self, subtitle_frame_no_box_dict):
        """
        添加额外的文本框，防止漏检
        """
        frame_no_list = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        for start_no, end_no in frame_no_list:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                if current_no + 1 != end_no and (current_no + 1) in subtitle_frame_no_box_dict.keys():
                    next_box_list = subtitle_frame_no_box_dict[current_no + 1]
                    if set(current_box_list).issubset(set(next_box_list)):
                        subtitle_frame_no_box_dict[current_no] = subtitle_frame_no_box_dict[current_no + 1]
                current_no += 1
                if current_no > end_no:
                    break
        return subtitle_frame_no_box_dict

    @staticmethod
    def get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict):
        sub_area_with_frequency = {}
        for start_no, end_no in sub_frame_no_list_continuous:
            current_no = start_no
            while True:
                current_box_list = subtitle_frame_no_box_dict[current_no]
                for current_box in current_box_list:
                    if str(current_box) not in sub_area_with_frequency.keys():
                        sub_area_with_frequency[f'{current_box}'] = 1
                    else:
                        sub_area_with_frequency[f'{current_box}'] += 1
                current_no += 1
                if current_no > end_no:
                    break
        return sub_area_with_frequency

    def filter_mistake_sub_area(self, subtitle_frame_no_box_dict, fps):
        """
        过滤错误的字幕区域
        """
        sub_frame_no_list_continuous = self.find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict)
        sub_area_with_frequency = self.get_frequency_in_range(sub_frame_no_list_continuous, subtitle_frame_no_box_dict)
        correct_sub_area = []
        for sub_area in sub_area_with_frequency.keys():
            if sub_area_with_frequency[sub_area] >= (fps // 2):
                correct_sub_area.append(sub_area)
            else:
                print(f'drop {sub_area}')
        correct_subtitle_frame_no_box_dict = dict()
        for frame_no in subtitle_frame_no_box_dict.keys():
            current_box_list = subtitle_frame_no_box_dict[frame_no]
            new_box_list = []
            for current_box in current_box_list:
                if str(current_box) in correct_sub_area and current_box not in new_box_list:
                    new_box_list.append(current_box)
            correct_subtitle_frame_no_box_dict[frame_no] = new_box_list
        return correct_subtitle_frame_no_box_dict


class SubtitleRemover:
    def __init__(self, vd_path, sub_area=None, gui_mode=False):
        importlib.reload(config)
        apply_gui_user_settings_overlay()
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置（支持单区域或多区域）
        self.sub_area = sub_area
        # 是否为gui运行，gui运行需要显示预览
        self.gui_mode = gui_mode
        # 判断是否为图片
        self.is_picture = False
        if is_image_file(str(vd_path)):
            self.sub_area = None
            self.is_picture = True
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建字幕检测对象
        self.sub_detector = SubtitleDetect(self.video_path, self.sub_area)
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象（可能被分片模式覆盖）
        self.video_writer = None
        self.video_out_name = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4')
        self.video_inpaint = None
        self.lama_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_name = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')
        if torch.cuda.is_available():
            print('use GPU for acceleration')
        if config.USE_DML:
            print('use DirectML for acceleration')
            if config.MODE != config.InpaintMode.STTN:
                print('Warning: DirectML acceleration is only available for STTN model. Falling back to CPU for other models.')
        for provider in config.ONNX_PROVIDERS:
            print(f"Detected execution provider: {provider}")


        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # Runtime timing stats (more reliable than tqdm ETA on variable workloads)
        self.run_started_at = None
        self.elapsed_seconds = 0.0
        self.eta_seconds = None
        self.eta_human = '--:--'
        self.elapsed_human = '00:00'
        self.eta_stage = 'starting'
        self._eta_last_log_ts = 0.0
        # 当前批次大小（用于GUI显示）
        self.current_chunk_size = 0
        # 预览帧
        self.preview_frame = None
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False
        # 抗闪烁缓存帧（上一帧输出）
        self.prev_output_frame = None
        # 终检使用的前一帧灰度图（用于检测局部闪烁）
        self.prev_final_check_gray = None
        self.selected_box_mask = self._build_selected_box_mask()
        # 进度检查点文件（用于异常中断后查看处理进度）
        self.progress_checkpoint_path = os.path.join(
            os.path.dirname(self.video_path),
            f'{self.vd_name}_no_sub.progress.json'
        )
        self.subtitle_intervals = []
        # 分片写入与断点续跑
        self.enable_chunk_checkpoint = (not self.is_picture) and bool(getattr(config, 'ENABLE_CHUNK_CHECKPOINT', True))
        self.chunk_seconds = max(1, int(getattr(config, 'CHUNK_SECONDS', 5)))
        self.chunk_frame_len = max(1, int(round(max(self.fps, 1) * self.chunk_seconds)))
        self.parts_dir = os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.parts')
        self.current_part_index = 0
        self.current_part_frame_count = 0
        self.output_frame_index = 0
        self.resume_processed_frames = 0
        self.completed_parts = []
        self.resume_start_frame = 0
        self.merged_parts_count = 0
        if self.enable_chunk_checkpoint:
            os.makedirs(self.parts_dir, exist_ok=True)
            self._load_checkpoint_for_resume()
            self._cleanup_incomplete_resume_part()
            self.current_part_index = self.output_frame_index // self.chunk_frame_len
            self.video_writer = self._open_part_writer(self.current_part_index)
        else:
            self.video_writer = cv2.VideoWriter(self.video_temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        # 字幕检测缓存
        self.detection_cache_path = self._build_detection_cache_path()

    @staticmethod
    def _normalize_sub_areas(sub_area):
        if sub_area is None:
            return []
        if isinstance(sub_area, (list, tuple)):
            if len(sub_area) == 4 and all(isinstance(v, (int, float)) for v in sub_area):
                return [tuple(int(v) for v in sub_area)]
            normalized = []
            for item in sub_area:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    normalized.append(tuple(int(v) for v in item))
            return normalized
        return []

    def _build_manual_only_submap(self):
        """Subtitle map from drawn boxes only; None if no manual areas."""
        manual_areas = self._normalize_sub_areas(self.sub_area)
        if not manual_areas:
            return None
        coords = []
        for ymin, ymax, xmin, xmax in manual_areas:
            coords.append((int(xmin), int(xmax), int(ymin), int(ymax)))
        if self.is_picture:
            return {1: coords}
        n = max(1, int(self.frame_count))
        return {i: list(coords) for i in range(1, n + 1)}

    def _build_selected_box_mask(self):
        """
        Build a mask from user-selected areas for fast final-check.
        """
        normalized = self._normalize_sub_areas(self.sub_area)
        if not normalized:
            return None
        mask = np.zeros(self.mask_size, dtype=np.uint8)
        for ymin, ymax, xmin, xmax in normalized:
            y0 = max(0, min(int(ymin), self.frame_height))
            y1 = max(0, min(int(ymax), self.frame_height))
            x0 = max(0, min(int(xmin), self.frame_width))
            x1 = max(0, min(int(xmax), self.frame_width))
            if y1 > y0 and x1 > x0:
                mask[y0:y1, x0:x1] = 255
        return mask if cv2.countNonZero(mask) > 0 else None

    def _build_detection_cache_path(self):
        """
        Cache key depends on source video + selected area.
        """
        if self.is_picture:
            return None
        try:
            area_repr = str(self._normalize_sub_areas(self.sub_area))
            key = f'{self.video_path}|{self.frame_count}|{self.frame_width}x{self.frame_height}|{area_repr}'
            digest = hashlib.md5(key.encode('utf-8')).hexdigest()[:12]
            return os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}.detect_cache.{digest}.pkl')
        except Exception:
            return None

    def _load_detection_cache(self):
        if not getattr(config, 'ENABLE_DETECTION_CACHE', False):
            return None
        if not self.detection_cache_path or not os.path.exists(self.detection_cache_path):
            return None
        try:
            with open(self.detection_cache_path, 'rb') as f:
                payload = pickle.load(f)
            if isinstance(payload, dict):
                print(f'[Cache] loaded subtitle detection cache: {os.path.basename(self.detection_cache_path)}')
                return payload
        except Exception:
            return None
        return None

    def _save_detection_cache(self, sub_list):
        if not getattr(config, 'ENABLE_DETECTION_CACHE', False):
            return
        if not self.detection_cache_path:
            return
        try:
            with open(self.detection_cache_path, 'wb') as f:
                pickle.dump(sub_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def _get_subtitle_map(self):
        """
        Reuse cached subtitle detections when available.
        """
        if bool(getattr(config, 'MANUAL_BOXES_ONLY', False)):
            manual_map = self._build_manual_only_submap()
            if manual_map is not None:
                print('[SubtitleMap] Manual boxes only — OCR skipped')
                return manual_map
        cached = self._load_detection_cache()
        if cached is not None:
            sub_list = cached
        else:
            sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            self._save_detection_cache(sub_list)
        if bool(getattr(config, 'FORCE_INPAINT_SELECTED_AREAS', False)):
            manual_areas = self._normalize_sub_areas(self.sub_area)
            if manual_areas and not self.is_picture:
                for i in range(1, int(self.frame_count) + 1):
                    frame_boxes = list(sub_list.get(i, []))
                    for ymin, ymax, xmin, xmax in manual_areas:
                        coord = (int(xmin), int(xmax), int(ymin), int(ymax))
                        if coord not in frame_boxes:
                            frame_boxes.append(coord)
                    if frame_boxes:
                        sub_list[i] = frame_boxes
            elif manual_areas and self.is_picture:
                frame_boxes = list(sub_list.get(1, []))
                for ymin, ymax, xmin, xmax in manual_areas:
                    coord = (int(xmin), int(xmax), int(ymin), int(ymax))
                    if coord not in frame_boxes:
                        frame_boxes.append(coord)
                if frame_boxes:
                    sub_list[1] = frame_boxes
        return sub_list

    @staticmethod
    def get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = min(x1, x2, x3, x4)
                xmax = max(x1, x2, x3, x4)
                ymin = min(y1, y2, y3, y4)
                ymax = max(y1, y2, y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f'{h:02d}:{m:02d}:{s:02d}'
        return f'{m:02d}:{s:02d}'

    def _refresh_eta_from_total_progress(self, stage='processing'):
        """
        Compute ETA from real elapsed wall-clock + total progress percent.
        More stable than tqdm's frame-rate estimate on mixed workloads.
        """
        if self.run_started_at is None:
            return
        now = time.time()
        self.elapsed_seconds = max(0.0, now - float(self.run_started_at))
        self.elapsed_human = self._format_duration(self.elapsed_seconds)
        self.eta_stage = str(stage or 'processing')
        p = float(max(0.0, min(100.0, float(self.progress_total))))
        if p <= 0.0:
            self.eta_seconds = None
            self.eta_human = '--:--'
            return
        if p >= 100.0:
            self.eta_seconds = 0.0
            self.eta_human = '00:00'
            return
        remaining = self.elapsed_seconds * (100.0 - p) / max(p, 1e-6)
        self.eta_seconds = max(0.0, remaining)
        self.eta_human = self._format_duration(self.eta_seconds)
        # Periodic console update (every ~10s) with measured ETA.
        if now - self._eta_last_log_ts >= 10.0:
            self._eta_last_log_ts = now
            print(f'[ETA] stage={self.eta_stage} elapsed={self.elapsed_human} remaining~{self.eta_human} progress={p:.1f}%')

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage) // 2
        self.progress_total = 50 + self.progress_remover
        self._refresh_eta_from_total_progress(stage='removing')
        # 按间隔落盘进度，避免频繁IO
        if int(tbar.n) % 20 == 0 or int(tbar.n) >= int(tbar.total):
            self._save_progress_checkpoint(processed_frames=int(tbar.n), stage='removing')

    def _save_progress_checkpoint(self, processed_frames, stage):
        if self.is_picture:
            return
        try:
            payload = {
                "source_video": self.video_path,
                "output_video": self.video_out_name,
                "processed_frames": int(processed_frames),
                "total_frames": int(self.frame_count),
                "progress_total_percent": int(self.progress_total),
                "stage": str(stage),
                "chunk_frame_len": int(self.chunk_frame_len),
                "parts_dir": self.parts_dir,
                "completed_parts": self.completed_parts,
                "updated_at": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(self.progress_checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_checkpoint_for_resume(self):
        try:
            if not os.path.exists(self.progress_checkpoint_path):
                return
            with open(self.progress_checkpoint_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            done = int(payload.get('processed_frames', 0) or 0)
            self.resume_processed_frames = max(0, min(done, self.frame_count))
            self.resume_start_frame = self.resume_processed_frames
            if isinstance(payload.get('completed_parts'), list):
                self.completed_parts = sorted([int(x) for x in payload['completed_parts'] if isinstance(x, int) or str(x).isdigit()])
            self.output_frame_index = self.resume_processed_frames
            self.current_part_frame_count = self.output_frame_index % self.chunk_frame_len
            if self.resume_processed_frames > 0:
                print(f'[Resume] checkpoint found, resume from frame {self.resume_processed_frames}')
        except Exception:
            self.resume_processed_frames = 0
            self.resume_start_frame = 0
            self.output_frame_index = 0
            self.current_part_frame_count = 0
            self.completed_parts = []

    def _cleanup_incomplete_resume_part(self):
        if not self.enable_chunk_checkpoint:
            return
        if self.resume_processed_frames <= 0:
            return
        # Remove current incomplete part so it can be regenerated cleanly.
        partial_idx = self.resume_processed_frames // self.chunk_frame_len
        partial_path = os.path.join(self.parts_dir, f'part_{partial_idx:05d}.mp4')
        if os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except Exception:
                pass

    def _open_part_writer(self, part_idx):
        part_path = os.path.join(self.parts_dir, f'part_{part_idx:05d}.mp4')
        return cv2.VideoWriter(part_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)

    def _write_output_frame(self, frame):
        if not self.enable_chunk_checkpoint:
            self.video_writer.write(frame)
            return
        self.video_writer.write(frame)
        self.output_frame_index += 1
        self.current_part_frame_count += 1
        if self.current_part_frame_count >= self.chunk_frame_len:
            self.video_writer.release()
            if self.current_part_index not in self.completed_parts:
                self.completed_parts.append(self.current_part_index)
                self.completed_parts = sorted(set(self.completed_parts))
            self.current_part_index += 1
            self.current_part_frame_count = 0
            self.video_writer = self._open_part_writer(self.current_part_index)

    def _merge_parts_to_temp_video(self):
        if not self.enable_chunk_checkpoint:
            return True
        # Close current part first.
        try:
            self.video_writer.release()
        except Exception:
            pass

        # Build ordered part list
        part_files = sorted(glob.glob(os.path.join(self.parts_dir, 'part_*.mp4')))
        self.merged_parts_count = len(part_files)
        if not part_files:
            return False
        concat_list = os.path.join(self.parts_dir, 'concat_list.txt')
        try:
            with open(concat_list, 'w', encoding='utf-8') as f:
                for p in part_files:
                    f.write(f"file '{p.replace(chr(92), '/')}'\n")
            merge_cmd = [
                config.FFMPEG_PATH, "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                "-loglevel", "error",
                self.video_temp_file.name
            ]
            use_shell = True if os.name == "nt" else False
            subprocess.check_output(merge_cmd, stdin=subprocess.DEVNULL, shell=use_shell)
            return True
        except Exception:
            return False
        finally:
            if os.path.exists(concat_list):
                try:
                    os.remove(concat_list)
                except Exception:
                    pass

    def _cleanup_parts_and_checkpoint(self):
        # Delete chunk files and checkpoint only after successful full completion.
        try:
            if self.enable_chunk_checkpoint and os.path.isdir(self.parts_dir):
                for p in glob.glob(os.path.join(self.parts_dir, 'part_*.mp4')):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                try:
                    os.rmdir(self.parts_dir)
                except Exception:
                    pass
        except Exception:
            pass
        if os.path.exists(self.progress_checkpoint_path):
            try:
                os.remove(self.progress_checkpoint_path)
            except Exception:
                pass

    def _log_write_frame(self, frame_no, masked=False):
        """
        Reduce console overhead by logging at intervals instead of every frame.
        """
        if frame_no <= 3 or frame_no % 20 == 0 or frame_no >= self.frame_count:
            if masked:
                print(f'write frame: {frame_no} with mask')
            else:
                print(f'write frame: {frame_no}')

    def _adaptive_propainter_batch_size(self, mask, total_frames):
        """
        Choose a safer/faster batch size from mask density + resolution.
        Keeps quality unchanged; only adjusts throughput/memory behavior.
        """
        base = int(max(1, config.PROPAINTER_MAX_LOAD_NUM))
        if total_frames <= 2:
            return min(base, total_frames)

        # Density of inpaint area (0~1)
        try:
            if not _opencv_image_ok(mask):
                density = 0.15
            else:
                non_zero = cv2.countNonZero(mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
                density = float(non_zero) / float(mask.shape[0] * mask.shape[1])
        except Exception:
            density = 0.15

        h, w = self.mask_size
        pixels = h * w

        # Heavier scenes -> smaller chunks; lighter scenes -> larger chunks.
        size_factor = 1.0
        if pixels >= 1920 * 1080:
            size_factor *= 0.65
        elif pixels >= 1280 * 720:
            size_factor *= 0.8

        if density >= 0.20:
            size_factor *= 0.65
        elif density >= 0.10:
            size_factor *= 0.8
        elif density <= 0.04:
            size_factor *= 1.15

        batch = int(max(2, min(base, round(base * size_factor))))
        return min(batch, total_frames)

    def _safe_propainter_inpaint(self, batch_frames, mask):
        """
        Try ProPainter inpaint; on CUDA OOM split batch recursively and retry.
        Quality intent is preserved by keeping the same model/settings.
        """
        if len(batch_frames) <= 0:
            return []
        try:
            return self.video_inpaint.inpaint(batch_frames, mask)
        except Exception as e:
            msg = str(e).lower()
            is_oom = ('out of memory' in msg) or isinstance(e, torch.OutOfMemoryError)
            if (not is_oom) or len(batch_frames) <= 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(batch_frames) // 2
            left = self._safe_propainter_inpaint(batch_frames[:mid], mask)
            right = self._safe_propainter_inpaint(batch_frames[mid:], mask)
            return left + right

    def _apply_mask_feather_blend(self, original_frame, inpainted_frame, mask):
        """
        Feathered compositing: blend inpainted pixels into the original using a
        soft-edge (Gaussian-blurred) mask as alpha. Center of mask = 100% inpainted,
        edges = smooth gradient back to original. Eliminates hard seams.
        """
        radius = int(getattr(config, 'MASK_FEATHER_RADIUS', 0))
        if radius <= 0:
            return inpainted_frame
        if not _opencv_image_ok(original_frame) or not _opencv_image_ok(inpainted_frame):
            return inpainted_frame
        if mask is None:
            return inpainted_frame

        gray_mask = mask
        if len(mask.shape) == 3:
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if not (gray_mask > 0).any():
            return inpainted_frame

        # Edge-only feather:
        # - Keep the core fully inpainted (alpha=1.0) to avoid reintroducing source
        #   pixels/ghosting in the center.
        # - Feather only along a thin boundary ring for seamless transitions.
        gray_bin = (gray_mask > 0).astype('uint8') * 255
        core_kernel = np.ones((3, 3), np.uint8)
        core = cv2.erode(gray_bin, core_kernel, iterations=max(1, radius))
        edge = cv2.subtract(gray_bin, core)
        ksize = max(3, radius * 2 + 1)
        soft_edge = cv2.GaussianBlur(edge.astype('float32') / 255.0, (ksize, ksize), 0)
        soft_edge = np.clip(soft_edge, 0.0, 1.0)
        alpha_map = np.where(core > 0, 1.0, soft_edge)
        alpha = np.expand_dims(alpha_map.astype('float32'), axis=2)
        blended = (inpainted_frame.astype('float32') * alpha +
                   original_frame.astype('float32') * (1.0 - alpha))
        return np.clip(blended, 0, 255).astype('uint8')

    def _apply_temporal_antiflicker(self, current_frame, mask=None):
        """
        Apply a lightweight temporal smoothing pass inside masked regions only.
        This reduces shimmer/flicker between adjacent frames.
        """
        if not getattr(config, 'ENABLE_TEMPORAL_ANTIFLICKER', False):
            if _opencv_image_ok(current_frame):
                self.prev_output_frame = current_frame.copy()
            return current_frame
        if not _opencv_image_ok(current_frame):
            return current_frame
        if self.prev_output_frame is None:
            self.prev_output_frame = current_frame.copy()
            return current_frame
        if mask is None:
            self.prev_output_frame = current_frame.copy()
            return current_frame
        if len(mask.shape) == 3:
            if not _opencv_image_ok(mask):
                self.prev_output_frame = current_frame.copy()
                return current_frame
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if not (mask > 0).any():
            self.prev_output_frame = current_frame.copy()
            return current_frame
        alpha = float(getattr(config, 'ANTIFLICKER_ALPHA', 0.82))
        alpha = max(0.0, min(1.0, alpha))
        # Motion-gated blend: smooth only low-motion pixels in mask.
        # Prevents ghosting/blur on moving edges while still reducing flicker.
        motion_th = float(getattr(config, 'ANTIFLICKER_MOTION_THRESHOLD', 8.0))
        curr_f = current_frame.astype('float32')
        prev_f = self.prev_output_frame.astype('float32')
        blended = cv2.addWeighted(curr_f, alpha, prev_f, 1.0 - alpha, 0.0)
        out = current_frame.copy()
        mask_bool = mask > 0
        try:
            curr_g = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            prev_g = cv2.cvtColor(self.prev_output_frame, cv2.COLOR_BGR2GRAY)
            motion = cv2.absdiff(curr_g, prev_g)
            stable_bool = motion <= motion_th
            apply_bool = np.logical_and(mask_bool, stable_bool)
        except Exception:
            apply_bool = mask_bool
        out[apply_bool] = blended[apply_bool].astype('uint8')
        self.prev_output_frame = out.copy()
        return out

    def _apply_fast_final_check(self, out_frame, mask):
        """
        Fast non-OCR final check:
        - Re-check selected boxes / edited mask for residual text-like edges and temporal jitter
        - Re-inpaint only suspicious frames (targeted, keeps runtime reasonable)
        """
        if not getattr(config, 'ENABLE_FAST_FINAL_CHECK', False):
            return out_frame

        # Safety: never run final-check without a current inpaint mask.
        # Running on full selected boxes causes over-blur.
        if mask is None:
            return out_frame
        if not _opencv_image_ok(out_frame):
            return out_frame
        if len(mask.shape) == 3:
            if not _opencv_image_ok(mask):
                return out_frame
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Constrain check region to current edited area.
        check_mask = mask.copy()
        if self.selected_box_mask is not None:
            check_mask = cv2.bitwise_and(check_mask, self.selected_box_mask)

        # Lightly shrink mask to avoid touching edges outside subtitle pixels.
        try:
            kernel = np.ones((3, 3), np.uint8)
            check_mask = cv2.erode(check_mask, kernel, iterations=1)
        except Exception:
            pass

        area = cv2.countNonZero(check_mask)
        if area < int(getattr(config, 'FINAL_CHECK_MIN_MASK_PIXELS', 100)):
            return out_frame

        gray = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 90, 180)
        edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, check_mask))) / float(max(1, area))

        temporal_delta = 0.0
        if self.prev_final_check_gray is not None:
            diff = cv2.absdiff(gray, self.prev_final_check_gray)
            temporal_delta = float(cv2.mean(diff, mask=check_mask)[0])
        self.prev_final_check_gray = gray

        edge_th = float(getattr(config, 'FINAL_CHECK_EDGE_DENSITY_THRESHOLD', 0.08))
        temp_th = float(getattr(config, 'FINAL_CHECK_TEMPORAL_DELTA_THRESHOLD', 14.0))
        # Motion-safe trigger:
        # - Edge density alone can indicate residual subtitle edges.
        # - Temporal delta alone is noisy on moving clips; require some edge evidence.
        suspicious = (edge_density >= edge_th) or (
            temporal_delta >= temp_th and edge_density >= (edge_th * 0.6)
        )
        if not suspicious:
            return out_frame

        if self.lama_inpaint is None:
            self.lama_inpaint = LamaInpaint()
        try:
            fixed = self.lama_inpaint(out_frame, check_mask)
            fixed = self._apply_mask_feather_blend(out_frame, fixed, check_mask)

            # Optional OCR refine: detect any residual text boxes and re-inpaint them.
            if bool(getattr(config, 'ENABLE_OCR_REFINE', False)):
                try:
                    dt_boxes, _ = self.sub_detector.detect_subtitle(fixed)
                    coords = []
                    if dt_boxes is not None and len(dt_boxes) > 0:
                        coords = self.sub_detector.get_coordinates(dt_boxes.tolist())
                    if coords:
                        pad = int(getattr(config, 'OCR_REFINE_PAD', 2))
                        pad = max(0, min(pad, 20))
                        refine_mask = np.zeros_like(check_mask)
                        for xmin, xmax, ymin, ymax in coords:
                            x1 = max(0, int(xmin) - pad)
                            x2 = min(self.frame_width, int(xmax) + pad)
                            y1 = max(0, int(ymin) - pad)
                            y2 = min(self.frame_height, int(ymax) + pad)
                            cv2.rectangle(refine_mask, (x1, y1), (x2, y2), 255, thickness=-1)
                        # Only refine inside user-selected region (prevents overreach).
                        if self.selected_box_mask is not None:
                            refine_mask = cv2.bitwise_and(refine_mask, self.selected_box_mask)
                        # Union with current check mask
                        union_mask = cv2.bitwise_or(check_mask, refine_mask)
                        if cv2.countNonZero(union_mask) > cv2.countNonZero(check_mask):
                            fixed2 = self.lama_inpaint(fixed, union_mask)
                            fixed2 = self._apply_mask_feather_blend(fixed, fixed2, union_mask)
                            return fixed2
                except Exception:
                    pass
            return fixed
        except Exception:
            return out_frame

    def _build_final_check_frame_set(self, subtitle_frame_dict):
        """
        Build a quality-safe frame set for fast final-check:
        - include detected subtitle frames
        - include +/- margin frames as safety buffer
        """
        if not subtitle_frame_dict:
            return set()
        margin = int(getattr(config, 'FINAL_CHECK_FRAME_MARGIN', 2))
        margin = max(0, margin)
        frame_set = set()
        for idx in subtitle_frame_dict.keys():
            i = int(idx)
            frame_set.add(i)
            for d in range(1, margin + 1):
                if i - d >= 1:
                    frame_set.add(i - d)
                if i + d <= self.frame_count:
                    frame_set.add(i + d)
        return frame_set

    def propainter_mode(self, tbar):
        print('use propainter mode')
        sub_list = self._get_subtitle_map()
        final_check_frames = self._build_final_check_frame_set(sub_list)
        continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        self.subtitle_intervals = continuous_frame_no_list
        scene_div_points = self.sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = self.sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        self.video_inpaint = VideoInpaint(config.PROPAINTER_MAX_LOAD_NUM)
        print('[Processing] start removing subtitles...')
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1
            if self.resume_processed_frames > 0 and index <= self.resume_processed_frames:
                self.update_progress(tbar, increment=1)
                continue
            # 如果当前帧没有水印/文本则直接写
            if index not in sub_list.keys():
                out_frame = self._apply_temporal_antiflicker(frame, None)
                # Quality-safe optimization: final-check only near subtitle ranges
                if index in final_check_frames:
                    out_frame = self._apply_fast_final_check(out_frame, None)
                self._write_output_frame(out_frame)
                if self.gui_mode:
                    self.preview_frame = cv2.hconcat([frame, out_frame])
                self._log_write_frame(index, masked=False)
                self.update_progress(tbar, increment=1)
                continue
            # 如果有水印，判断该帧是不是开头帧
            else:
                # 如果是开头帧，则批推理到尾帧
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # print(f'No 1 Current index: {index}')
                    start_frame_no = index
                    print(f'find start: {start_frame_no}')
                    # 找到结束帧
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 判断当前帧号是不是字幕起始位置
                    # 如果获取的结束帧号不为-1则说明
                    if end_frame_no != -1:
                        print(f'find end: {end_frame_no}')
                        # ************ 读取该区间所有帧 start ************
                        temp_frames = list()
                        # 将头帧加入处理列表
                        temp_frames.append(frame)
                        inner_index = 0
                        # 一直读取到尾帧
                        while index < end_frame_no:
                            ret, frame = self.video_cap.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ 读取该区间所有帧 end ************
                        if len(temp_frames) < 1:
                            # 没有待处理，直接跳过
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = create_mask(self.mask_size, sub_list[index])
                            if self.lama_inpaint is None:
                                self.lama_inpaint = LamaInpaint()
                            inpainted_frame = self.lama_inpaint(frame, single_mask)
                            inpainted_frame = self._apply_mask_feather_blend(frame, inpainted_frame, single_mask)
                            out_frame = self._apply_temporal_antiflicker(inpainted_frame, single_mask)
                            out_frame = self._apply_fast_final_check(out_frame, single_mask)
                            self._write_output_frame(out_frame)
                            if self.gui_mode:
                                self.preview_frame = cv2.hconcat([frame, out_frame])
                            self._log_write_frame(start_frame_no + inner_index, masked=True)
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # 将读取的视频帧分批处理
                            # 1. 获取当前批次使用的mask
                            # Union all detected boxes across the entire interval so
                            # boxes 2, 3, etc. that appear at different frames are all covered.
                            all_interval_coords = []
                            for _fi in range(start_frame_no, end_frame_no + 1):
                                for _coord in sub_list.get(_fi, []):
                                    if _coord not in all_interval_coords:
                                        all_interval_coords.append(_coord)
                            mask = create_mask(self.mask_size, all_interval_coords or sub_list.get(start_frame_no, []))
                            adaptive_batch = self._adaptive_propainter_batch_size(mask, len(temp_frames))
                            self.current_chunk_size = adaptive_batch
                            for batch in batch_generator(temp_frames, adaptive_batch):
                                # Clear VRAM before each batch to prevent fragmentation OOM
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                # 2. 调用批推理
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, all_interval_coords or sub_list.get(start_frame_no, []))
                                    if self.lama_inpaint is None:
                                        self.lama_inpaint = LamaInpaint()
                                    inpainted_frame = self.lama_inpaint(batch[0], single_mask)
                                    inpainted_frame = self._apply_mask_feather_blend(batch[0], inpainted_frame, single_mask)
                                    out_frame = self._apply_temporal_antiflicker(inpainted_frame, single_mask)
                                    out_frame = self._apply_fast_final_check(out_frame, single_mask)
                                    self._write_output_frame(out_frame)
                                    if self.gui_mode:
                                        self.preview_frame = cv2.hconcat([batch[0], out_frame])
                                    self._log_write_frame(start_frame_no + inner_index, masked=True)
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    inpainted_frames = self._safe_propainter_inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        inpainted_frame = self._apply_mask_feather_blend(batch[i], inpainted_frame, mask)
                                        out_frame = self._apply_temporal_antiflicker(inpainted_frame, mask)
                                        out_frame = self._apply_fast_final_check(out_frame, mask)
                                        self._write_output_frame(out_frame)
                                        self._log_write_frame(start_frame_no + inner_index, masked=True)
                                        inner_index += 1
                                        self.update_progress(tbar, increment=1)
                                        if self.gui_mode:
                                            self.preview_frame = cv2.hconcat([batch[i], out_frame])

    def _e2fgvi_inpaint_frames(self, frames, mask, max_load, e2fgvi_inpaint):
        """
        Pure inference helper: inpaint `frames` using `mask` with crossfade batch overlap.
        Returns a list of inpainted frames without any writing or post-processing.

        Multi-batch crossfade eliminates temporal discontinuities at batch boundaries:
          Batch 0: frames [0, M)       → collect [0, M-O), save tail [M-O, M) in buffer
          Batch 1: frames [M-O, 2M-O) → crossfade zone [M-O, M) = linear blend(buf, new)
                                         collect [M, 2M-2O) directly
        OVERLAP = E2FGVI_NEIGHBOR_LENGTH (number of frames that overlap between batches).
        """
        OVERLAP = min(int(getattr(config, 'E2FGVI_NEIGHBOR_LENGTH', 10)), max_load // 4)
        n = len(frames)

        # ── Single-batch fast path ─────────────────────────────────────────────
        if n <= max_load:
            if n == 1:
                if self.lama_inpaint is None:
                    self.lama_inpaint = LamaInpaint()
                return [self.lama_inpaint(frames[0], mask)]
            return e2fgvi_inpaint.inpaint(frames, mask)

        # ── Multi-batch with crossfade ─────────────────────────────────────────
        result = []
        overlap_buffer = None
        is_first = True
        batch_start = 0

        while batch_start < n:
            batch_end = min(batch_start + max_load, n)
            is_last = (batch_end >= n)
            batch = frames[batch_start:batch_end]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(batch) < 2:
                if self.lama_inpaint is None:
                    self.lama_inpaint = LamaInpaint()
                inpainted = [self.lama_inpaint(batch[0], mask)]
            else:
                inpainted = e2fgvi_inpaint.inpaint(batch, mask)

            # Crossfade zone: blend previous tail with new head using a linear ramp
            if not is_first and overlap_buffer is not None:
                actual_ov = min(OVERLAP, len(inpainted))
                for j in range(actual_ov):
                    alpha = (j + 1) / (actual_ov + 1)   # 0 → 1
                    blended = cv2.addWeighted(
                        inpainted[j].astype(np.float32), alpha,
                        overlap_buffer[j].astype(np.float32), 1.0 - alpha,
                        0.0
                    ).astype(np.uint8)
                    result.append(blended)

            # Non-overlap frames unique to this batch
            local_start = OVERLAP if not is_first else 0
            local_end = len(inpainted) if is_last else max(local_start, len(inpainted) - OVERLAP)
            for i in range(local_start, local_end):
                result.append(inpainted[i])

            if not is_last:
                ov_start = max(0, len(inpainted) - OVERLAP)
                overlap_buffer = inpainted[ov_start:]

            batch_start = batch_end - OVERLAP if not is_last else batch_end
            is_first = False

        return result

    def e2fgvi_mode(self, tbar):
        """
        E2FGVI-HQ inpainting mode.

        Workflow per subtitle segment:
          1. Detect subtitle frames (same OCR pipeline as ProPainter)
          2. Build one union mask for all boxes in the segment
          3. Single E2FGVI pass over that mask (batched by max_load); crossfade across batches
          4. Feather blend + temporal anti-flicker post-processing

        Advantages over ProPainter mode:
          - Single end-to-end model (no RAFT chain)
          - No hard chunk-boundary flickering
          - Better fill consistency for YouTube-quality output
        """
        print('use E2FGVI-HQ mode (e2fgvi_hq.py + HQ weights)')
        sub_list = self._get_subtitle_map()
        final_check_frames = self._build_final_check_frame_set(sub_list)
        continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        self.subtitle_intervals = continuous_frame_no_list
        manual_only = bool(getattr(config, 'MANUAL_BOXES_ONLY', False))
        force_scene = bool(getattr(config, 'E2FGVI_FORCE_SCENE_DETECT', False))
        if manual_only and not force_scene:
            scene_div_points = []
            print(
                '[E2FGVI-HQ] Skipping scene-cut scan (MANUAL_BOXES_ONLY; set E2FGVI_FORCE_SCENE_DETECT=True to enable).'
            )
            sys.stdout.flush()
        else:
            print(
                '[E2FGVI-HQ] Scanning for scene cuts — progress bar may stay at 0% until this finishes...'
            )
            sys.stdout.flush()
            scene_div_points = self.sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = self.sub_detector.split_range_by_scene(
            continuous_frame_no_list, scene_div_points)

        e2fgvi_inpaint = E2FGVIInpaint()
        max_load = int(getattr(config, 'E2FGVI_MAX_LOAD_NUM', 80))

        print('[Processing] start removing subtitles with E2FGVI-HQ...')
        index = 0
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            index += 1

            # Resume support: skip already-processed frames
            if self.resume_processed_frames > 0 and index <= self.resume_processed_frames:
                self.update_progress(tbar, increment=1)
                continue

            # No subtitle on this frame → write directly
            if index not in sub_list.keys():
                out_frame = self._apply_temporal_antiflicker(frame, None)
                if index in final_check_frames:
                    out_frame = self._apply_fast_final_check(out_frame, None)
                self._write_output_frame(out_frame)
                if self.gui_mode:
                    self.preview_frame = cv2.hconcat([frame, out_frame])
                self._log_write_frame(index, masked=False)
                self.update_progress(tbar, increment=1)
                continue

            # This frame is the start of a subtitle segment
            if self.is_current_frame_no_start(index, continuous_frame_no_list):
                start_frame_no = index
                print(f'[E2FGVI-HQ] segment start: {start_frame_no}')
                end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                if end_frame_no == -1:
                    continue
                print(f'[E2FGVI-HQ] segment end:   {end_frame_no}')

                # ── Read all frames in this segment ──────────────────────────
                temp_frames = [frame]
                while index < end_frame_no:
                    ret, frame = self.video_cap.read()
                    if not ret:
                        break
                    index += 1
                    temp_frames.append(frame)

                if not temp_frames:
                    continue

                # Single-frame segment → use LaMa (E2FGVI needs ≥2 frames)
                if len(temp_frames) == 1:
                    single_mask = create_mask(self.mask_size, sub_list.get(start_frame_no, []))
                    if self.lama_inpaint is None:
                        self.lama_inpaint = LamaInpaint()
                    inpainted = self.lama_inpaint(temp_frames[0], single_mask)
                    inpainted = self._apply_mask_feather_blend(temp_frames[0], inpainted, single_mask)
                    out_frame = self._apply_temporal_antiflicker(inpainted, single_mask)
                    out_frame = self._apply_fast_final_check(out_frame, single_mask)
                    self._write_output_frame(out_frame)
                    if self.gui_mode:
                        self.preview_frame = cv2.hconcat([temp_frames[0], out_frame])
                    self._log_write_frame(start_frame_no, masked=True)
                    self.update_progress(tbar, increment=1)
                    continue

                # ── Collect unique box coordinates for this segment ──────────
                unique_coords = []
                for fi in range(start_frame_no, end_frame_no + 1):
                    for coord in sub_list.get(fi, []):
                        if coord not in unique_coords:
                            unique_coords.append(coord)
                if not unique_coords:
                    unique_coords = sub_list.get(start_frame_no, [])

                # Union mask: one E2FGVI pass over all boxes (sequential per-box was
                # N× full-frame passes on the GPU — looked "stuck" for minutes).
                union_mask = create_mask(self.mask_size, unique_coords)

                print(
                    f'[E2FGVI-HQ] frames {start_frame_no}-{end_frame_no}: {len(temp_frames)} fr, '
                    f'{len(unique_coords)} box(es) — single union pass (this may take 1–5 min on 8GB GPU)'
                )
                sys.stdout.flush()
                current_frames = self._e2fgvi_inpaint_frames(
                    temp_frames, union_mask, max_load, e2fgvi_inpaint
                )

                # ── Apply post-processing and write ──────────────────────────
                inner_index = 0
                for i, inpainted_frame in enumerate(current_frames):
                    inpainted_frame = self._apply_mask_feather_blend(
                        temp_frames[i], inpainted_frame, union_mask)
                    out_frame = self._apply_temporal_antiflicker(inpainted_frame, union_mask)
                    out_frame = self._apply_fast_final_check(out_frame, union_mask)
                    self._write_output_frame(out_frame)
                    if self.gui_mode:
                        self.preview_frame = cv2.hconcat([temp_frames[i], out_frame])
                    self._log_write_frame(start_frame_no + inner_index, masked=True)
                    inner_index += 1
                    self.update_progress(tbar, increment=1)

    def sttn_mode_with_no_detection(self, tbar):
        """
        使用sttn对选中区域进行重绘，不进行字幕检测
        """
        print('use sttn mode with no detection')
        print('[Processing] start removing subtitles...')
        normalized_sub_areas = self._normalize_sub_areas(self.sub_area)
        if normalized_sub_areas:
            mask_area_coordinates = []
            for ymin, ymax, xmin, xmax in normalized_sub_areas:
                mask_area_coordinates.append((xmin, xmax, ymin, ymax))
        else:
            print('[Info] No subtitle area has been set. Video will be processed in full screen. As a result, the final outcome might be suboptimal.')
            mask_area_coordinates = [(0, self.frame_width, 0, self.frame_height)]
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNVideoInpaint(self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def sttn_mode(self, tbar):
        # 是否跳过字幕帧寻找
        if config.STTN_SKIP_DETECTION:
            # 若跳过则世界使用sttn模式
            self.sttn_mode_with_no_detection(tbar)
        else:
            print('use sttn mode')
            sttn_inpaint = STTNInpaint()
            sub_list = self._get_subtitle_map()
            final_check_frames = self._build_final_check_frame_set(sub_list)
            continuous_frame_no_list = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
            self.subtitle_intervals = continuous_frame_no_list
            print(continuous_frame_no_list)
            continuous_frame_no_list = self.sub_detector.filter_and_merge_intervals(continuous_frame_no_list)
            print(continuous_frame_no_list)
            start_end_map = dict()
            for interval in continuous_frame_no_list:
                start, end = interval
                start_end_map[start] = end
            current_frame_index = 0
            print('[Processing] start removing subtitles...')
            while True:
                ret, frame = self.video_cap.read()
                # 如果读取到为，则结束
                if not ret:
                    break
                current_frame_index += 1
                if self.resume_processed_frames > 0 and current_frame_index <= self.resume_processed_frames:
                    self.update_progress(tbar, increment=1)
                    continue
                # 判断当前帧号是不是字幕区间开始, 如果不是，则直接写
                if current_frame_index not in start_end_map.keys():
                    out_frame = self._apply_temporal_antiflicker(frame, None)
                    if current_frame_index in final_check_frames:
                        out_frame = self._apply_fast_final_check(out_frame, None)
                    self._write_output_frame(out_frame)
                    self._log_write_frame(current_frame_index, masked=False)
                    self.update_progress(tbar, increment=1)
                    if self.gui_mode:
                        self.preview_frame = cv2.hconcat([frame, out_frame])
                # 如果是区间开始，则找到尾巴
                else:
                    start_frame_index = current_frame_index
                    end_frame_index = start_end_map[current_frame_index]
                    print(f'processing frame {start_frame_index} to {end_frame_index}')
                    # 用于存储需要去字幕的视频帧
                    frames_need_inpaint = list()
                    frames_need_inpaint.append(frame)
                    inner_index = 0
                    # 接着往下读，直到读取到尾巴
                    for j in range(end_frame_index - start_frame_index):
                        ret, frame = self.video_cap.read()
                        if not ret:
                            break
                        current_frame_index += 1
                        frames_need_inpaint.append(frame)
                    mask_area_coordinates = []
                    # 1. 获取当前批次的mask坐标全集
                    for mask_index in range(start_frame_index, end_frame_index):
                        if mask_index in sub_list.keys():
                            for area in sub_list[mask_index]:
                                xmin, xmax, ymin, ymax = area
                                # 判断是不是非字幕区域(如果宽大于长，则认为是错误检测)
                                if (ymax - ymin) - (xmax - xmin) > config.THRESHOLD_HEIGHT_WIDTH_DIFFERENCE:
                                    continue
                                if area not in mask_area_coordinates:
                                    mask_area_coordinates.append(area)
                    # 1. 获取当前批次使用的mask
                    mask = create_mask(self.mask_size, mask_area_coordinates)
                    print(f'inpaint with mask: {mask_area_coordinates}')
                    for batch in batch_generator(frames_need_inpaint, config.STTN_MAX_LOAD_NUM):
                        self.current_chunk_size = len(batch)
                        if len(batch) >= 1:
                            inpainted_frames = sttn_inpaint(batch, mask)
                            for i, inpainted_frame in enumerate(inpainted_frames):
                                inpainted_frame = self._apply_mask_feather_blend(batch[i], inpainted_frame, mask)
                                out_frame = self._apply_temporal_antiflicker(inpainted_frame, mask)
                                out_frame = self._apply_fast_final_check(out_frame, mask)
                                self._write_output_frame(out_frame)
                                self._log_write_frame(start_frame_index + inner_index, masked=True)
                                inner_index += 1
                                if self.gui_mode:
                                    self.preview_frame = cv2.hconcat([batch[i], out_frame])
                        self.update_progress(tbar, increment=len(batch))

    def lama_mode(self, tbar):
        print('use lama mode')
        sub_list = self._get_subtitle_map()
        final_check_frames = self._build_final_check_frame_set(sub_list)
        self.subtitle_intervals = self.sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        if self.lama_inpaint is None:
            self.lama_inpaint = LamaInpaint()
        index = 0
        print('[Processing] start removing subtitles...')
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            original_frame = frame
            index += 1
            if self.resume_processed_frames > 0 and index <= self.resume_processed_frames:
                tbar.update(1)
                self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
                self.progress_total = 50 + self.progress_remover
                self._refresh_eta_from_total_progress(stage='removing')
                continue
            if index in sub_list.keys():
                mask = create_mask(self.mask_size, sub_list[index])
                if config.LAMA_SUPER_FAST:
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                else:
                    frame = self.lama_inpaint(frame, mask)
                frame = self._apply_mask_feather_blend(original_frame, frame, mask)
                frame = self._apply_temporal_antiflicker(frame, mask)
                frame = self._apply_fast_final_check(frame, mask)
            else:
                frame = self._apply_temporal_antiflicker(frame, None)
                if index in final_check_frames:
                    frame = self._apply_fast_final_check(frame, None)
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, frame])
            if self.is_picture:
                cv2.imencode(self.ext, frame)[1].tofile(self.video_out_name)
            else:
                self._write_output_frame(frame)
            tbar.update(1)
            self.progress_remover = 100 * float(index) / float(self.frame_count) // 2
            self.progress_total = 50 + self.progress_remover
            self._refresh_eta_from_total_progress(stage='removing')

    def _auto_select_algorithm(self):
        """
        Sample frames from the video, measure background motion complexity,
        and automatically select the best inpainting algorithm for this clip.

        Motion score = median of mean-absolute-difference between sampled frame pairs
        in the non-subtitle region (top 70% of frame, away from typical subtitle band).

        Thresholds:
          score < 3.0  → LaMa   (nearly static: locked-off camera, interrogation room)
          score < 12.0 → E2FGVI (moderate motion: slow pan, handheld interview)
          score ≥ 12.0 → ProPainter (high motion: outdoor chase, fast camera)

        Also auto-tunes E2FGVI/ProPainter parameters to match measured complexity.
        """
        importlib.reload(config)
        cap = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 2:
            cap.release()
            return

        sample_count = min(40, max(10, total // 30))
        step = max(1, total // (sample_count + 1))

        motion_scores = []
        brightness_scores = []
        prev_gray = None

        for i in range(step, total, step):
            if len(motion_scores) >= sample_count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            h = frame.shape[0]
            # Exclude bottom 30% (subtitle area) from motion measurement
            roi = frame[:int(h * 0.70), :]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            brightness_scores.append(float(np.mean(gray)))
            if prev_gray is not None and prev_gray.shape == gray.shape:
                diff = cv2.absdiff(prev_gray, gray)
                motion_scores.append(float(np.mean(diff)))
            prev_gray = gray

        cap.release()

        if not motion_scores:
            return

        median_motion = float(np.median(motion_scores))
        mean_brightness = float(np.mean(brightness_scores)) if brightness_scores else 128.0
        is_dark = mean_brightness < 60.0   # night / low-light scene

        if median_motion < 3.0:
            mode = config.InpaintMode.LAMA
            reason = 'static background'
        elif median_motion < 12.0:
            mode = config.InpaintMode.E2FGVI
            # Match backend/config.py 8GB tuning (do not downgrade to NEIGHBOR=15 here)
            config.E2FGVI_NEIGHBOR_LENGTH = 20
            config.E2FGVI_REF_STRIDE = 3
            config.E2FGVI_MAX_LOAD_NUM = 60
            config.E2FGVI_MAX_CROP_SIDE = 1280
            config.E2FGVI_CROP_MARGIN = 128
            reason = f'moderate motion ({"dark" if is_dark else "normal"} scene)'
        else:
            mode = config.InpaintMode.PROPAINTER
            # Tune ProPainter: more neighbor frames for very high motion
            config.PROPAINTER_NEIGHBOR_LENGTH = 20 if median_motion > 20.0 else 15
            config.PROPAINTER_MAX_LOAD_NUM = 40
            config.PROPAINTER_REF_STRIDE = 10
            reason = 'high motion'

        config.MODE = mode
        print(f'[AutoAlgo] motion={median_motion:.1f}  brightness={mean_brightness:.0f}'
              f'  → {mode.value}  ({reason})')

    def run(self):
        # 记录开始时间
        start_time = time.time()
        self.run_started_at = start_time
        self._eta_last_log_ts = 0.0
        # 重置进度条
        self.progress_total = 0
        self._refresh_eta_from_total_progress(stage='starting')
        self._save_progress_checkpoint(processed_frames=0, stage='starting')
        tbar = tqdm(
            total=int(self.frame_count),
            unit='frame',
            position=0,
            file=sys.__stdout__,
            desc='Subtitle Removing',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
        )
        if self.is_picture:
            if bool(getattr(config, 'MANUAL_BOXES_ONLY', False)):
                manual_map = self._build_manual_only_submap()
                sub_list = manual_map if manual_map is not None else self.sub_detector.find_subtitle_frame_no(
                    sub_remover=self
                )
            else:
                sub_list = self.sub_detector.find_subtitle_frame_no(sub_remover=self)
            self.lama_inpaint = LamaInpaint()
            original_frame = cv2.imread(self.video_path)
            if len(sub_list):
                mask = create_mask(original_frame.shape[0:2], sub_list[1])
                inpainted_frame = self.lama_inpaint(original_frame, mask)
            else:
                inpainted_frame = original_frame
            if self.gui_mode:
                self.preview_frame = cv2.hconcat([original_frame, inpainted_frame])
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_name)
            tbar.update(1)
            self.progress_total = 100
            self._refresh_eta_from_total_progress(stage='removing')
        else:
            # Auto-select algorithm based on measured background motion
            if getattr(config, 'AUTO_SELECT_ALGORITHM', False) and not self.is_picture:
                self._auto_select_algorithm()
            # 精准模式下，获取场景分割的帧号，进一步切割
            if config.MODE == config.InpaintMode.E2FGVI:
                self.e2fgvi_mode(tbar)
            elif config.MODE == config.InpaintMode.PROPAINTER:
                self.propainter_mode(tbar)
            elif config.MODE == config.InpaintMode.STTN:
                self.sttn_mode(tbar)
            else:
                self.lama_mode(tbar)
        self.video_cap.release()
        try:
            self.video_writer.release()
        except Exception:
            pass
        if not self.is_picture:
            if self.enable_chunk_checkpoint:
                ok_merge_parts = self._merge_parts_to_temp_video()
                if not ok_merge_parts:
                    print('[WARN] failed to merge part files, trying existing temp output')
            # Optional post-process deflicker on temporary silent video.
            self._postprocess_deflicker_temp_video()
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
            print(f"[Finished]Subtitle successfully removed, video generated at：{self.video_out_name}")
        else:
            print(f"[Finished]Subtitle successfully removed, picture generated at：{self.video_out_name}")
        print(f'time cost: {round(time.time() - start_time, 2)}s')
        self.isFinished = True
        self.progress_total = 100
        self._refresh_eta_from_total_progress(stage='finished')
        # 处理完成后删除检查点和分片
        if not self.is_picture:
            self._cleanup_parts_and_checkpoint()
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                if platform.system() in ['Windows']:
                    pass
                else:
                    print(f'failed to delete temp file {self.video_temp_file.name}')

    def merge_audio_to_video(self):
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [config.FFMPEG_PATH,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=subprocess.DEVNULL, shell=use_shell)
        except Exception:
            print('fail to extract audio')
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [config.FFMPEG_PATH,
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp.name,
                                       "-vcodec", "libx264" if config.USE_H264 else "copy"]
                if config.USE_H264:
                    audio_merge_command.extend([
                        "-preset", str(getattr(config, "OUTPUT_PRESET", "slow")),
                        "-crf", str(getattr(config, "OUTPUT_CRF", 16)),
                        "-pix_fmt", "yuv420p"
                    ])
                audio_merge_command.extend([
                    "-acodec", "copy",
                    "-loglevel", "error",
                    self.video_out_name
                ])
                try:
                    subprocess.check_output(audio_merge_command, stdin=subprocess.DEVNULL, shell=use_shell)
                except Exception:
                    print('fail to merge audio')
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    if platform.system() in ['Windows']:
                        pass
                    else:
                        print(f'failed to delete temp file {temp.name}')
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_name)
                except IOError as e:
                    print("Unable to copy file. %s" % e)
            self.video_temp_file.close()

    def _postprocess_deflicker_temp_video(self):
        """
        Optional blind deflicker pass after inpainting and before audio remux.
        - Preferred: external FILM / blind-video-deflicker command (if configured)
        - Fallback: ffmpeg deflicker filter
        """
        if self.is_picture:
            return
        if not bool(getattr(config, 'ENABLE_POST_DEFLICKER', False)):
            return
        in_path = self.video_temp_file.name
        if not in_path or not os.path.exists(in_path):
            return

        method = str(getattr(config, 'POST_DEFLICKER_METHOD', 'ffmpeg-deflicker') or 'ffmpeg-deflicker').strip().lower()
        out_tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        out_path = out_tmp.name
        out_tmp.close()
        success = False
        use_shell = True if os.name == "nt" else False

        try:
            # 1) Try external deflicker tool (FILM / blind-video-deflicker)
            if method in ('blind-video-deflicker', 'film', 'external'):
                cmd_tpl = str(getattr(config, 'POST_DEFLICKER_EXTERNAL_CMD', '') or '').strip()
                if cmd_tpl:
                    try:
                        cmd = cmd_tpl.format(input=f'"{in_path}"', output=f'"{out_path}"')
                        print(f'[PostDeflicker] Running external method: {method}')
                        subprocess.check_output(cmd, stdin=subprocess.DEVNULL, shell=True)
                        success = os.path.exists(out_path) and os.path.getsize(out_path) > 0
                    except Exception as e:
                        print(f'[PostDeflicker] external method failed: {e}')
                else:
                    print('[PostDeflicker] external command not configured; skipping external pass')

            # 2) ffmpeg deflicker only when explicitly requested
            if (not success) and method == 'ffmpeg-deflicker':
                strength = int(max(1, min(10, int(getattr(config, 'POST_DEFLICKER_FFMPEG_STRENGTH', 4)))))
                print(f'[PostDeflicker] Running ffmpeg deflicker (strength={strength})')
                ff_cmd = [
                    config.FFMPEG_PATH,
                    "-y",
                    "-i", in_path,
                    "-vf", f"deflicker=s={strength}:m=am",
                    "-vcodec", "libx264" if config.USE_H264 else "copy",
                ]
                if config.USE_H264:
                    ff_cmd.extend([
                        "-preset", str(getattr(config, "OUTPUT_PRESET", "slow")),
                        "-crf", str(getattr(config, "OUTPUT_CRF", 16)),
                        "-pix_fmt", "yuv420p"
                    ])
                ff_cmd.extend([
                    "-an",
                    "-loglevel", "error",
                    out_path
                ])
                subprocess.check_output(ff_cmd, stdin=subprocess.DEVNULL, shell=use_shell)
                success = os.path.exists(out_path) and os.path.getsize(out_path) > 0

            # Replace temp video only on success
            if success:
                try:
                    os.remove(in_path)
                except Exception:
                    pass
                shutil.move(out_path, in_path)
                print('[PostDeflicker] done')
            else:
                print('[PostDeflicker] skipped (no output produced)')
        except Exception as e:
            print(f'[PostDeflicker] failed: {e}')
        finally:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # 1. 提示用户输入视频路径
    video_path = input(f"Please input video or image file path: ").strip()
    # 判断视频路径是不是一个目录，是目录的化，批量处理改目录下的所有视频文件
    # 2. 按以下顺序传入字幕区域
    # sub_area = (ymin, ymax, xmin, xmax)
    # 3. 新建字幕提取对象
    if is_video_or_image(video_path):
        sd = SubtitleRemover(video_path, sub_area=None)
        sd.run()
    else:
        print(f'Invalid video path: {video_path}')
