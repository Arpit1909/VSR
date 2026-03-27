# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2023/4/1 6:07 下午
@FileName: gui.py
@desc: 字幕去除器图形化界面
"""
import os
import sys

# MPS: remove the default memory upper limit so Apple Silicon can use full unified RAM.
# Must be set before torch is first imported (done here, before any torch-importing module).
if sys.platform == 'darwin':
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

# Windows: DPI awareness must be set before PySimpleGUI/Tkinter touches the display
# (calling this only from __main__ is too late — imports may initialize Tk indirectly).
if sys.platform == 'win32':
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import math
import configparser
import PySimpleGUI as sg
import cv2
import numpy as np
import time
import io
import glob
import json
from threading import Thread
import multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend.main
from backend.main import apply_gui_user_settings_overlay, GUI_USER_SETTINGS_PATH, GUI_USER_SETTINGS_KEYS
from backend.tools.common_tools import is_image_file


class _FilteredStream(io.TextIOBase):
    """Suppress known noisy but harmless console lines."""
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self._skip_phrases = [
            'INFO: Could not find files for the given pattern(s).',
        ]

    def write(self, s):
        try:
            text = str(s)
            if any(p in text for p in self._skip_phrases):
                return len(text)
            # Paddle/tqdm sometimes emits this without going through exact phrase match
            if 'Could not find files for the given pattern' in text:
                return len(text)
            return self._wrapped.write(text)
        except Exception:
            return 0

    def flush(self):
        try:
            return self._wrapped.flush()
        except Exception:
            return None


class SubtitleRemoverGUI:
    # Tk PhotoImage + very large JPEG updates often fail on Windows (black preview / Tcl errors).
    _PREVIEW_MAX_W = 1920
    _PREVIEW_MAX_H = 1080

    @classmethod
    def _clamp_preview_wh(cls, w, h):
        w = int(max(1, w))
        h = int(max(1, h))
        scale = min(1.0, cls._PREVIEW_MAX_W / w, cls._PREVIEW_MAX_H / h)
        return max(1, int(w * scale)), max(1, int(h * scale))

    def __init__(self):
        self.font = ('Segoe UI', 10)
        self.theme = 'LightBrown12'
        sg.theme(self.theme)
        self.icon = os.path.join(os.path.dirname(__file__), 'design', 'vsr.ico')
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        self.subtitle_config_file = os.path.join(os.path.dirname(__file__), 'subtitle.ini')
        print(self.screen_width, self.screen_height)
        # Compute the target window size first, then derive preview from it.
        sw, sh = int(self.screen_width), int(self.screen_height)
        self._target_win_w = max(960, min(1480, int(sw * 0.92)))
        self._target_win_h = max(680, min(940, int(sh * 0.90)))
        # Controls below the preview (file bar, timeline, console, sliders, playback,
        # quality two rows, boxes, box list, status). Preview gets the rest.
        self._CONTROLS_HEIGHT = 600
        self.video_preview_width = max(640, int(self._target_win_w * 0.96))
        self.video_preview_height = max(300, self._target_win_h - self._CONTROLS_HEIGHT)
        # 默认组件大小
        self.horizontal_slider_size = (120, 20)
        self.output_size = (100, 8)
        self.progressbar_size = (60, 20)
        if self.screen_width // 2 < 960:
            self.video_preview_width = max(480, int(self._target_win_w * 0.90))
            self.video_preview_height = max(200, self._target_win_h - self._CONTROLS_HEIGHT)
            self.horizontal_slider_size = (60, 20)
            self.output_size = (58, 6)
            self.progressbar_size = (28, 20)
        self.video_preview_width, self.video_preview_height = self._clamp_preview_wh(
            self.video_preview_width, self.video_preview_height
        )
        print(f'[Layout] window target: {self._target_win_w}x{self._target_win_h}, preview: {self.video_preview_width}x{self.video_preview_height}')
        # 字幕提取器布局
        self.layout = None
        # 字幕提取其窗口
        self.window = None
        # 视频路径
        self.video_path = None
        self.video_paths = []
        # 视频cap
        self.video_cap = None
        # 视频的帧率
        self.fps = None
        # 视频的帧数
        self.frame_count = None
        # 视频的宽
        self.frame_width = None
        # 视频的高
        self.frame_height = None
        # 设置字幕区域高宽
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        # 多区域手动框选
        self.subtitle_areas = []
        self.selected_box_index = None
        self.box_undo_stack = []
        self.quality_preset = self._suggest_quality_preset_by_specs()
        # Off = pure E2FGVI/ProPainter/LaMa output — no LaMa second pass (reduces flicker/blur)
        self.final_check_mode = 'Off'
        # Default: two-pass (fast STTN detect + E2FGVI quality) for best results on 8GB-class GPUs.
        self.pass_mode = 'Two-Pass Best'
        self._two_pass_mode = False
        self._max_quality_mode = False
        self.console_visible = True
        self.current_preview_frame = None
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.is_playing = False
        self.last_play_ts = 0.0
        # 字幕提取器
        self.sr = None
        # Merge gui_user_settings.json over config (same as each processing run after reload).
        apply_gui_user_settings_overlay()

    @staticmethod
    def _frame_to_bgr(frame):
        """OpenCV may return grayscale (1ch) or BGRA; GUI assumes 3-channel BGR."""
        if frame is None:
            return None
        if not hasattr(frame, 'shape') or len(frame.shape) < 2:
            return None
        try:
            if frame.size == 0 or int(frame.shape[0]) < 1 or int(frame.shape[1]) < 1:
                return None
        except Exception:
            return None
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        c = frame.shape[2]
        if c == 3:
            return frame
        if c == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if c == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame[:, :, :3].copy()

    @staticmethod
    def _sanitize_cap_int(val, fallback, minimum=1):
        try:
            x = float(val)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(x) or x < minimum:
            return fallback
        return int(x)

    @staticmethod
    def _sanitize_fps(val, fallback=25.0):
        try:
            x = float(val)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(x) or x <= 0:
            return fallback
        return x

    def _safe_display_update(self, bgr_image):
        """Encode preview as PNG and push to sg.Image.
        PNG is required: Tkinter's PhotoImage only supports PNG/GIF natively.
        JPEG bytes cause a silent TclError → black screen when PIL/ImageTk is absent."""
        if self.window is None:
            return
        try:
            ok, buf = cv2.imencode('.png', bgr_image)
            if not ok or buf is None:
                print('[DISPLAY] imencode failed for preview')
                return
            self.window['-DISPLAY-'].update(data=buf.tobytes())
        except Exception as e:
            print(f'[DISPLAY] update failed: {type(e).__name__}: {e}')

    @staticmethod
    def _suggest_quality_preset_by_specs():
        """Pick a smart default preset from detected specs."""
        import platform, multiprocessing
        def _fenv(name, default=0.0):
            try:
                return float(os.environ.get(name, str(default)))
            except Exception:
                return float(default)

        vram = _fenv('VSR_GPU_VRAM_GB', 0.0)
        ram  = _fenv('VSR_SYS_RAM_GB',  0.0)
        cpu  = _fenv('VSR_CPU_THREADS', 0.0)

        # On Mac, env vars from .bat are not set — detect directly
        if platform.system() == 'Darwin':
            if cpu == 0.0:
                cpu = float(multiprocessing.cpu_count())
            if ram == 0.0:
                try:
                    import subprocess
                    out = subprocess.check_output(['sysctl', '-n', 'hw.memsize'],
                                                  stderr=subprocess.DEVNULL).decode().strip()
                    ram = round(int(out) / (1024 ** 3), 1)
                except Exception:
                    pass
            # Apple Silicon: unified memory acts as both RAM and VRAM
            if vram == 0.0:
                vram = ram

        # 8GB+ VRAM: default to Best Quality (E2FGVI); lower tiers stay conservative.
        if vram >= 12 and ram >= 16 and cpu >= 8:
            preset = 'Best Quality'
        elif vram >= 8:
            preset = 'Best Quality'
        elif vram >= 6 and ram >= 16:
            preset = 'Balanced'
        else:
            preset = 'Fast'
        print(f'[AutoPreset] Selected default: {preset} (VRAM={vram}GB, RAM={ram}GB, CPU threads={int(cpu)})')
        return preset

    def run(self):
        # 创建布局
        self._create_layout()
        # Explicit window size avoids Tk packing bugs on Windows where maximize() + a large
        # black Image row makes the preview swallow the whole client area (all-black window).
        sw, sh = int(self.screen_width), int(self.screen_height)
        win_w = max(960, min(1480, int(sw * 0.92)))
        win_h = max(680, min(940, int(sh * 0.90)))
        loc_x = max(0, (sw - win_w) // 2)
        loc_y = max(0, (sh - win_h) // 2)
        self.window = sg.Window(
            title=f'Video Subtitle Remover v{backend.main.config.VERSION}',
            layout=self.layout,
            icon=self.icon,
            resizable=True,
            finalize=True,
            size=(win_w, win_h),
            location=(loc_x, loc_y),
        )
        # Maximize only when opted in — often triggers the all-black layout on Win10/11 + DPI.
        _mx = os.environ.get('VSR_START_MAXIMIZED', '0').strip().lower()
        if _mx in ('1', 'true', 'yes', 'on'):
            try:
                self.window.maximize()
            except Exception:
                pass
        # Prevent the Tk Label from auto-expanding when image data is loaded.
        # pack_propagate(False) on the label only; the parent must stay
        # propagation-enabled so _resize_preview_to_window / set_size works.
        try:
            display_widget = self.window['-DISPLAY-'].Widget
            display_widget.pack_propagate(False)
        except Exception:
            pass
        try:
            self.window.TKroot.update_idletasks()
            self._resize_preview_to_window()
        except Exception:
            pass
        self._bind_mouse_events()
        self.window.bind('<space>', '-SPACE-')
        self.window.bind('<Control-z>', '-UNDO-BOX-')
        self.window.bind('<Left>', '-LEFT-')
        self.window.bind('<Right>', '-RIGHT-')
        self.window.bind('<Shift-Left>', '-SHIFT-LEFT-')
        self.window.bind('<Shift-Right>', '-SHIFT-RIGHT-')
        self.window.bind('<Configure>', '-RESIZE-')
        self.window.bind('<Delete>', '-DEL-KEY-')
        self._set_status('Ready')
        # Schedule a resize after Tkinter has fully processed the maximize geometry.
        # Direct calls before the event loop starts read pre-maximize dimensions;
        # after(300) fires once the event loop is running and winfo_width/height
        # reflect the real maximized window size.
        try:
            self.window.TKroot.after(300, self._resize_preview_to_window)
        except Exception:
            pass
        while True:
            try:
                event, values = self.window.read(timeout=10)
            except Exception as e:
                print(f'[GUI] window.read failed: {type(e).__name__}: {e}')
                import traceback
                traceback.print_exc()
                break
            try:
                # 处理【打开】事件
                self._file_event_handler(event, values)
                # 处理【滑动】事件
                self._slide_event_handler(event, values)
                # 处理【运行】事件
                self._run_event_handler(event, values)
                # 处理鼠标框选事件
                self._mouse_event_handler(event, values)
                # 处理播放
                self._playback_tick(values)
                self._frame_nav_event_handler(event, values)
                # 处理窗口尺寸变化
                if event == '-RESIZE-':
                    self._resize_preview_to_window()
                # 如果关闭软件，退出
                if event == sg.WIN_CLOSED:
                    break
                # 更新进度条
                if self.sr is not None:
                    self.window['-PROG-'].update(self.sr.progress_total)
                    chunk_sz = getattr(self.sr, 'current_chunk_size', 0)
                    self.window['-CHUNK-INFO-'].update(f'Chunk: {chunk_sz}' if chunk_sz else 'Chunk: -')
                    elapsed_h = str(getattr(self.sr, 'elapsed_human', '00:00') or '00:00')
                    eta_h = str(getattr(self.sr, 'eta_human', '--:--') or '--:--')
                    eta_stage = str(getattr(self.sr, 'eta_stage', 'processing') or 'processing')
                    self.window['-TIME-INFO-'].update(
                        f'Elapsed: {elapsed_h} | ETA: {eta_h} | Stage: {eta_stage}'
                    )
                    resume_frame = int(getattr(self.sr, 'resume_start_frame', 0) or 0)
                    merged_parts = int(getattr(self.sr, 'merged_parts_count', 0) or 0)
                    if not self._two_pass_mode:
                        self.window['-PIPELINE-INFO-'].update(
                            f'Pass: {self.pass_mode} | Resume: frame {resume_frame} | Merged parts: {merged_parts}'
                        )
                    intervals = getattr(self.sr, 'subtitle_intervals', None)
                    if intervals:
                        self._draw_subtitle_timeline(intervals)
                    if self.sr.preview_frame is not None:
                        self._safe_display_update(self._img_resize(self.sr.preview_frame))
                    if self.sr.isFinished:
                        # 1) 打开修改字幕滑块区域按钮
                        self.window['-Y-SLIDER-'].update(disabled=False)
                        self.window['-X-SLIDER-'].update(disabled=False)
                        self.window['-Y-SLIDER-H-'].update(disabled=False)
                        self.window['-X-SLIDER-W-'].update(disabled=False)
                        # 2) 打开【运行】、【打开】和【识别语言】按钮
                        self.window['-RUN-'].update(disabled=False)
                        self.window['-QUALITY-'].update(disabled=False)
                        self.window['-ANTIFLICKER-'].update(disabled=False)
                        self.window['-FINAL-CHECK-'].update(disabled=False)
                        self.window['-PASS-MODE-'].update(disabled=False)
                        self.window['-ADD-BOX-'].update(disabled=False)
                        self.window['-DEL-BOX-'].update(disabled=False)
                        self.window['-CLEAR-BOXES-'].update(disabled=False)
                        self.window['-BOX-LIST-'].update(disabled=False)
                        self.window['-FILE-'].update(disabled=False)
                        self.window['-FILE_BTN-'].update(disabled=False)
                        self.window['-ADV-SETTINGS-'].update(disabled=False)
                        self.window['-CHUNK-INFO-'].update('Chunk: -')
                        self.window['-PIPELINE-INFO-'].update(
                            f'Pass: {self.pass_mode} | Resume: frame 0 | Merged parts: 0'
                        )
                        self.window['-TIME-INFO-'].update('Elapsed: 00:00 | ETA: --:-- | Stage: idle')
                        self._set_status('Ready')
                        self.sr = None
                    if len(self.video_paths) >= 1:
                        # 1) 关闭修改字幕滑块区域按钮
                        self.window['-Y-SLIDER-'].update(disabled=True)
                        self.window['-X-SLIDER-'].update(disabled=True)
                        self.window['-Y-SLIDER-H-'].update(disabled=True)
                        self.window['-X-SLIDER-W-'].update(disabled=True)
                        # 2) 关闭【运行】、【打开】和【识别语言】按钮
                        self.window['-RUN-'].update(disabled=True)
                        self.window['-QUALITY-'].update(disabled=True)
                        self.window['-ANTIFLICKER-'].update(disabled=True)
                        self.window['-FINAL-CHECK-'].update(disabled=True)
                        self.window['-PASS-MODE-'].update(disabled=True)
                        self.window['-ADD-BOX-'].update(disabled=True)
                        self.window['-DEL-BOX-'].update(disabled=True)
                        self.window['-CLEAR-BOXES-'].update(disabled=True)
                        self.window['-BOX-LIST-'].update(disabled=True)
                        self.window['-FILE-'].update(disabled=True)
                        self.window['-FILE_BTN-'].update(disabled=True)
                        self.window['-ADV-SETTINGS-'].update(disabled=True)
            except Exception as e:
                print(f'[GUI loop] {type(e).__name__}: {e}')
                import traceback
                traceback.print_exc()
                try:
                    self._set_status('ERROR', str(e)[:200])
                except Exception:
                    pass

    def _create_layout(self):
        """
        创建字幕提取器布局
        """
        garbage = os.path.join(os.path.dirname(__file__), 'output')
        if os.path.exists(garbage):
            import shutil
            shutil.rmtree(garbage, True)
        self.layout = [
            # 显示视频预览
            # NO expand_x / expand_y — both cause Tkinter's geometry manager to override
            # set_size(), stretching the black label to fill the entire window and making
            # everything appear all-black after maximize.  Size is managed exclusively
            # via _resize_preview_to_window() → set_size().
            [sg.Image(size=(self.video_preview_width, self.video_preview_height), background_color='black',
                      key='-DISPLAY-')],
            # 打开按钮 + 快进快退条
            [sg.Input(key='-FILE-', visible=False, enable_events=True),
             sg.FilesBrowse(button_text='Open', file_types=((
                            'All Files', '*.*'), ('mp4', '*.mp4'),
                            ('flv', '*.flv'),
                            ('wmv', '*.wmv'),
                            ('avi', '*.avi')),
                            key='-FILE_BTN-', size=(10, 1), font=self.font),
             sg.Slider(size=self.horizontal_slider_size, range=(1, 1), key='-SLIDER-', orientation='h',
                       enable_events=True, font=self.font,
                       disable_number_display=True, expand_x=True),
             sg.Text('Frame: 0 / 0', key='-FRAME-INFO-', font=self.font),
             ],
            [sg.Graph(canvas_size=(int(self.video_preview_width * 0.9), 22),
                      graph_bottom_left=(0, 0),
                      graph_top_right=(1000, 22),
                      background_color='#222222',
                      key='-TL-GRAPH-',
                      expand_x=True)],
            # 控制区（专业化分组）
            [sg.Frame(
                title='Console',
                font=self.font,
                key='-CONSOLE-FRAME-',
                layout=[[sg.Output(size=self.output_size, font=self.font, expand_x=True)]],
                expand_x=True
             ),
             sg.Frame(title='Vertical Region', font=self.font, key='-FRAME1-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 12),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((6, 6), (4, 4)),
                           default_value=0, key='-Y-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 12),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((6, 6), (4, 4)),
                           default_value=0, key='-Y-SLIDER-H-'),
             ]], pad=((8, 6), (0, 0))),
             sg.Frame(title='Horizontal Region', font=self.font, key='-FRAME2-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 12),
                           disable_number_display=True,
                           pad=((6, 6), (4, 4)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 12),
                           disable_number_display=True,
                           pad=((6, 6), (4, 4)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-W-'),
             ]], pad=((8, 6), (0, 0)))
             ],
            [sg.Frame(
                title='Playback',
                font=self.font,
                layout=[[
                    sg.Button(button_text='Run', key='-RUN-', font=self.font, size=(8, 1)),
                    sg.Button(button_text='Resume Last', key='-RESUME-LAST-', font=self.font, size=(10, 1)),
                    sg.Button(button_text='Play', key='-PLAY-', font=self.font, size=(6, 1)),
                    sg.Button(button_text='Console', key='-TOGGLE-CONSOLE-', font=self.font, size=(8, 1)),
                    sg.Button(button_text='Reset', key='-RESET-LAYOUT-', font=self.font, size=(6, 1)),
                    sg.Text('Chunk: -', key='-CHUNK-INFO-', font=self.font),
                    sg.ProgressBar(100, orientation='h', size=self.progressbar_size, key='-PROG-', auto_size_text=True, expand_x=True)
                ]],
                expand_x=True
             ),
             sg.Frame(
                title='Quality',
                font=self.font,
                layout=[
                    [
                        sg.Text('Preset:', font=self.font),
                        sg.Combo(values=['Fast', 'Balanced', 'Best Quality'],
                                 default_value=self.quality_preset,
                                 readonly=True,
                                 key='-QUALITY-',
                                 size=(12, 1),
                                 font=self.font),
                        sg.Button(button_text='Recommended', key='-RECOMMENDED-',
                                  font=self.font, size=(12, 1),
                                  tooltip='Best Speed+Quality (Recommended)\nSets: Two-Pass Best + Best Quality + Anti-Flicker ON + Check Fast'),
                        sg.Button(button_text='BQ Optimized', key='-BQ-OPT-',
                                  font=self.font, size=(12, 1),
                                  tooltip='Best Quality (Optimized)\nSets: Two-Pass Best + Best Quality + Anti-Flicker ON + Check Fast\nEnables light OCR refine + sharp blending (best quality without big slowdown)'),
                        sg.Button(button_text='Max Quality', key='-MAX-QUALITY-',
                                  font=self.font, size=(10, 1),
                                  tooltip='Max Quality (Slow)\nSets: Two-Pass Best + Best Quality + Anti-Flicker ON + Check Strict\nAlso enables stronger refine + sharper blending (slower)'),
                        sg.Checkbox('Anti-Flicker', key='-ANTIFLICKER-', default=True, font=self.font),
                        sg.Text('Check:', font=self.font),
                        sg.Combo(values=['Off', 'Fast', 'Strict'],
                                 default_value=self.final_check_mode,
                                 readonly=True,
                                 key='-FINAL-CHECK-',
                                 size=(7, 1),
                                 font=self.font,
                                 tooltip='Off: best consistency (default)\nFast: light LaMa fix on suspicious frames\nStrict: stronger cleanup if text remains'),
                    ],
                    [
                        sg.Text('Pass:', font=self.font),
                        sg.Combo(values=['Preview Fast', 'Final Quality', 'Two-Pass Best'],
                                 default_value=self.pass_mode,
                                 readonly=True,
                                 key='-PASS-MODE-',
                                 size=(18, 1),
                                 font=self.font,
                                 tooltip='Preview Fast: quick draft\nFinal Quality: single production pass\nTwo-Pass Best: fast detect → quality inpaint (best results)'),
                        sg.Button(
                            button_text='Settings',
                            key='-ADV-SETTINGS-',
                            font=self.font,
                            size=(12, 1),
                            tooltip='Inpaint overrides: E2FGVI, feather, pad, antiflicker, manual-OCR, etc.\n'
                                    'Saved to gui_user_settings.json (overrides backend\\config.py on each Run).',
                        ),
                        sg.Text(
                            '(E2FGVI / feather / pad …)',
                            font=('Segoe UI', 9),
                            text_color='#666666',
                        ),
                    ],
                ]
             )],
            [sg.Frame(
                title='Boxes',
                font=self.font,
                layout=[[
                    sg.Button(button_text='Add Box', key='-ADD-BOX-', font=self.font, size=(9, 1)),
                    sg.Button(button_text='Delete Box', key='-DEL-BOX-', font=self.font, size=(10, 1)),
                    sg.Button(button_text='Undo', key='-UNDO-BOX-', font=self.font, size=(6, 1)),
                    sg.Button(button_text='Clear All', key='-CLEAR-BOXES-', font=self.font, size=(8, 1)),
                    sg.Text('Boxes: 0', key='-BOX-COUNT-', font=self.font),
                ]],
                expand_x=True
             )],
            [sg.Frame(
                title='Box List',
                font=self.font,
                layout=[[sg.Listbox(values=[], size=(120, 3), key='-BOX-LIST-', enable_events=True, font=self.font, expand_x=True)]],
                expand_x=True
             )],
            [sg.Text('Status: Ready', key='-STATUS-', font=self.font, text_color='#9CDC7C')],
            [sg.Text('Pass: - | Resume: frame 0 | Merged parts: 0', key='-PIPELINE-INFO-', font=self.font, text_color='#666666')],
            [sg.Text('Elapsed: 00:00 | ETA: --:-- | Stage: idle', key='-TIME-INFO-', font=self.font, text_color='#666666')],
        ]

    def _file_event_handler(self, event, values):
        """
        当点击打开按钮时：
        1）打开视频文件，将画布显示视频帧
        2）获取视频信息，初始化进度条滑块范围
        """
        if event != '-FILE-':
            return
        try:
            self.video_paths = values.get('-FILE-', '').split(';')
            self.video_path = (self.video_paths[0] or '').strip()
            if not self.video_path:
                return
            if self.video_cap is not None:
                try:
                    self.video_cap.release()
                except Exception:
                    pass
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                self._set_status('ERROR', 'Could not open video')
                sg.popup_error(
                    'Could not open this file as a video. Try MP4 (H.264) or re-encode the file.',
                    title='Open video failed',
                )
                return
            ret, frame = self.video_cap.read()
            if not ret:
                self._set_status('ERROR', 'Could not read first frame')
                sg.popup_error('Could not read the first frame from this file.', title='Read failed')
                return
            frame = self._frame_to_bgr(frame)
            if frame is None:
                self._set_status('ERROR', 'Unsupported frame format')
                sg.popup_error('Unsupported pixel format (could not convert to BGR).', title='Video format')
                return
            for video in self.video_paths:
                print(f"Open Video Success：{video}")
            fh, fw = int(frame.shape[0]), int(frame.shape[1])
            self.frame_count = self._sanitize_cap_int(
                self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT), fallback=1, minimum=1
            )
            self.frame_height = self._sanitize_cap_int(
                self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT), fallback=fh, minimum=1
            )
            self.frame_width = self._sanitize_cap_int(
                self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH), fallback=fw, minimum=1
            )
            self.fps = self._sanitize_fps(self.video_cap.get(cv2.CAP_PROP_FPS))
            self.window['-SLIDER-'].update(range=(1, max(1, self.frame_count)))
            self.window['-SLIDER-'].update(1)
            self._update_frame_info(1)
            y_p, h_p, x_p, w_p = self.parse_subtitle_config()
            fh_f = float(self.frame_height)
            fw_f = float(self.frame_width)
            y = int(min(max(0, fh_f * y_p), max(0, self.frame_height - 1)))
            h = int(min(max(1, fh_f * h_p), max(1, self.frame_height - y)))
            x = int(min(max(0, fw_f * x_p), max(0, self.frame_width - 1)))
            w = int(min(max(1, fw_f * w_p), max(1, self.frame_width - x)))
            self.window['-Y-SLIDER-'].update(range=(0, max(1, self.frame_height - 1)), disabled=False)
            self.window['-Y-SLIDER-'].update(y)
            self.window['-X-SLIDER-'].update(range=(0, max(1, self.frame_width - 1)), disabled=False)
            self.window['-X-SLIDER-'].update(x)
            yh_max = max(1, self.frame_height - y)
            self.window['-Y-SLIDER-H-'].update(range=(0, yh_max))
            self.window['-Y-SLIDER-H-'].update(min(h, yh_max))
            xw_max = max(1, self.frame_width - x)
            self.window['-X-SLIDER-W-'].update(range=(0, xw_max))
            self.window['-X-SLIDER-W-'].update(min(w, xw_max))
            self.subtitle_areas = []
            self.selected_box_index = None
            self.box_undo_stack = []
            self._refresh_box_count()
            self._refresh_box_list()
            try:
                self.window['-TL-GRAPH-'].erase()
            except Exception:
                pass
            self._update_preview(frame, (y, h, x, w))
            self._set_status('Ready', os.path.basename(self.video_path))
        except Exception as e:
            self._set_status('ERROR', str(e))
            print(f'[Open video] {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            sg.popup_error(
                f'Failed while loading the video:\n\n{e}\n\nSee the console for the full traceback.',
                title='Video load error',
            )

    def __disable_button(self):
        # 1) 禁止修改字幕滑块区域
        self.window['-Y-SLIDER-'].update(disabled=True)
        self.window['-X-SLIDER-'].update(disabled=True)
        self.window['-Y-SLIDER-H-'].update(disabled=True)
        self.window['-X-SLIDER-W-'].update(disabled=True)
        # 2) 禁止再次点击【运行】、【打开】和【识别语言】按钮
        self.window['-RUN-'].update(disabled=True)
        self.window['-RESUME-LAST-'].update(disabled=True)
        self.window['-PLAY-'].update(disabled=True)
        self.window['-QUALITY-'].update(disabled=True)
        self.window['-ANTIFLICKER-'].update(disabled=True)
        self.window['-FINAL-CHECK-'].update(disabled=True)
        self.window['-PASS-MODE-'].update(disabled=True)
        self.window['-ADD-BOX-'].update(disabled=True)
        self.window['-DEL-BOX-'].update(disabled=True)
        self.window['-UNDO-BOX-'].update(disabled=True)
        self.window['-CLEAR-BOXES-'].update(disabled=True)
        self.window['-BOX-LIST-'].update(disabled=True)
        self.window['-FILE-'].update(disabled=True)
        self.window['-FILE_BTN-'].update(disabled=True)
        self.window['-ADV-SETTINGS-'].update(disabled=True)

    def __enable_button(self):
        self.window['-Y-SLIDER-'].update(disabled=False)
        self.window['-X-SLIDER-'].update(disabled=False)
        self.window['-Y-SLIDER-H-'].update(disabled=False)
        self.window['-X-SLIDER-W-'].update(disabled=False)
        self.window['-RUN-'].update(disabled=False)
        self.window['-RESUME-LAST-'].update(disabled=False)
        self.window['-PLAY-'].update(disabled=False)
        self.window['-QUALITY-'].update(disabled=False)
        self.window['-ANTIFLICKER-'].update(disabled=False)
        self.window['-FINAL-CHECK-'].update(disabled=False)
        self.window['-PASS-MODE-'].update(disabled=False)
        self.window['-ADD-BOX-'].update(disabled=False)
        self.window['-DEL-BOX-'].update(disabled=False)
        self.window['-UNDO-BOX-'].update(disabled=False)
        self.window['-CLEAR-BOXES-'].update(disabled=False)
        self.window['-BOX-LIST-'].update(disabled=False)
        self.window['-FILE-'].update(disabled=False)
        self.window['-FILE_BTN-'].update(disabled=False)
        self.window['-ADV-SETTINGS-'].update(disabled=False)

    def _set_play_state(self, playing):
        self.is_playing = bool(playing)
        if self.window is not None:
            self.window['-PLAY-'].update(text='Pause' if self.is_playing else 'Play')
        self.last_play_ts = time.time()

    def _set_status(self, state, detail=''):
        if self.window is None:
            return
        state_upper = str(state).strip().upper()
        if state_upper == 'READY':
            color = '#9CDC7C'
            msg = 'Status: Ready'
        elif state_upper == 'PROCESSING':
            color = '#F5D76E'
            msg = 'Status: Processing'
        elif state_upper == 'ERROR':
            color = '#FF7F7F'
            msg = 'Status: Error'
        else:
            color = '#D0D0D0'
            msg = f'Status: {state}'
        if detail:
            msg = f'{msg} - {detail}'
        self.window['-STATUS-'].update(value=msg, text_color=color)

    def _playback_tick(self, values):
        if not self.is_playing:
            return
        if self.video_cap is None or not self.video_cap.isOpened():
            self._set_play_state(False)
            return
        if not self.fps or self.fps <= 0:
            self.fps = 25.0
        now = time.time()
        if now - self.last_play_ts < (1.0 / float(self.fps)):
            return
        self.last_play_ts = now
        cur = int(values.get('-SLIDER-', 1))
        if self.frame_count is None or cur >= int(self.frame_count):
            self._set_play_state(False)
            return
        next_frame_no = cur + 1
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_no)
        ret, frame = self.video_cap.read()
        if not ret:
            self._set_play_state(False)
            return
        self.window['-SLIDER-'].update(next_frame_no)
        self._update_frame_info(next_frame_no)
        yy = int(values['-Y-SLIDER-'])
        xx = int(values['-X-SLIDER-'])
        self.window['-Y-SLIDER-H-'].update(range=(0, max(1, int(self.frame_height) - yy)))
        self.window['-X-SLIDER-W-'].update(range=(0, max(1, int(self.frame_width) - xx)))
        y = int(values['-Y-SLIDER-'])
        h = int(values['-Y-SLIDER-H-'])
        x = int(values['-X-SLIDER-'])
        w = int(values['-X-SLIDER-W-'])
        self._update_preview(frame, (y, h, x, w))

    def _update_frame_info(self, frame_no):
        if self.window is None:
            return
        total = int(self.frame_count) if self.frame_count else 0
        cur = int(frame_no) if frame_no else 0
        self.window['-FRAME-INFO-'].update(f'Frame: {cur} / {total}')

    def _draw_subtitle_timeline(self, intervals):
        if self.window is None or self.frame_count is None or self.frame_count <= 0:
            return
        graph = self.window['-TL-GRAPH-']
        try:
            graph.erase()
            # baseline
            graph.draw_rectangle((0, 0), (1000, 22), fill_color='#1c1c1c', line_color='#1c1c1c')
            for start, end in intervals:
                x0 = int(max(0, min(1000, (start / self.frame_count) * 1000)))
                x1 = int(max(0, min(1000, (end / self.frame_count) * 1000)))
                if x1 <= x0:
                    x1 = min(1000, x0 + 1)
                graph.draw_rectangle((x0, 2), (x1, 20), fill_color='#e0b34d', line_color='#e0b34d')
        except Exception:
            pass

    def _step_to_frame(self, target_frame, values):
        if self.video_cap is None or not self.video_cap.isOpened():
            return
        if self.frame_count is None:
            return
        frame_no = max(1, min(int(target_frame), int(self.frame_count)))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self.video_cap.read()
        if not ret:
            return
        self.window['-SLIDER-'].update(frame_no)
        self._update_frame_info(frame_no)
        yy = int(values['-Y-SLIDER-'])
        xx = int(values['-X-SLIDER-'])
        self.window['-Y-SLIDER-H-'].update(range=(0, max(1, int(self.frame_height) - yy)))
        self.window['-X-SLIDER-W-'].update(range=(0, max(1, int(self.frame_width) - xx)))
        y = int(values['-Y-SLIDER-'])
        h = int(values['-Y-SLIDER-H-'])
        x = int(values['-X-SLIDER-'])
        w = int(values['-X-SLIDER-W-'])
        self._update_preview(frame, (y, h, x, w))

    def _frame_nav_event_handler(self, event, values):
        if event not in ('-LEFT-', '-RIGHT-', '-SHIFT-LEFT-', '-SHIFT-RIGHT-'):
            return
        if self.video_cap is None:
            return
        self._set_play_state(False)
        cur = int(values.get('-SLIDER-', 1))
        if event == '-LEFT-':
            self._step_to_frame(cur - 1, values)
        elif event == '-RIGHT-':
            self._step_to_frame(cur + 1, values)
        elif event == '-SHIFT-LEFT-':
            self._step_to_frame(cur - 10, values)
        elif event == '-SHIFT-RIGHT-':
            self._step_to_frame(cur + 10, values)

    def _resize_preview_to_window(self):
        if self.window is None:
            return
        try:
            win_w, win_h = self.window.size
            if win_w is None or win_h is None:
                return
            win_w, win_h = int(win_w), int(win_h)
            new_w = max(640, min(int(win_w * 0.96), self._PREVIEW_MAX_W))
            new_h = max(300, min(win_h - self._CONTROLS_HEIGHT, self._PREVIEW_MAX_H))
            new_w, new_h = self._clamp_preview_wh(new_w, new_h)
            if new_w == self.video_preview_width and new_h == self.video_preview_height:
                return
            self.video_preview_width = new_w
            self.video_preview_height = new_h
            try:
                self.window['-DISPLAY-'].set_size((new_w, new_h))
                display_widget = self.window['-DISPLAY-'].Widget
                display_widget.configure(width=new_w, height=new_h)
                self.window.TKroot.update_idletasks()
            except Exception:
                pass
            if self.current_preview_frame is not None:
                y = int(self.window['-Y-SLIDER-'].get())
                h = int(self.window['-Y-SLIDER-H-'].get())
                x = int(self.window['-X-SLIDER-'].get())
                w = int(self.window['-X-SLIDER-W-'].get())
                self._update_preview(self.current_preview_frame.copy(), (y, h, x, w))
        except Exception:
            return

    def _reset_layout(self):
        self._set_play_state(False)
        self.console_visible = True
        try:
            self.window['-CONSOLE-FRAME-'].update(visible=True)
            self.window['-TOGGLE-CONSOLE-'].update(text='Hide Console')
        except Exception:
            pass
        self._resize_preview_to_window()
        try:
            if self.current_preview_frame is not None:
                y = int(self.window['-Y-SLIDER-'].get())
                h = int(self.window['-Y-SLIDER-H-'].get())
                x = int(self.window['-X-SLIDER-'].get())
                w = int(self.window['-X-SLIDER-W-'].get())
                self._update_preview(self.current_preview_frame.copy(), (y, h, x, w))
        except Exception:
            pass
        self._set_status('READY', 'Layout reset')

    @staticmethod
    def _apply_quality_preset(preset_name):
        """
        Auto-hardware profile:
          - 8GB class: stable quality without frequent OOM
          - 12GB class: higher quality / larger temporal context
          - 16GB+ class: max quality settings
        """
        cfg = backend.main.config
        vram_gb = 0.0
        ram_gb = 0.0
        try:
            if backend.main.torch.cuda.is_available():
                vram_gb = backend.main.torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            vram_gb = 0.0
        # Prefer .bat detected hardware values when available
        try:
            env_vram = float(os.environ.get('VSR_GPU_VRAM_GB', '0'))
            if env_vram > 0:
                vram_gb = env_vram
        except Exception:
            pass
        try:
            ram_gb = float(os.environ.get('VSR_SYS_RAM_GB', '0'))
        except Exception:
            ram_gb = 0.0

        if vram_gb >= 15.0:
            hw_profile = '16GB+'
        elif vram_gb >= 11.0:
            hw_profile = '12GB'
        else:
            hw_profile = '8GB'
        low_ram = ram_gb > 0 and ram_gb < 16.0

        cfg.USE_H264 = True
        cfg.STTN_SKIP_DETECTION = False

        # Best settings possible per preset for each VRAM class
        if preset_name == 'Fast':
            cfg.MODE = cfg.InpaintMode.STTN
            cfg.STTN_SKIP_DETECTION = True
            cfg.STTN_MAX_LOAD_NUM = 30 if hw_profile == '8GB' else (40 if hw_profile == '12GB' else 50)
            cfg.PROPAINTER_MAX_LOAD_NUM = 4 if hw_profile == '8GB' else (40 if hw_profile == '12GB' else 60)
            if low_ram:
                cfg.STTN_MAX_LOAD_NUM = max(20, int(cfg.STTN_MAX_LOAD_NUM * 0.8))
                cfg.PROPAINTER_MAX_LOAD_NUM = max(2, int(cfg.PROPAINTER_MAX_LOAD_NUM * 0.8))
            cfg.OUTPUT_CRF = 22
            cfg.OUTPUT_PRESET = 'faster'
            cfg.ENABLE_TEMPORAL_ANTIFLICKER = False
            cfg.MASK_FEATHER_RADIUS = 0
            cfg.PROPAINTER_NEIGHBOR_LENGTH = 10
            cfg.PROPAINTER_REF_STRIDE = 10
            cfg.PROPAINTER_RAFT_ITER = 15
            print(f'[Preset] Fast applied (auto profile: {hw_profile})')
        elif preset_name == 'Balanced':
            cfg.MODE = cfg.InpaintMode.PROPAINTER
            cfg.STTN_MAX_LOAD_NUM = 40 if hw_profile == '8GB' else (55 if hw_profile == '12GB' else 70)
            cfg.PROPAINTER_MAX_LOAD_NUM = 6 if hw_profile == '8GB' else (48 if hw_profile == '12GB' else 70)
            if low_ram:
                cfg.STTN_MAX_LOAD_NUM = max(28, int(cfg.STTN_MAX_LOAD_NUM * 0.8))
                cfg.PROPAINTER_MAX_LOAD_NUM = max(2, int(cfg.PROPAINTER_MAX_LOAD_NUM * 0.8))
            cfg.OUTPUT_CRF = 18 if hw_profile == '16GB+' else 19
            cfg.OUTPUT_PRESET = 'medium'
            cfg.ENABLE_TEMPORAL_ANTIFLICKER = True
            cfg.ANTIFLICKER_ALPHA = 0.97
            cfg.SUBTITLE_AREA_DEVIATION_PIXEL = 20
            cfg.MASK_FEATHER_RADIUS = 3
            cfg.PROPAINTER_NEIGHBOR_LENGTH = 10
            cfg.PROPAINTER_REF_STRIDE = 10
            cfg.PROPAINTER_RAFT_ITER = 15
            print(f'[Preset] Balanced applied (auto profile: {hw_profile})')
        else:
            # Best Quality uses E2FGVI: joint flow+inpainting, no chunk-boundary flicker,
            # better temporal consistency than ProPainter on subtitle bands.
            cfg.MODE = cfg.InpaintMode.E2FGVI
            cfg.STTN_MAX_LOAD_NUM = 40 if hw_profile == '8GB' else (55 if hw_profile == '12GB' else 70)
            if low_ram:
                cfg.STTN_MAX_LOAD_NUM = max(28, int(cfg.STTN_MAX_LOAD_NUM * 0.8))
            cfg.E2FGVI_MAX_LOAD_NUM = 60 if hw_profile == '8GB' else (100 if hw_profile == '12GB' else 120)
            cfg.E2FGVI_NEIGHBOR_LENGTH = 35 if hw_profile == '8GB' else (35 if hw_profile == '12GB' else 40)
            cfg.E2FGVI_REF_STRIDE = 2 if hw_profile == '8GB' else 4
            cfg.E2FGVI_MAX_CROP_SIDE = 1600 if hw_profile == '8GB' else 1920
            cfg.E2FGVI_USE_FP16 = True
            cfg.E2FGVI_USE_HQ = True
            cfg.OUTPUT_CRF = 14 if hw_profile == '8GB' else (16 if hw_profile != '16GB+' else 15)
            cfg.OUTPUT_PRESET = 'slow' if hw_profile != '16GB+' else 'slower'
            cfg.ENABLE_TEMPORAL_ANTIFLICKER = True
            cfg.ANTIFLICKER_ALPHA = 0.97
            cfg.MASK_FEATHER_RADIUS = 3
            cfg.SUBTITLE_AREA_DEVIATION_PIXEL = 20
            cfg.MASK_MORPH_CLOSE = False
            cfg.ENABLE_FAST_FINAL_CHECK = False
            cfg.ENABLE_OCR_REFINE = False
            print(f'[Preset] Best Quality applied (E2FGVI, auto profile: {hw_profile}, RAM: {ram_gb:.1f}GB)')

    @staticmethod
    def _apply_final_check_mode(mode_name):
        cfg = backend.main.config
        mode = str(mode_name or 'Off').strip().lower()
        cfg.ENABLE_OCR_REFINE = False
        if mode == 'off':
            cfg.ENABLE_FAST_FINAL_CHECK = False
            print('[FinalCheck] Off (E2FGVI/selected algo only)')
        elif mode == 'strict':
            cfg.ENABLE_FAST_FINAL_CHECK = True
            cfg.FINAL_CHECK_EDGE_DENSITY_THRESHOLD = 0.07
            cfg.FINAL_CHECK_TEMPORAL_DELTA_THRESHOLD = 13.0
            print('[FinalCheck] Strict (LaMa touch-up; may flicker on motion)')
        else:
            # Fast: optional light second pass if remnants remain
            cfg.ENABLE_FAST_FINAL_CHECK = True
            cfg.FINAL_CHECK_EDGE_DENSITY_THRESHOLD = 0.08
            cfg.FINAL_CHECK_TEMPORAL_DELTA_THRESHOLD = 14.0
            print('[FinalCheck] Fast (LaMa touch-up on flagged frames)')

    def _persist_gui_user_settings(self):
        cfg = backend.main.config
        data = {}
        for k in GUI_USER_SETTINGS_KEYS:
            if hasattr(cfg, k):
                data[k] = getattr(cfg, k)
        with open(GUI_USER_SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f'[GUI] Settings saved → {GUI_USER_SETTINGS_PATH}')

    def _show_advanced_settings_dialog(self):
        """Modal editor for values stored in gui_user_settings.json (overrides config.py on each Run)."""
        cfg = backend.main.config

        def _s(x):
            return '' if x is None else str(x)

        layout = [
            [sg.Text(
                'These values override backend/config.py for each processing run.',
                font=self.font,
            )],
            [sg.Text(
                f'Saved file: {GUI_USER_SETTINGS_PATH}',
                font=('Segoe UI', 9),
                text_color='#555555',
            )],
            [sg.HorizontalSeparator()],
            [sg.Text('Antiflicker alpha (0.5–0.99)', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'ANTIFLICKER_ALPHA', 0.97)), key='-ADV-AF-', size=(10, 1), font=self.font)],
            [sg.Text('Motion threshold (px)', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'ANTIFLICKER_MOTION_THRESHOLD', 8)), key='-ADV-MOT-', size=(10, 1), font=self.font)],
            [sg.Text('E2FGVI neighbor length', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'E2FGVI_NEIGHBOR_LENGTH', 35)), key='-ADV-NGH-', size=(10, 1), font=self.font)],
            [sg.Text('E2FGVI ref stride', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'E2FGVI_REF_STRIDE', 2)), key='-ADV-RST-', size=(10, 1), font=self.font)],
            [sg.Text('Mask feather radius', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'MASK_FEATHER_RADIUS', 3)), key='-ADV-FTH-', size=(10, 1), font=self.font)],
            [sg.Text('Subtitle pad (px)', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'SUBTITLE_AREA_DEVIATION_PIXEL', 20)), key='-ADV-PAD-', size=(10, 1), font=self.font)],
            [sg.Text('E2FGVI max batch (frames)', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'E2FGVI_MAX_LOAD_NUM', 60)), key='-ADV-ML-', size=(10, 1), font=self.font)],
            [sg.Text('E2FGVI stream batch', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'E2FGVI_STREAM_MAX_LOAD', 32)), key='-ADV-ST-', size=(10, 1), font=self.font)],
            [sg.Text('E2FGVI crop margin', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'E2FGVI_CROP_MARGIN', 128)), key='-ADV-CM-', size=(10, 1), font=self.font)],
            [sg.Text('E2FGVI max crop side', size=(26, 1), font=self.font),
             sg.Input(_s(getattr(cfg, 'E2FGVI_MAX_CROP_SIDE', 1600)), key='-ADV-MCS-', size=(10, 1), font=self.font)],
            [sg.Checkbox(
                'Manual boxes only (skip OCR)',
                default=bool(getattr(cfg, 'MANUAL_BOXES_ONLY', False)),
                key='-ADV-MANUAL-',
                font=self.font,
            )],
            [sg.Checkbox(
                'Force inpaint selected areas on every frame',
                default=bool(getattr(cfg, 'FORCE_INPAINT_SELECTED_AREAS', False)),
                key='-ADV-FORCE-',
                font=self.font,
            )],
            [sg.Checkbox(
                'Scene-cut scan (PySceneDetect; slow startup)',
                default=bool(getattr(cfg, 'E2FGVI_FORCE_SCENE_DETECT', False)),
                key='-ADV-SCENE-',
                font=self.font,
            )],
            [sg.Checkbox(
                'Post deflicker (if tool configured)',
                default=bool(getattr(cfg, 'ENABLE_POST_DEFLICKER', False)),
                key='-ADV-POST-',
                font=self.font,
            )],
            [sg.Button('Apply', key='-ADV-APPLY-', font=self.font),
             sg.Button('Reload from config.py', key='-ADV-RESET-', font=self.font),
             sg.Button('Cancel', key='-ADV-CANCEL-', font=self.font)],
        ]

        win = sg.Window(
            'Inpaint settings',
            layout,
            modal=True,
            finalize=True,
            font=self.font,
        )

        def _refresh_fields_from_cfg():
            c = backend.main.config
            win['-ADV-AF-'].update(_s(getattr(c, 'ANTIFLICKER_ALPHA', 0.97)))
            win['-ADV-MOT-'].update(_s(getattr(c, 'ANTIFLICKER_MOTION_THRESHOLD', 8)))
            win['-ADV-NGH-'].update(_s(getattr(c, 'E2FGVI_NEIGHBOR_LENGTH', 35)))
            win['-ADV-RST-'].update(_s(getattr(c, 'E2FGVI_REF_STRIDE', 2)))
            win['-ADV-FTH-'].update(_s(getattr(c, 'MASK_FEATHER_RADIUS', 3)))
            win['-ADV-PAD-'].update(_s(getattr(c, 'SUBTITLE_AREA_DEVIATION_PIXEL', 20)))
            win['-ADV-ML-'].update(_s(getattr(c, 'E2FGVI_MAX_LOAD_NUM', 60)))
            win['-ADV-ST-'].update(_s(getattr(c, 'E2FGVI_STREAM_MAX_LOAD', 32)))
            win['-ADV-CM-'].update(_s(getattr(c, 'E2FGVI_CROP_MARGIN', 128)))
            win['-ADV-MCS-'].update(_s(getattr(c, 'E2FGVI_MAX_CROP_SIDE', 1600)))
            win['-ADV-MANUAL-'].update(value=bool(getattr(c, 'MANUAL_BOXES_ONLY', False)))
            win['-ADV-FORCE-'].update(value=bool(getattr(c, 'FORCE_INPAINT_SELECTED_AREAS', False)))
            win['-ADV-SCENE-'].update(value=bool(getattr(c, 'E2FGVI_FORCE_SCENE_DETECT', False)))
            win['-ADV-POST-'].update(value=bool(getattr(c, 'ENABLE_POST_DEFLICKER', False)))

        while True:
            ev, vals = win.read()
            if ev in (sg.WIN_CLOSED, '-ADV-CANCEL-'):
                break
            if ev == '-ADV-RESET-':
                try:
                    if os.path.isfile(GUI_USER_SETTINGS_PATH):
                        os.remove(GUI_USER_SETTINGS_PATH)
                    import importlib
                    importlib.reload(backend.main.config)
                    _refresh_fields_from_cfg()
                    sg.popup_ok(
                        'Removed gui_user_settings.json and reloaded backend/config.py.\n'
                        'Click Apply to save these as your new GUI override.',
                        title='Reloaded from config.py',
                    )
                except Exception as e:
                    sg.popup_error(str(e), title='Reload failed')
                continue
            if ev != '-ADV-APPLY-':
                continue
            try:
                c = backend.main.config
                af = float(str(vals.get('-ADV-AF-', '0.97')).strip())
                mot = float(str(vals.get('-ADV-MOT-', '8')).strip())
                ngh = int(float(str(vals.get('-ADV-NGH-', '35')).strip()))
                rst = int(float(str(vals.get('-ADV-RST-', '2')).strip()))
                fth = int(float(str(vals.get('-ADV-FTH-', '3')).strip()))
                pad = int(float(str(vals.get('-ADV-PAD-', '20')).strip()))
                ml = int(float(str(vals.get('-ADV-ML-', '60')).strip()))
                st = int(float(str(vals.get('-ADV-ST-', '32')).strip()))
                cm = int(float(str(vals.get('-ADV-CM-', '128')).strip()))
                mcs = int(float(str(vals.get('-ADV-MCS-', '1600')).strip()))
                if not 0.5 <= af <= 0.999:
                    raise ValueError('Antiflicker alpha must be between 0.5 and 0.999')
                if not 0.0 <= mot <= 255.0:
                    raise ValueError('Motion threshold must be 0–255')
                if not 3 <= ngh <= 80:
                    raise ValueError('E2FGVI neighbor length must be 3–80')
                if not 1 <= rst <= 20:
                    raise ValueError('E2FGVI ref stride must be 1–20')
                if not 0 <= fth <= 30:
                    raise ValueError('Mask feather must be 0–30')
                if not 0 <= pad <= 120:
                    raise ValueError('Subtitle pad must be 0–120')
                if not 4 <= ml <= 200:
                    raise ValueError('E2FGVI max batch must be 4–200')
                if not 4 <= st <= 128:
                    raise ValueError('E2FGVI stream batch must be 4–128')
                if not 0 <= cm <= 400:
                    raise ValueError('Crop margin must be 0–400')
                if not 256 <= mcs <= 3840:
                    raise ValueError('Max crop side must be 256–3840')
                c.ANTIFLICKER_ALPHA = af
                c.ANTIFLICKER_MOTION_THRESHOLD = mot
                c.E2FGVI_NEIGHBOR_LENGTH = ngh
                c.E2FGVI_REF_STRIDE = rst
                c.MASK_FEATHER_RADIUS = fth
                c.SUBTITLE_AREA_DEVIATION_PIXEL = pad
                c.E2FGVI_MAX_LOAD_NUM = ml
                c.E2FGVI_STREAM_MAX_LOAD = st
                c.E2FGVI_CROP_MARGIN = cm
                c.E2FGVI_MAX_CROP_SIDE = mcs
                c.MANUAL_BOXES_ONLY = bool(vals.get('-ADV-MANUAL-', False))
                c.FORCE_INPAINT_SELECTED_AREAS = bool(vals.get('-ADV-FORCE-', False))
                c.E2FGVI_FORCE_SCENE_DETECT = bool(vals.get('-ADV-SCENE-', False))
                c.ENABLE_POST_DEFLICKER = bool(vals.get('-ADV-POST-', False))
                self._persist_gui_user_settings()
                sg.popup_ok('Saved. These values apply on the next Run.', title='Settings')
                break
            except Exception as e:
                sg.popup_error(f'Invalid value:\n\n{e}', title='Settings')
        try:
            win.close()
        except Exception:
            pass

    def _current_area_from_values(self, values):
        xmin = int(values['-X-SLIDER-'])
        xmax = int(values['-X-SLIDER-'] + values['-X-SLIDER-W-'])
        ymin = int(values['-Y-SLIDER-'])
        ymax = int(values['-Y-SLIDER-'] + values['-Y-SLIDER-H-'])
        if self.frame_height is not None and ymax > self.frame_height:
            ymax = int(self.frame_height)
        if self.frame_width is not None and xmax > self.frame_width:
            xmax = int(self.frame_width)
        return ymin, ymax, xmin, xmax

    def _refresh_box_count(self):
        if self.window is not None:
            self.window['-BOX-COUNT-'].update(f'Boxes: {len(self.subtitle_areas)}')

    def _push_box_undo(self):
        self.box_undo_stack.append((list(self.subtitle_areas), self.selected_box_index))
        if len(self.box_undo_stack) > 50:
            self.box_undo_stack.pop(0)

    def _undo_box_action(self):
        if not self.box_undo_stack:
            print('No box action to undo')
            return
        prev_areas, prev_selected = self.box_undo_stack.pop()
        self.subtitle_areas = list(prev_areas)
        self.selected_box_index = prev_selected
        self._refresh_box_count()
        self._refresh_box_list()
        print('Undo completed')

    def _refresh_box_list(self):
        if self.window is None:
            return
        values = []
        for i, (ymin, ymax, xmin, xmax) in enumerate(self.subtitle_areas, start=1):
            values.append(f'{i}. y:{ymin}-{ymax}, x:{xmin}-{xmax}')
        self.window['-BOX-LIST-'].update(values=values)
        if self.selected_box_index is not None and 0 <= self.selected_box_index < len(values):
            self.window['-BOX-LIST-'].set_value([values[self.selected_box_index]])
        else:
            self.window['-BOX-LIST-'].set_value([])

    def _add_current_box(self, values):
        if self.video_cap is None and not is_image_file(self.video_path):
            print('Please Open Video First')
            return
        area = self._current_area_from_values(values)
        if area not in self.subtitle_areas:
            self._push_box_undo()
            self.subtitle_areas.append(area)
            self.selected_box_index = len(self.subtitle_areas) - 1
            print(f'Added box: {area}')
        else:
            print(f'Box already exists: {area}')
            self.selected_box_index = self.subtitle_areas.index(area)
        self._refresh_box_count()
        self._refresh_box_list()

    def _delete_selected_box(self):
        if self.selected_box_index is None:
            print('Please select a box from the list first')
            return
        if 0 <= self.selected_box_index < len(self.subtitle_areas):
            self._push_box_undo()
            removed = self.subtitle_areas.pop(self.selected_box_index)
            print(f'Removed box: {removed}')
            if len(self.subtitle_areas) == 0:
                self.selected_box_index = None
            else:
                self.selected_box_index = min(self.selected_box_index, len(self.subtitle_areas) - 1)
        self._refresh_box_count()
        self._refresh_box_list()

    def _bind_mouse_events(self):
        try:
            widget = self.window['-DISPLAY-'].Widget
            widget.bind('<ButtonPress-1>', self._tk_mouse_down)
            widget.bind('<B1-Motion>', self._tk_mouse_move)
            widget.bind('<ButtonRelease-1>', self._tk_mouse_up)
        except Exception as e:
            print(f'Bind mouse failed: {e}')

    def _tk_mouse_down(self, event):
        self.window.write_event_value('-MOUSE-DOWN-', (event.x, event.y))

    def _tk_mouse_move(self, event):
        self.window.write_event_value('-MOUSE-MOVE-', (event.x, event.y))

    def _tk_mouse_up(self, event):
        self.window.write_event_value('-MOUSE-UP-', (event.x, event.y))

    def _preview_to_frame_point(self, x, y):
        if self.frame_width is None or self.frame_height is None:
            return None

        # Always query fresh widget geometry for accurate mapping
        try:
            widget = self.window['-DISPLAY-'].Widget
            widget.update_idletasks()
            widget_w = max(1, int(widget.winfo_width()))
            widget_h = max(1, int(widget.winfo_height()))
        except Exception:
            widget_w = int(self.video_preview_width)
            widget_h = int(self.video_preview_height)

        # Recompute content placement inside the widget using the same logic
        # as _img_resize, so coordinates are always consistent even after
        # resize/maximize where cached values may be stale.
        fw = int(self.frame_width)
        fh = int(self.frame_height)
        scale = min(widget_w / max(1, fw), widget_h / max(1, fh))
        content_w = max(1, int(fw * scale))
        content_h = max(1, int(fh * scale))
        content_off_x = (widget_w - content_w) // 2
        content_off_y = (widget_h - content_h) // 2

        xc = int(x) - content_off_x
        yc = int(y) - content_off_y
        if xc < 0 or yc < 0 or xc >= content_w or yc >= content_h:
            return None

        fx = int(xc * fw / max(1, content_w))
        fy = int(yc * fh / max(1, content_h))
        fx = max(0, min(fx, fw - 1))
        fy = max(0, min(fy, fh - 1))
        return fx, fy

    def _mouse_event_handler(self, event, values):
        if event == '-RUN-FAILED-':
            self.sr = None
            self.__enable_button()
            msg = values.get('-RUN-FAILED-', '')
            self._set_status('ERROR', 'Processing failed')
            if 'out of memory' in str(msg).lower():
                print('[TIP] GPU out of memory. Use Balanced/Fast, smaller boxes, or shorter clips.')
            return
        if event == '-PIPELINE-INFO-UPDATE-':
            info = values.get('-PIPELINE-INFO-UPDATE-', '')
            try:
                self.window['-PIPELINE-INFO-'].update(info)
            except Exception:
                pass
            return
        if self.current_preview_frame is None:
            return
        if event == '-MOUSE-DOWN-':
            self.dragging = True
            self.drag_start = values['-MOUSE-DOWN-']
            self.drag_end = values['-MOUSE-DOWN-']
        elif event == '-MOUSE-MOVE-' and self.dragging:
            self.drag_end = values['-MOUSE-MOVE-']
            y = int(values['-Y-SLIDER-'])
            h = int(values['-Y-SLIDER-H-'])
            x = int(values['-X-SLIDER-'])
            w = int(values['-X-SLIDER-W-'])
            self._update_preview(self.current_preview_frame.copy(), (y, h, x, w))
        elif event == '-MOUSE-UP-' and self.dragging:
            self.dragging = False
            self.drag_end = values['-MOUSE-UP-']
            p1 = self._preview_to_frame_point(*self.drag_start)
            p2 = self._preview_to_frame_point(*self.drag_end)
            if p1 is None or p2 is None:
                return
            xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
            ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
            if (xmax - xmin) < 3 or (ymax - ymin) < 3:
                return
            self.window['-X-SLIDER-'].update(xmin)
            self.window['-Y-SLIDER-'].update(ymin)
            self.window['-X-SLIDER-W-'].update(max(1, xmax - xmin))
            self.window['-Y-SLIDER-H-'].update(max(1, ymax - ymin))
            area = (ymin, ymax, xmin, xmax)
            if area not in self.subtitle_areas:
                self._push_box_undo()
                self.subtitle_areas.append(area)
                self.selected_box_index = len(self.subtitle_areas) - 1
                print(f'Added box by mouse: {area}')
                self._refresh_box_count()
                self._refresh_box_list()

    def _run_event_handler(self, event, values):
        """
        当点击运行按钮时：
        1) 禁止修改字幕滑块区域
        2) 禁止再次点击【运行】和【打开】按钮
        3) 设定字幕区域位置
        """
        if event == '-RESUME-LAST-':
            self._resume_last_job(values)
            return
        if event == '-ADD-BOX-':
            self._add_current_box(values)
            return
        if event == '-PLAY-' or event == '-SPACE-':
            if self.video_cap is None:
                print('Please Open Video First')
            else:
                self._set_play_state(not self.is_playing)
            return
        if event == '-TOGGLE-CONSOLE-':
            self.console_visible = not self.console_visible
            self.window['-CONSOLE-FRAME-'].update(visible=self.console_visible)
            self.window['-TOGGLE-CONSOLE-'].update(text='Hide Console' if self.console_visible else 'Show Console')
            return
        if event == '-RESET-LAYOUT-':
            self._reset_layout()
            return
        if event == '-ADV-SETTINGS-':
            self._show_advanced_settings_dialog()
            return
        if event == '-RECOMMENDED-':
            # One-click best speed+quality (stable on moving clips)
            try:
                self.window['-PASS-MODE-'].update(value='Two-Pass Best')
                self.window['-QUALITY-'].update(value='Best Quality')
                self.window['-ANTIFLICKER-'].update(value=True)
                self.window['-FINAL-CHECK-'].update(value='Off')
            except Exception:
                pass
            self.pass_mode = 'Two-Pass Best'
            self._two_pass_mode = True
            self._max_quality_mode = False
            self.quality_preset = 'Best Quality'
            self.final_check_mode = 'Off'
            self._apply_quality_preset(self.quality_preset)
            self._apply_final_check_mode(self.final_check_mode)
            backend.main.config.ENABLE_TEMPORAL_ANTIFLICKER = True
            print('[Recommended] Two-Pass Best + Best Quality + Anti-Flicker ON + Final Check Off')
            return
        if event == '-BQ-OPT-':
            # Same as Recommended; optional slight temporal smoothing if you still see shimmer
            try:
                self.window['-PASS-MODE-'].update(value='Two-Pass Best')
                self.window['-QUALITY-'].update(value='Best Quality')
                self.window['-ANTIFLICKER-'].update(value=True)
                self.window['-FINAL-CHECK-'].update(value='Off')
            except Exception:
                pass
            self.pass_mode = 'Two-Pass Best'
            self._two_pass_mode = True
            self._max_quality_mode = False
            self.quality_preset = 'Best Quality'
            self.final_check_mode = 'Off'
            self._apply_quality_preset(self.quality_preset)
            self._apply_final_check_mode(self.final_check_mode)
            try:
                cfg = backend.main.config
                cfg.MASK_FEATHER_RADIUS = 3
                cfg.ANTIFLICKER_ALPHA = 0.97
                cfg.ENABLE_OCR_REFINE = False
            except Exception:
                pass
            backend.main.config.ENABLE_TEMPORAL_ANTIFLICKER = True
            print('[BQ Optimized] Two-Pass Best + Best Quality + feather=3 + Final Check Off')
            return
        if event == '-MAX-QUALITY-':
            # One-click max quality (slow) — keeps Strict enabled
            try:
                self.window['-PASS-MODE-'].update(value='Two-Pass Best')
                self.window['-QUALITY-'].update(value='Best Quality')
                self.window['-ANTIFLICKER-'].update(value=True)
                self.window['-FINAL-CHECK-'].update(value='Strict')
            except Exception:
                pass
            self.pass_mode = 'Two-Pass Best'
            self._two_pass_mode = True
            self._max_quality_mode = True
            self.quality_preset = 'Best Quality'
            self.final_check_mode = 'Strict'
            self._apply_quality_preset(self.quality_preset)
            self._apply_final_check_mode(self.final_check_mode)
            try:
                cfg = backend.main.config
                cfg.MASK_FEATHER_RADIUS = 3
                cfg.ANTIFLICKER_ALPHA = 0.97
                cfg.ENABLE_OCR_REFINE = False
            except Exception:
                pass
            backend.main.config.ENABLE_TEMPORAL_ANTIFLICKER = True
            print('[MaxQuality] Two-Pass Best + Best Quality + Strict final check (LaMa only if needed)')
            return
        if event in ('-DEL-BOX-', '-DEL-KEY-'):
            self._delete_selected_box()
            if self.current_preview_frame is not None:
                y = int(values.get('-Y-SLIDER-', 0))
                h = int(values.get('-Y-SLIDER-H-', 0))
                x = int(values.get('-X-SLIDER-', 0))
                w = int(values.get('-X-SLIDER-W-', 0))
                self._update_preview(self.current_preview_frame.copy(), (y, h, x, w))
            return
        if event == '-UNDO-BOX-':
            self._undo_box_action()
            return
        if event == '-CLEAR-BOXES-':
            self._push_box_undo()
            self.subtitle_areas = []
            self.selected_box_index = None
            self._refresh_box_count()
            self._refresh_box_list()
            print('Cleared all boxes')
            return
        if event == '-BOX-LIST-':
            selected = values.get('-BOX-LIST-', [])
            if selected:
                try:
                    index_str = str(selected[0]).split('.', 1)[0].strip()
                    self.selected_box_index = int(index_str) - 1
                except Exception:
                    self.selected_box_index = None
                if self.selected_box_index is not None and 0 <= self.selected_box_index < len(self.subtitle_areas):
                    area = self.subtitle_areas[self.selected_box_index]
                    ymin, ymax, xmin, xmax = area
                    self.window['-X-SLIDER-'].update(xmin)
                    self.window['-Y-SLIDER-'].update(ymin)
                    self.window['-X-SLIDER-W-'].update(max(1, xmax - xmin))
                    self.window['-Y-SLIDER-H-'].update(max(1, ymax - ymin))
            return
        if event == '-RUN-':
            if self.video_cap is None:
                print('Please Open Video First')
            else:
                self.pass_mode = values.get('-PASS-MODE-', 'Final Quality')
                self._two_pass_mode = False
                if self.pass_mode == 'Preview Fast':
                    self.quality_preset = 'Fast'
                    self.final_check_mode = 'Off'
                    anti_flicker = False
                    print('[PassMode] Preview Fast: forcing Fast + FinalCheck Off')
                elif self.pass_mode == 'Two-Pass Best':
                    self._two_pass_mode = True
                    self.quality_preset = values.get('-QUALITY-', 'Best Quality')
                    self.final_check_mode = values.get('-FINAL-CHECK-', 'Off')
                    anti_flicker = bool(values.get('-ANTIFLICKER-', True))
                    print('[PassMode] Two-Pass Best: Pass 1 = STTN fast detect, Pass 2 = quality inpaint with cached detection')
                else:
                    self.quality_preset = values.get('-QUALITY-', 'Best Quality')
                    self.final_check_mode = values.get('-FINAL-CHECK-', 'Off')
                    anti_flicker = bool(values.get('-ANTIFLICKER-', True))
                self._apply_quality_preset(self.quality_preset)
                backend.main.config.ENABLE_TEMPORAL_ANTIFLICKER = anti_flicker
                print(f"[Option] Anti-Flicker: {backend.main.config.ENABLE_TEMPORAL_ANTIFLICKER}")
                self._apply_final_check_mode(self.final_check_mode)
                self._set_play_state(False)
                self._set_status('PROCESSING')
                self.window['-PIPELINE-INFO-'].update(
                    f'Pass: {self.pass_mode} | Resume: frame 0 | Merged parts: 0'
                )
                self.window['-TIME-INFO-'].update('Elapsed: 00:00 | ETA: --:-- | Stage: starting')
                # 禁用按钮
                self.__disable_button()
                # 3) 设定字幕区域位置
                self.ymin, self.ymax, self.xmin, self.xmax = self._current_area_from_values(values)
                selected_areas = list(self.subtitle_areas)
                current_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                if current_area not in selected_areas:
                    selected_areas.append(current_area)
                if len(self.video_paths) <= 1:
                    subtitle_area = selected_areas if len(selected_areas) > 1 else selected_areas[0]
                else:
                    print(f"{'Processing multiple videos or images'}")
                    # 先判断每个视频的分辨率是否一致，一致的话设置相同的字幕区域，否则设置为None
                    global_size = None
                    all_same_size = True
                    for temp_video_path in self.video_paths:
                        temp_cap = cv2.VideoCapture(temp_video_path)
                        temp_w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        temp_h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        temp_cap.release()
                        if global_size is None:
                            global_size = (temp_w, temp_h)
                        elif (temp_w, temp_h) != global_size:
                            print('not all video/images in same size, processing in full screen')
                            subtitle_area = None
                            all_same_size = False
                            break
                    if all_same_size:
                        subtitle_area = selected_areas if len(selected_areas) > 1 else selected_areas[0]
                y_p = self.ymin / self.frame_height
                h_p = (self.ymax - self.ymin) / self.frame_height
                x_p = self.xmin / self.frame_width
                w_p = (self.xmax - self.xmin) / self.frame_width
                self.set_subtitle_config(y_p, h_p, x_p, w_p)

                two_pass = self._two_pass_mode
                saved_quality_preset = self.quality_preset
                saved_final_check = self.final_check_mode

                def task():
                    while self.video_paths:
                        video_path = self.video_paths.pop()
                        if subtitle_area is not None:
                            print(f"{'SubtitleArea'}: {subtitle_area}")

                        if two_pass:
                            # --- Pass 1: fast STTN detection to build cache ---
                            print('=' * 60)
                            print('[Two-Pass] === PASS 1/2: Fast detection (STTN) ===')
                            print('=' * 60)
                            try:
                                self.window.write_event_value('-PIPELINE-INFO-UPDATE-',
                                    'Two-Pass | Pass 1/2: detecting subtitles...')
                            except Exception:
                                pass
                            cfg = backend.main.config
                            cfg.MODE = cfg.InpaintMode.STTN
                            cfg.STTN_SKIP_DETECTION = True
                            cfg.ENABLE_DETECTION_CACHE = True
                            cfg.ENABLE_TEMPORAL_ANTIFLICKER = False
                            cfg.ENABLE_FAST_FINAL_CHECK = False
                            cfg.MASK_FEATHER_RADIUS = 0
                            cfg.OUTPUT_CRF = 30
                            cfg.OUTPUT_PRESET = 'ultrafast'

                            self.sr = backend.main.SubtitleRemover(video_path, subtitle_area, True)
                            try:
                                self.sr.run()
                            except Exception as e:
                                print(f'[ERROR] Pass 1 failed: {e}')
                                try:
                                    self.window.write_event_value('-RUN-FAILED-', f'Pass 1: {e}')
                                except Exception:
                                    pass
                                continue

                            pass1_output = self.sr.video_out_name
                            print('[Two-Pass] Pass 1 complete – detection cache saved')

                            # --- Pass 2: quality inpainting using cached detections ---
                            print('=' * 60)
                            print('[Two-Pass] === PASS 2/2: Quality inpainting ===')
                            print('=' * 60)
                            try:
                                self.window.write_event_value('-PIPELINE-INFO-UPDATE-',
                                    'Two-Pass | Pass 2/2: quality inpainting...')
                            except Exception:
                                pass
                            self._apply_quality_preset(saved_quality_preset)
                            self._apply_final_check_mode(saved_final_check)
                            cfg.ENABLE_DETECTION_CACHE = True
                            cfg.ENABLE_TEMPORAL_ANTIFLICKER = True

                            self.sr = backend.main.SubtitleRemover(video_path, subtitle_area, True)
                            try:
                                self.sr.run()
                            except Exception as e:
                                print(f'[ERROR] Pass 2 failed: {e}')
                                try:
                                    self.window.write_event_value('-RUN-FAILED-', f'Pass 2: {e}')
                                except Exception:
                                    pass
                                continue

                            # Remove pass-1 throwaway if pass-2 succeeded and overwrote it
                            if os.path.exists(pass1_output) and pass1_output == self.sr.video_out_name:
                                print('[Two-Pass] Output overwritten by pass 2 – done')
                            print('[Two-Pass] === Both passes complete ===')
                        else:
                            self.sr = backend.main.SubtitleRemover(video_path, subtitle_area, True)
                            try:
                                self.sr.run()
                            except Exception as e:
                                print(f'[ERROR] Processing failed: {e}')
                                try:
                                    self.window.write_event_value('-RUN-FAILED-', str(e))
                                except Exception:
                                    pass
                Thread(target=task, daemon=True).start()
                self.video_cap.release()
                self.video_cap = None

    def _resume_last_job(self, values):
        """
        Resume helper:
        - reads latest checkpoint json
        - auto-opens source video
        - jumps preview to last processed frame
        - auto-starts processing
        """
        candidates = []
        try:
            if self.video_path:
                candidates.extend(glob.glob(os.path.join(os.path.dirname(self.video_path), '*_no_sub.progress.json')))
            candidates.extend(glob.glob(os.path.join(os.getcwd(), '**', '*_no_sub.progress.json'), recursive=True))
        except Exception:
            candidates = []
        candidates = [p for p in candidates if os.path.isfile(p)]
        if not candidates:
            print('No checkpoint file found.')
            return
        ckpt_path = max(candidates, key=lambda p: os.path.getmtime(p))
        try:
            with open(ckpt_path, 'r', encoding='utf-8') as f:
                ckpt = json.load(f)
        except Exception as e:
            print(f'Failed to read checkpoint: {e}')
            return
        source_video = ckpt.get('source_video', '')
        processed_frames = int(ckpt.get('processed_frames', 1) or 1)
        if not source_video or not os.path.exists(source_video):
            print('Checkpoint source video not found.')
            return

        # Load file into GUI just like file browse
        self._file_event_handler('-FILE-', {'-FILE-': source_video})
        if self.video_cap is None or not self.video_cap.isOpened():
            print('Failed to open source video from checkpoint.')
            return

        frame_no = max(1, min(processed_frames, int(self.frame_count or processed_frames)))
        self.window['-SLIDER-'].update(frame_no)
        self._slide_event_handler('-SLIDER-', {
            '-SLIDER-': frame_no,
            '-Y-SLIDER-': values.get('-Y-SLIDER-', 0),
            '-Y-SLIDER-H-': values.get('-Y-SLIDER-H-', 0),
            '-X-SLIDER-': values.get('-X-SLIDER-', 0),
            '-X-SLIDER-W-': values.get('-X-SLIDER-W-', 0)
        })
        print(f'[Resume] Loaded checkpoint: {os.path.basename(ckpt_path)} (frame {frame_no})')
        print('[Resume] Auto-starting processing...')

        # Reuse current UI settings and trigger run
        run_values = {
            '-QUALITY-': self.window['-QUALITY-'].get(),
            '-ANTIFLICKER-': self.window['-ANTIFLICKER-'].get(),
            '-FINAL-CHECK-': self.window['-FINAL-CHECK-'].get(),
            '-X-SLIDER-': self.window['-X-SLIDER-'].get(),
            '-X-SLIDER-W-': self.window['-X-SLIDER-W-'].get(),
            '-Y-SLIDER-': self.window['-Y-SLIDER-'].get(),
            '-Y-SLIDER-H-': self.window['-Y-SLIDER-H-'].get(),
        }
        self._run_event_handler('-RUN-', run_values)

    def _slide_event_handler(self, event, values):
        """
        当滑动视频进度条/滑动字幕选择区域滑块时：
        1) 判断视频是否存在，如果存在则显示对应的视频帧
        2) 绘制rectangle
        """
        if event == '-SLIDER-' or event == '-Y-SLIDER-' or event == '-Y-SLIDER-H-' or event == '-X-SLIDER-' or event \
                == '-X-SLIDER-W-':
            if event == '-SLIDER-':
                self._set_play_state(False)
            # 判断是否时单张图片
            if is_image_file(self.video_path):
                img = cv2.imread(self.video_path)
                yy = int(values['-Y-SLIDER-'])
                xx = int(values['-X-SLIDER-'])
                self.window['-Y-SLIDER-H-'].update(range=(0, max(1, int(self.frame_height) - yy)))
                self.window['-X-SLIDER-W-'].update(range=(0, max(1, int(self.frame_width) - xx)))
                # 画字幕框
                y = int(values['-Y-SLIDER-'])
                h = int(values['-Y-SLIDER-H-'])
                x = int(values['-X-SLIDER-'])
                w = int(values['-X-SLIDER-W-'])
                self._update_preview(img, (y, h, x, w))
            elif self.video_cap is not None and self.video_cap.isOpened():
                frame_no = int(values['-SLIDER-'])
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = self.video_cap.read()
                if ret:
                    self._update_frame_info(frame_no)
                    yy = int(values['-Y-SLIDER-'])
                    xx = int(values['-X-SLIDER-'])
                    self.window['-Y-SLIDER-H-'].update(range=(0, max(1, int(self.frame_height) - yy)))
                    self.window['-X-SLIDER-W-'].update(range=(0, max(1, int(self.frame_width) - xx)))
                    # 画字幕框
                    y = int(values['-Y-SLIDER-'])
                    h = int(values['-Y-SLIDER-H-'])
                    x = int(values['-X-SLIDER-'])
                    w = int(values['-X-SLIDER-W-'])
                    self._update_preview(frame, (y, h, x, w))

    def _update_preview(self, frame, y_h_x_w):
        frame = self._frame_to_bgr(frame)
        if frame is None:
            return
        self.current_preview_frame = frame.copy()
        y, h, x, w = y_h_x_w
        draw = frame.copy()
        # 先画已添加区域（蓝色）
        for i, area in enumerate(self.subtitle_areas):
            ymin, ymax, xmin, xmax = area
            color = (0, 0, 255) if i == self.selected_box_index else (255, 0, 0)
            draw = cv2.rectangle(img=draw, pt1=(int(xmin), int(ymin)), pt2=(int(xmax), int(ymax)),
                                 color=color, thickness=2)
            draw = cv2.putText(draw, str(i + 1), (int(xmin), max(15, int(ymin) - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        # 再画当前编辑区域（绿色）
        draw = cv2.rectangle(img=draw, pt1=(int(x), int(y)), pt2=(int(x) + int(w), int(y) + int(h)),
                             color=(0, 255, 0), thickness=3)
        if self.dragging and self.drag_start is not None and self.drag_end is not None:
            fx1_fy1 = self._preview_to_frame_point(*self.drag_start)
            fx2_fy2 = self._preview_to_frame_point(*self.drag_end)
            if fx1_fy1 is not None and fx2_fy2 is not None:
                fx1, fy1 = fx1_fy1
                fx2, fy2 = fx2_fy2
                draw = cv2.rectangle(img=draw, pt1=(min(fx1, fx2), min(fy1, fy2)),
                                     pt2=(max(fx1, fx2), max(fy1, fy2)),
                                     color=(0, 255, 255), thickness=2)
        # 调整视频帧大小，使播放器能够显示
        resized_frame = self._img_resize(draw)
        self._safe_display_update(resized_frame)

    def _img_resize(self, image):
        image = self._frame_to_bgr(image)
        if image is None:
            return np.zeros((max(1, int(self.video_preview_height)), max(1, int(self.video_preview_width)), 3), dtype=np.uint8)
        frame_h, frame_w = image.shape[:2]

        # Use the actual widget dimensions so the rendered canvas always
        # matches the display, even after resize/maximize.
        try:
            widget = self.window['-DISPLAY-'].Widget
            widget.update_idletasks()
            canvas_w = max(1, int(widget.winfo_width()))
            canvas_h = max(1, int(widget.winfo_height()))
            if canvas_w < 10 or canvas_h < 10:
                raise ValueError('widget too small')
        except Exception:
            canvas_w = int(max(1, self.video_preview_width))
            canvas_h = int(max(1, self.video_preview_height))

        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        off_x = (canvas_w - new_w) // 2
        off_y = (canvas_h - new_h) // 2
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

        return canvas

    def set_subtitle_config(self, y, h, x, w):
        # 写入配置文件
        with open(self.subtitle_config_file, mode='w', encoding='utf-8') as f:
            f.write('[AREA]\n')
            f.write(f'Y = {y}\n')
            f.write(f'H = {h}\n')
            f.write(f'X = {x}\n')
            f.write(f'W = {w}\n')

    def parse_subtitle_config(self):
        y_p, h_p, x_p, w_p = .78, .21, .05, .9
        # 如果配置文件不存在，则写入配置文件
        if not os.path.exists(self.subtitle_config_file):
            self.set_subtitle_config(y_p, h_p, x_p, w_p)
            return y_p, h_p, x_p, w_p
        else:
            try:
                config = configparser.ConfigParser()
                config.read(self.subtitle_config_file, encoding='utf-8')
                conf_y_p, conf_h_p, conf_x_p, conf_w_p = float(config['AREA']['Y']), float(config['AREA']['H']), float(config['AREA']['X']), float(config['AREA']['W'])
                return conf_y_p, conf_h_p, conf_x_p, conf_w_p
            except Exception:
                self.set_subtitle_config(y_p, h_p, x_p, w_p)
                return y_p, h_p, x_p, w_p


if __name__ == '__main__':
    try:
        # DPI awareness is applied at module import (before PySimpleGUI / Tk).
        # Filter noisy but harmless Windows message from dependencies.
        sys.stdout = _FilteredStream(sys.stdout)
        sys.stderr = _FilteredStream(sys.stderr)
        multiprocessing.set_start_method("spawn")
        # 运行图形化界面
        subtitleRemoverGUI = SubtitleRemoverGUI()
        subtitleRemoverGUI.run()
    except Exception as e:
        print(f'[{type(e)}] {e}')
        import traceback
        traceback.print_exc()
        msg = traceback.format_exc()
        err_log_path = os.path.join(os.path.expanduser('~'), 'VSR-Error-Message.log')
        with open(err_log_path, 'w', encoding='utf-8') as f:
            f.writelines(msg)
        import platform
        if platform.system() == 'Windows':
            os.system('pause')
        else:
            input()
