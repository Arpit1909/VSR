import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import onnxruntime as ort

# 项目版本号
VERSION = "1.1.12"
# ×××××××××××××××××××× [不要改] start ××××××××××××××××××××
logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印
try:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    USE_DML = True
except:
    USE_DML = False
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

# 查看该路径下是否有模型完整文件，没有的话合并小文件生成完整文件
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

# 指定ffmpeg可执行程序路径
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
    FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
    FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)
else:
    # Use Homebrew ffmpeg on Mac — bundled binary is Intel-only, won't run on Apple Silicon
    import shutil as _shutil
    _brew_ffmpeg = _shutil.which('ffmpeg') or '/opt/homebrew/bin/ffmpeg'
    FFMPEG_PATH = _brew_ffmpeg

if sys_str == "Windows" and 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# 将ffmpeg添加可执行权限
os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 是否使用ONNX(DirectML/AMD/Intel)
ONNX_PROVIDERS = []
available_providers = ort.get_available_providers()
for provider in available_providers:
    if provider in [
        "CPUExecutionProvider"
    ]:
        continue
    if provider not in [
        "DmlExecutionProvider",         # DirectML，适用于 Windows GPU
        "ROCMExecutionProvider",        # AMD ROCm
        "MIGraphXExecutionProvider",    # AMD MIGraphX
        "VitisAIExecutionProvider",     # AMD VitisAI，适用于 RyzenAI & Windows, 实测和DirectML性能似乎差不多
        "OpenVINOExecutionProvider",    # Intel GPU
        # CoreML excluded: PaddleOCR detection model has ops CoreML doesn't implement
        # "CoreMLExecutionProvider",
        "MetalExecutionProvider",       # Apple macOS
        "CUDAExecutionProvider",        # Nvidia GPU
    ]:
        continue
    ONNX_PROVIDERS.append(provider)
# ×××××××××××××××××××× [不要改] end ××××××××××××××××××××


@unique
class InpaintMode(Enum):
    """
    图像重绘算法枚举
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'
    E2FGVI = 'e2fgvi'


# ×××××××××××××××××××× [可以改] start ××××××××××××××××××××
# 是否使用h264编码，如果需要安卓手机分享生成的视频，请打开该选项
USE_H264 = True
# 输出编码质量（仅USE_H264=True时生效）
# CRF 越小质量越高、体积越大；18 通常接近视觉无损，16 更高质量
OUTPUT_CRF = 14
# 编码预设：越慢质量/压缩效率通常越好
OUTPUT_PRESET = "slow"
# 抗闪烁后处理（仅对掩码区域进行时序平滑）
ENABLE_TEMPORAL_ANTIFLICKER = True
# 越高越清晰（少与上一帧混合）；越低越稳。
ANTIFLICKER_ALPHA = 0.97
# Motion gating (pixel intensity delta) for temporal antiflicker.
# Only low-motion pixels inside mask are blended; moving pixels keep detail.
ANTIFLICKER_MOTION_THRESHOLD = 8.0
# 0 = 硬边合成，最清晰；>0 会略糊边（终检 LaMa 已默认关闭）
MASK_FEATHER_RADIUS = 3
# LaMa 终检叠在 E2FGVI 上易造成闪烁/发糊 — 默认关闭；仅 GUI 选 Strict 时开启
ENABLE_FAST_FINAL_CHECK = False
ENABLE_OCR_REFINE = False
OCR_REFINE_PAD = 2
# 可疑判定阈值（越严格越容易触发二次修复）
FINAL_CHECK_EDGE_DENSITY_THRESHOLD = 0.08
FINAL_CHECK_TEMPORAL_DELTA_THRESHOLD = 14.0
FINAL_CHECK_MIN_MASK_PIXELS = 100
# 终检安全边界：对检测到字幕帧前后额外复检，降低漏检风险
FINAL_CHECK_FRAME_MARGIN = 2
# 分片与断点续跑（视频）
ENABLE_CHUNK_CHECKPOINT = True
CHUNK_SECONDS = 5
# 检测缓存：复用字幕检测结果，加速重复处理同一视频
ENABLE_DETECTION_CACHE = True
# Drawn boxes merged into every frame when True (dense map; slower). Editable in GUI Advanced.
FORCE_INPAINT_SELECTED_AREAS = True
# Skip OCR; only inpaint inside drawn boxes (GUI Advanced). Requires FORCE-style dense map if True.
MANUAL_BOXES_ONLY = True
# Post-process deflicker (run after inpainting, before audio merge).
# Methods:
#   - blind-video-deflicker / film / external: run POST_DEFLICKER_EXTERNAL_CMD if provided
#   - ffmpeg-deflicker: use ffmpeg built-in deflicker filter
# If method is blind-video-deflicker/film and external command is missing,
# deflicker step is skipped (no hidden ffmpeg fallback blur).
ENABLE_POST_DEFLICKER = True
POST_DEFLICKER_METHOD = "blind-video-deflicker"
# Template command for external deflicker tools (must output to {output}).
# Example (blind-video-deflicker):
#   python "D:/tools/blind-video-deflicker/inference.py" --input {input} --output {output}
# Example (FILM wrapper script):
#   python "D:/tools/film/run_film_deflicker.py" --in {input} --out {output}
POST_DEFLICKER_EXTERNAL_CMD = ""
# FFmpeg deflicker strength (default 4). 2-6 is typical.
POST_DEFLICKER_FFMPEG_STRENGTH = 4
# Smart algorithm auto-selection: measure background motion before processing and
# automatically pick LaMa / E2FGVI / ProPainter for best quality per clip.
# True = 按运动切换 LaMa/E2FGVI/ProPainter（ProPainter 很慢，易误以为卡死）
AUTO_SELECT_ALGORITHM = False

# ×××××××××× 通用设置 start ××××××××××
"""
MODE可选算法类型
- InpaintMode.STTN 算法：对于真人视频效果较好，速度快，可以跳过字幕检测
- InpaintMode.LAMA 算法：对于动画类视频效果好，速度一般，不可以跳过字幕检测
- InpaintMode.PROPAINTER 算法： 需要消耗大量显存，速度较慢，对运动非常剧烈的视频效果较好
"""
# 【设置inpaint算法】
# E2FGVI  = E2FGVI-HQ only in this app (model/e2fgvi_hq.py + E2FGVI-HQ-CVPR22.pth)
# PROPAINTER = good quality but needs large batch (PROPAINTER_MAX_LOAD_NUM≥40) to avoid flicker
# STTN    = fast, decent quality for talking-head videos
# LAMA    = frame-by-frame, fast, best for static backgrounds
MODE = InpaintMode.LAMA
# 【设置像素点偏差】
# 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 6
# Mask padding around detected text boxes. Larger = fewer text remnants but more blur.
# 掩码外扩像素。较高可减少残留，较低可减少过度修复。
SUBTITLE_AREA_DEVIATION_PIXEL = 20
# Optional morphology closing for subtitle masks. Wide closing can over-expand masks
# and create blur; keep disabled by default for sharpest output.
MASK_MORPH_CLOSE = False
MASK_MORPH_KERNEL_W = 32
MASK_MORPH_KERNEL_H = 3
# 同于判断两个文本框是否为同一行字幕，高度差距指定像素点以内认为是同一行
THRESHOLD_HEIGHT_DIFFERENCE = 20
# 用于判断两个字幕文本的矩形框是否相似，如果X轴和Y轴偏差都在指定阈值内，则认为时同一个文本框
PIXEL_TOLERANCE_Y = 20  # 允许检测框纵向偏差的像素点数
PIXEL_TOLERANCE_X = 20  # 允许检测框横向偏差的像素点数
# ×××××××××× 通用设置 end ××××××××××

# ×××××××××× InpaintMode.STTN算法设置 start ××××××××××
# 以下参数仅适用STTN算法时，才生效
"""
1. STTN_SKIP_DETECTION
含义：是否使用跳过检测
效果：设置为True跳过字幕检测，会省去很大时间，但是可能误伤无字幕的视频帧或者会导致去除的字幕漏了

2. STTN_NEIGHBOR_STRIDE
含义：相邻帧数步长, 如果需要为第50帧填充缺失的区域，STTN_NEIGHBOR_STRIDE=5，那么算法会使用第45帧、第40帧等作为参照。
效果：用于控制参考帧选择的密度，较大的步长意味着使用更少、更分散的参考帧，较小的步长意味着使用更多、更集中的参考帧。

3. STTN_REFERENCE_LENGTH
含义：参数帧数量，STTN算法会查看每个待修复帧的前后若干帧来获得用于修复的上下文信息
效果：调大会增加显存占用，处理效果变好，但是处理速度变慢

4. STTN_MAX_LOAD_NUM
含义：STTN算法每次最多加载的视频帧数量
效果：设置越大速度越慢，但效果越好
注意：要保证STTN_MAX_LOAD_NUM大于STTN_NEIGHBOR_STRIDE和STTN_REFERENCE_LENGTH
"""
STTN_SKIP_DETECTION = True
# 参考帧步长
STTN_NEIGHBOR_STRIDE = 5
# 参考帧长度（数量）
STTN_REFERENCE_LENGTH = 10
# 设置STTN算法最大同时处理的帧数量
STTN_MAX_LOAD_NUM = 50
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ×××××××××× InpaintMode.STTN算法设置 end ××××××××××

# ×××××××××× InpaintMode.PROPAINTER算法设置 start ××××××××××
# 【根据自己的GPU显存大小设置】最大同时处理的图片数量，设置越大处理效果越好，但是要求显存越高
# 1280x720p视频设置80需要25G显存，设置50需要19G显存
# 720x480p视频设置80需要8G显存，设置50需要7G显存
# 8GB VRAM guide:
#   720p  → MAX_LOAD_NUM=40, NEIGHBOR_LENGTH=20  safe
#   1080p → MAX_LOAD_NUM=20, NEIGHBOR_LENGTH=10  safe
if sys_str == "Darwin":  # Mac (MPS / Apple Silicon)
    PROPAINTER_MAX_LOAD_NUM = 25
    PROPAINTER_NEIGHBOR_LENGTH = 12
    PROPAINTER_REF_STRIDE = 10
    PROPAINTER_RAFT_ITER = 20
else:  # Windows / Linux (CUDA)
    PROPAINTER_MAX_LOAD_NUM = 40
    PROPAINTER_NEIGHBOR_LENGTH = 20
    PROPAINTER_REF_STRIDE = 10
    PROPAINTER_RAFT_ITER = 20
PROPAINTER_MASK_DILATION = 4
# ×××××××××× InpaintMode.PROPAINTER算法设置 end ××××××××××

# ×××××××××× InpaintMode.E2FGVI算法设置 start ××××××××××
"""
E2FGVI-HQ (End-to-End Flow-Guided Video Inpainting, CVPR 2022)

Setup (one-time):
  1. git clone https://github.com/MCG-NKU/E2FGVI backend/inpaint/e2fgvi_repo
  2. Download weights → backend/models/e2fgvi/E2FGVI-HQ-CVPR22.pth
     https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3

Advantages over ProPainter:
  - Single end-to-end model: no RAFT + FlowCompleteNet chain
  - Flow and inpainting trained jointly → more consistent fills
  - Less chunk-boundary flickering at moderate neighbor_length
  - Only ~200 MB model file (vs ~500 MB for ProPainter)

VRAM guidance:
  Resolution  | NEIGHBOR_LENGTH | MAX_LOAD_NUM | VRAM used
  ------------|-----------------|--------------|----------
  720p        |       10        |      80      | ~3-4 GB   ← safe on 8GB
  720p        |       20        |      80      | ~5-6 GB   ← safe on 8GB
  1080p       |       10        |      60      | ~4-5 GB   ← safe on 8GB  ✓ ACTIVE
  1080p       |       20        |      60      | ~6-7 GB   ← tight on 8GB
  1080p       |       20        |      80      | ~7-8 GB   ← may OOM on 8GB
"""
# Maximum number of frames fed to E2FGVIInpaint per subtitle segment.
# 8GB 1080p：60 帧/批 ~6–7GB；更少分段接缝。OOM 时改为 50
if sys_str == "Darwin":  # Mac (MPS / Apple Silicon)
    E2FGVI_MAX_LOAD_NUM = 12       # ~6 GB per batch on MPS — safe for 16 GB unified memory
    E2FGVI_NEIGHBOR_LENGTH = 15    # good temporal context without excess memory
    E2FGVI_CROP_MARGIN = 96
    E2FGVI_MAX_CROP_SIDE = 1024    # safe for MPS
    E2FGVI_STREAM_MAX_LOAD = 12
else:  # Windows / Linux (CUDA)
    E2FGVI_MAX_LOAD_NUM = 60       # high quality on 8GB+ VRAM
    E2FGVI_NEIGHBOR_LENGTH = 35
    E2FGVI_CROP_MARGIN = 128
    E2FGVI_MAX_CROP_SIDE = 1600
    E2FGVI_STREAM_MAX_LOAD = 32
# 更密的参考帧，复杂背景更稳
E2FGVI_REF_STRIDE = 2
E2FGVI_USE_FP16 = True             # CUDA only (MPS FP16 disabled in e2fgvi_inpaint.py)
# Must stay True: this build loads only the HQ network (not model/e2fgvi.py).
E2FGVI_USE_HQ = True
# PySceneDetect for segment splits; slow at 0% progress. GUI Advanced.
E2FGVI_FORCE_SCENE_DETECT = False
# ×××××××××× InpaintMode.E2FGVI算法设置 end ××××××××××

# ×××××××××× InpaintMode.LAMA算法设置 start ××××××××××
# 是否开启极速模式，开启后不保证inpaint效果，仅仅对包含文本的区域文本进行去除
LAMA_SUPER_FAST = False
# ×××××××××× InpaintMode.LAMA算法设置 end ××××××××××
# ×××××××××××××××××××× [可以改] end ××××××××××××××××××××
