import os
import cv2
import glob
import numpy as np
# numpy>=1.24 호환용 임시 션트
if not hasattr(np, "bool"):
    np.bool = np.bool_
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

from skimage import io, color, img_as_float
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects
from albumentations import Normalize, Compose
from albumentations.pytorch import ToTensorV2
from skimage.morphology import skeletonize
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None


# ============================
# 사용자 설정
# ============================
SRC_DIR = r"C:\Users\이채영\Downloads\crack_upgrade_20260709\crack_images1"
DST_DIR = r"C:\Users\이채영\Downloads\crack_upgrade_20260709\crack_masks1"

# 기존처럼 마스크가 없는 파일부터 시작합니다.
# P로 이전 파일에 돌아갈 때는 저장된 마스크를 불러와 오버레이합니다.
REVIEW_EXISTING_MASKS = False

MODEL_PATH_HC = r"C:\Users\이채영\Downloads\crack_upgrade_20260709\model\HC_unetpp_Quebec117_50"  # HC-Unet++ 가중치 경로로 설정
MODEL_PATH_UNETPP = r"C:\Users\이채영\Downloads\crack_upgrade_20260709\model\unetpp_204_crack_epoch_300.pth"  # UNet++ 가중치 경로

WINDOW_SIZE = (512, 512)
STRIDE = (WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2)
UNETPP_WINDOW_SIZE = (256, 256)

WINDOW_MAIN = "Sato+CLAHE + (Optional) Model Overlay  [L: Original Color, R: Color+Mask Overlay]"
WINDOW_CTRL = "Control (All Params Visible)"
WINDOW_INFO = "Info / Shortcuts / Edit Mode"

VIEW_SCALE = 1.0  # 원본 100% 기본
MAIN_VIEW_W = 1680
MAIN_VIEW_H = 980
SAVE_EMPTY_ON_SKIP = False

TB_SIGMA = "01 Sigma max"
TB_BLACK = "02 Black ridges"
TB_CLAHE_ON = "03 CLAHE on"
TB_CLAHE_CLIP = "04 CLAHE clip x10"
TB_CLAHE_TILE = "05 CLAHE tile"
TB_HYST_LOW = "06 Hyst low x1000"
TB_HYST_HIGH = "07 Hyst high x1000"
TB_MIN_SIZE = "08 Min object size"
TB_CLOSE_ON = "09 Morph close on"
TB_CLOSE_K = "10 Close kernel"
TB_ALPHA = "11 Overlay alpha"
TB_BRUSH = "12 Brush px"
TB_EDIT_MODE = "13 Edit mode"
TB_ZOOM_MODE = "14 Zoom pan"
TB_USE_MODEL = "15 Model on"
TB_MODEL_TYPE = "16 Model type"
TB_MODEL_THR = "17 Model thr x1000"
TB_SKEL_ON = "18 Skeleton on"
TB_SKEL_THICK = "19 Skeleton thick"
TB_HELP_ITEM = "20 Help item"

HELP_ITEMS = [
    ("전체", "번호가 붙은 조절 항목을 선택하면 이 창에서 한글 설명을 볼 수 있습니다."),
    ("Sigma max", "Sato 필터가 확인할 균열 선의 최대 두께 범위입니다. 값이 클수록 더 굵은 선 구조까지 찾습니다."),
    ("Black ridges", "어두운 균열을 찾을지 선택합니다. 포장 균열처럼 배경보다 어두운 선이면 1이 보통 맞습니다."),
    ("CLAHE on", "명암 대비 향상 전처리 사용 여부입니다. 균열과 배경 대비가 약할 때 켜면 도움이 됩니다."),
    ("CLAHE clip", "CLAHE 대비 강화 정도입니다. 너무 높으면 노이즈도 같이 강해질 수 있습니다."),
    ("CLAHE tile", "CLAHE를 적용할 지역 블록 크기입니다. 지역별 밝기 차이가 큰 이미지에서 조절합니다."),
    ("Hyst low", "Sato 결과의 약한 후보를 포함할 낮은 임계값입니다. 낮추면 더 많이 잡히지만 노이즈도 늘 수 있습니다."),
    ("Hyst high", "확실한 균열 후보를 정하는 높은 임계값입니다. 높이면 더 엄격하게 잡습니다."),
    ("Min object size", "작은 잡음을 제거하는 최소 객체 크기입니다. 값이 크면 작은 점/짧은 선이 사라집니다."),
    ("Morph close", "끊어진 균열 조각을 이어주는 후처리 사용 여부입니다."),
    ("Close kernel", "Morph close의 커널 크기입니다. 값이 클수록 더 멀리 떨어진 조각까지 이어질 수 있습니다."),
    ("Overlay alpha", "오른쪽 미리보기에서 마스크 색이 보이는 진하기입니다. 저장 마스크에는 영향 없습니다."),
    ("Brush px", "수동 편집 브러쉬 선 두께입니다. 1이면 실제 1픽셀 선으로 그립니다."),
    ("Edit mode", "0은 지우기/복원, 1은 흰색 균열 직접 그리기 모드입니다. D 키로 전환할 수 있습니다."),
    ("Zoom pan", "1이면 휠로 확대/축소, 오른쪽 드래그로 화면 이동합니다. V 키로 전환할 수 있습니다."),
    ("Model on", "모델 예측 마스크를 Sato 필터 결과와 함께 사용할지 선택합니다. M 키로 전환할 수 있습니다."),
    ("Model type", "0은 HC-Unet++, 1은 UNet++ 모델을 사용합니다."),
    ("Model threshold", "모델 sigmoid 확률을 마스크로 바꾸는 임계값입니다. 낮추면 더 많이 검출됩니다."),
    ("Skeleton on", "저장 전 필터/모델 결과를 중심선 형태로 얇게 만드는 옵션입니다."),
    ("Skeleton thick", "Skeleton 결과의 두께입니다. 직접 그린 선은 이 옵션과 별도로 브러쉬 두께를 유지합니다."),
]

LEGACY_DEFAULTS = {
    TB_SIGMA: 4,
    TB_BLACK: 1,
    TB_CLAHE_ON: 1,
    TB_CLAHE_CLIP: 20,
    TB_CLAHE_TILE: 4,
    TB_HYST_LOW: 100,
    TB_HYST_HIGH: 200,
    TB_MIN_SIZE: 30,
    TB_CLOSE_ON: 0,
    TB_CLOSE_K: 1,
    TB_ALPHA: 45,
    TB_BRUSH: 3,
    TB_EDIT_MODE: 0,
    TB_ZOOM_MODE: 0,
    TB_USE_MODEL: 0,
    TB_MODEL_TYPE: 1,
    TB_MODEL_THR: 500,
    TB_SKEL_ON: 0,
    TB_SKEL_THICK: 1,
    TB_HELP_ITEM: 0,
}
# ============================

# ----------------------------
# HC-Unet++ (추론용 경량 정의)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class DPFFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c = out_channels // 2
        self.upper = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.lower = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
    def forward(self, x): return torch.cat([self.upper(x), self.lower(x)], dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(self.pool(x))

class BlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([[1.,2.,1.],[2.,4.,2.],[1.,2.,1.]])
        kernel = kernel / kernel.sum()
        self.register_buffer('filter', kernel[None,None,:,:].repeat(channels,1,1,1))
        self.channels = channels
    def forward(self, x): return F.conv2d(x, self.filter, stride=2, padding=1, groups=self.channels)

class HCUnetPlusPlus(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        f = [64, 128, 256, 512, 1024]
        self.pool0 = BlurPool(f[0]); self.pool1 = BlurPool(f[1])
        self.pool2 = BlurPool(f[2]); self.pool3 = BlurPool(f[3])
        self.up1_0 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(f[4], f[4], 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)

        self.conv0_0 = ConvBlock(input_channels, f[0], f[0])
        self.conv1_0 = ConvBlock(f[0], f[1], f[1])
        self.conv2_0 = ConvBlock(f[1], f[2], f[2])
        self.conv3_0 = ConvBlock(f[2], f[3], f[3])
        self.conv4_0 = DPFFB(f[3], f[4])

        self.conv0_1 = ConvBlock(f[0] + f[1], f[0], f[0])
        self.conv1_1 = ConvBlock(f[1] + f[2], f[1], f[1])
        self.conv2_1 = ConvBlock(f[2] + f[3], f[2], f[2])
        self.conv3_1 = ConvBlock(f[3] + f[4], f[3], f[3])

        self.conv0_2 = ConvBlock(f[0]*2 + f[1], f[0], f[0])
        self.conv1_2 = ConvBlock(f[1]*2 + f[2], f[1], f[1])
        self.conv2_2 = ConvBlock(f[2]*2 + f[3], f[2], f[2])

        self.conv0_3 = ConvBlock(f[0]*3 + f[1], f[0], f[0])
        self.conv1_3 = ConvBlock(f[1]*3 + f[2], f[1], f[1])
        self.conv0_4 = ConvBlock(f[0]*4 + f[1], f[0], f[0])

        self.se3_1 = SEBlock(f[3])
        self.output = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))
        x3_1 = self.se3_1(x3_1)

        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))

        return self.output(x0_4)

# ----------------------------
# UNet++ (standard inference definition)
# ----------------------------
class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super().__init__()
        f = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool2d(2, 2)
        self.up1_0 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(f[4], f[4], 2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up3_1 = nn.ConvTranspose2d(f[3], f[3], 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)
        self.up2_2 = nn.ConvTranspose2d(f[2], f[2], 2, stride=2)
        self.up1_3 = nn.ConvTranspose2d(f[1], f[1], 2, stride=2)

        self.conv0_0 = ConvBlock(input_channels, f[0], f[0])
        self.conv1_0 = ConvBlock(f[0], f[1], f[1])
        self.conv2_0 = ConvBlock(f[1], f[2], f[2])
        self.conv3_0 = ConvBlock(f[2], f[3], f[3])
        self.conv4_0 = ConvBlock(f[3], f[4], f[4])

        self.conv0_1 = ConvBlock(f[0] + f[1], f[0], f[0])
        self.conv1_1 = ConvBlock(f[1] + f[2], f[1], f[1])
        self.conv2_1 = ConvBlock(f[2] + f[3], f[2], f[2])
        self.conv3_1 = ConvBlock(f[3] + f[4], f[3], f[3])

        self.conv0_2 = ConvBlock(f[0]*2 + f[1], f[0], f[0])
        self.conv1_2 = ConvBlock(f[1]*2 + f[2], f[1], f[1])
        self.conv2_2 = ConvBlock(f[2]*2 + f[3], f[2], f[2])

        self.conv0_3 = ConvBlock(f[0]*3 + f[1], f[0], f[0])
        self.conv1_3 = ConvBlock(f[1]*3 + f[2], f[1], f[1])
        self.conv0_4 = ConvBlock(f[0]*4 + f[1], f[0], f[0])
        self.output = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1(x3_1)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], dim=1))
        return self.output(x0_4)

# ----------------------------
# 유틸
# ----------------------------
def list_images(folder):
    exts = ("*.jpg","*.jpeg","*.png","*.tif","*.tiff","*.bmp")
    files = []
    for e in exts: files += glob.glob(os.path.join(folder, e))
    return sorted(files)

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def to_u8_gray(f01): return (np.clip(f01, 0, 1) * 255).astype(np.uint8)

def imwrite_unicode(path, image):
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError as e:
        print(f"[!] save failed with OSError: {e}")
        return False

def colorize_gray(f01, scale=1.0):
    g8 = to_u8_gray(f01)
    if scale != 1.0:
        g8 = cv2.resize(g8, (int(g8.shape[1]*scale), int(g8.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)

def overlay_mask(gray_f01, mask_u8, alpha=0.45, color=(0,255,255), scale=1.0):
    base = colorize_gray(gray_f01, scale=scale)
    mu = cv2.resize(mask_u8, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else mask_u8
    out = base.copy()
    sel = mu > 0
    out[sel] = (out[sel]*(1-alpha) + np.array(color)*alpha).astype(np.uint8)
    return out

def overlay_mask_on_bgr(bgr_img, mask_u8, alpha=0.45, color=(0,255,255), scale=1.0):
    base = bgr_img
    if scale != 1.0:
        base = cv2.resize(base, (int(base.shape[1]*scale), int(base.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    mu = cv2.resize(mask_u8, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else mask_u8
    out = base.astype(np.float32)
    sel = mu > 0
    out[sel] = out[sel]*(1.0 - alpha) + np.array(color, dtype=np.float32)*alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def bgr_scaled(bgr, scale=1.0):
    if scale == 1.0: return bgr
    return cv2.resize(bgr, (int(bgr.shape[1]*scale), int(bgr.shape[0]*scale)), interpolation=cv2.INTER_AREA)

def get_gaussian_kernel(window_size, sigma_scale=1./8):
    y, x = np.mgrid[0:window_size[0], 0:window_size[1]]
    cy, cx = (window_size[0]-1)/2., (window_size[1]-1)/2.
    sy, sx = window_size[0]*sigma_scale, window_size[1]*sigma_scale
    return np.exp(-((x-cx)**2/(2*sx**2) + (y-cy)**2/(2*sy**2)))

def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    return state

def build_model(model_type, device):
    if model_type == 0:
        return HCUnetPlusPlus(num_classes=1, input_channels=3).to(device)

    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch가 설치되어 있지 않습니다. "
            "UNet++(Model Type 1)는 `pip install segmentation-models-pytorch` 후 사용할 수 있습니다."
        )

    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

def model_infer_config(model_type):
    if model_type == 1:
        return dict(
            window_size=UNETPP_WINDOW_SIZE,
            stride=UNETPP_WINDOW_SIZE,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            use_gaussian=False,
        )
    return dict(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        use_gaussian=True,
    )

def get_korean_font(size=18):
    if ImageFont is None:
        return None
    for path in (
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

# ----------------------------
# 앱
# ----------------------------
class App:
    def apply_skeleton(self, mask_u8, p):
        """
         mask_u8 : 최종 편집까지 끝난 0/255 마스크
        p["use_skel"] : 스켈레톤 on/off
        p["skel_thick"] : 두께 단계 (1 = 가장 얇음, 2부터 서서히 두꺼워짐)
        """

        if not p.get("use_skel", False):
            return mask_u8

        if mask_u8.max() == 0:
            return mask_u8  # 어차피 빈 마스크면 그대로
        
        # 1) 스켈레톤(1픽셀 라인)
        orig = mask_u8
        mask_bool = orig > 0
        skel_bool = skeletonize(mask_bool)
        skel = (skel_bool.astype(np.uint8) * 255)

        # 2) 두께 조절
        t = int(p.get("skel_thick", 1))
        # thickness=1 -> radius=0 (dilate 없음)
        radius = max(0, t - 1)


        if radius > 0:
            # 예: radius=1 => 3x3, radius=2 => 5x5
            ksize = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            skel = cv2.dilate(skel, kernel, iterations=1)

            # 원래 마스크보다 두꺼워지지 않게 클리핑
            skel = cv2.bitwise_and(skel, orig)

        return skel
    


    def __init__(self, files, dst_dir):
        self.files = files
        self.dst   = dst_dir
        ensure_dir(self.dst)

        self.done = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(self.dst, "*.png"))}
        self.base_to_index = {os.path.splitext(os.path.basename(p))[0]: i for i, p in enumerate(self.files)}

        self.idx = 0
        self.rgb = None
        self.bgr = None
        self.gray = None
        self.H = self.W = 0

        # 편집 마스크: draw_mask는 흰색 크랙 추가, erase_mask는 최종 마스크 제거
        self.erase_mask = None
        self.draw_mask = None
        self.review_mask = None
        self.viewing_existing_mask = False
        self.undo_stack = []
        self._stroke_tmp = None
        self._erase_before = None
        self._draw_before = None
        self._last_paint_pos = None

        # 모델 관련
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_type = None
        self.model_loaded = False
        self.model_prob = None
        self.model_bin  = None
        self.model_cached_for = None
        self.model_thr_cached = None

        # 캐시
        self.gray_proc_cache = None; self.gray_proc_key = None
        self.sato_cache = None; self.sato_key = None

        # 보기 배율 상태
        self.view_scale = VIEW_SCALE
        self.pan_x = 0
        self.pan_y = 0
        self._panning = False
        self._pan_start_mouse = None
        self._pan_start_offset = None
        self._last_scaled_panel_shape = None
        self._help_hit_rows = []
        self.font_title = get_korean_font(22)
        self.font_text = get_korean_font(17)
        self.font_small = get_korean_font(15)

        # UI
        self.setup_windows()
        if not REVIEW_EXISTING_MASKS:
            self.jump_to_first_unlabeled()  # ← 내부 로직을 '마지막 저장 이후'로 변경
        if 0 <= self.idx < len(self.files): self.load(self.files[self.idx])

    # ---------- 경로/상태 ----------
    def out_path_for(self, idx):
        name = os.path.splitext(os.path.basename(self.files[idx]))[0] + ".png"
        return os.path.join(self.dst, name)
    def out_path_current(self): return self.out_path_for(self.idx)
    def mask_exists_for(self, idx):
        base = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return os.path.exists(self.out_path_for(idx)) or (base in self.done)

    def jump_to_first_unlabeled(self):
        """
        변경점:
        - DST_DIR 내 *.png 중 '수정시간이 가장 최근'인 마스크 파일을 찾아
          그 파일에 해당하는 원본의 '다음 인덱스'에서 시작.
        - 그 다음 인덱스가 이미 라벨링 되어 있다면, 최초 미라벨링 지점까지 전진.
        """
        mask_paths = glob.glob(os.path.join(self.dst, "*.png"))
        start = 0
        if mask_paths:
            last_mask = max(mask_paths, key=os.path.getmtime)  # 최신 저장본
            base = os.path.splitext(os.path.basename(last_mask))[0]
            last_idx = self.base_to_index.get(base, -1)
            start = last_idx + 1

        i = start
        while 0 <= i < len(self.files) and self.mask_exists_for(i):
            i += 1

        if 0 <= i < len(self.files):
            self.idx = i
            print(f"[*] Resuming AFTER last saved → #{self.idx+1}: {os.path.basename(self.files[self.idx])}")
            return

        self.idx = len(self.files)
        print("[*] 모든 이미지가 이미 라벨링되어 있습니다.]")

    def advance_to_next_unlabeled(self, direction=+1, include_current=False):
        if not self.files: return False
        i = self.idx if include_current else self.idx + direction
        if direction > 0 and not REVIEW_EXISTING_MASKS:
            while 0 <= i < len(self.files) and self.mask_exists_for(i): i += direction
        if 0 <= i < len(self.files):
            self.idx = i
            self.viewing_existing_mask = direction < 0 and self.mask_exists_for(i)
            self.load(self.files[self.idx])
            return True
        print("[*] 더 이상 진행할 미라벨링 이미지가 없습니다."); return False

    # ---------- 상태바/오버레이 ----------
    def show_status(self, text, ms=2500):
        try:
            cv2.displayStatusBar(WINDOW_CTRL, text, ms)
        except Exception:
            try:
                cv2.displayOverlay(WINDOW_CTRL, text, ms)
            except Exception:
                print(text)

    def make_cb(self, name):
        def _cb(v, _n=name):
            self.show_status(f"{_n}: {v}")
        return _cb

    def show_help(self):
        help_text = (
            "Controls:\n"
            "  Mouse: EditMode=Erase → L-drag=지움, R-drag=복원\n"
            "         EditMode=Draw  → L-drag=흰색 크랙 그리기, R-drag=직접 그린 부분 지움\n"
            "         Zoom/Pan ON    → Wheel=확대/축소, R-drag=화면 이동\n"
            "  Keys : S=Save&Next  N=Next  P=Prev  Z=Undo  C=Clear manual edits\n"
            "         M=Toggle Model  D=Toggle Draw/Erase  V=Toggle Zoom/Pan  H=Help  Q/Esc=Quit\n"
            "Trackbars:\n"
            "  01~19: filter/model/edit controls\n"
            "  20 Help item: select a Korean description in the Info window\n"
            "  You can also click numbered rows in the Info window.\n"
        )
        self.show_status(help_text, 0)

    # ---------- 창/트랙바 ----------
    def setup_windows(self):
        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_MAIN, MAIN_VIEW_W, MAIN_VIEW_H)

        cv2.namedWindow(WINDOW_CTRL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_CTRL, 760, 760)
        cv2.moveWindow(WINDOW_CTRL, 40, 900)

        cv2.namedWindow(WINDOW_INFO, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_INFO, 780, 520)
        cv2.moveWindow(WINDOW_INFO, 830, 900)

        self.show_help()

        cv2.createTrackbar(TB_SIGMA,      WINDOW_CTRL, LEGACY_DEFAULTS[TB_SIGMA],      8,    self.make_cb(TB_SIGMA))
        cv2.createTrackbar(TB_BLACK,      WINDOW_CTRL, LEGACY_DEFAULTS[TB_BLACK],      1,    self.make_cb(TB_BLACK))
        cv2.createTrackbar(TB_CLAHE_ON,   WINDOW_CTRL, LEGACY_DEFAULTS[TB_CLAHE_ON],   1,    self.make_cb(TB_CLAHE_ON))
        cv2.createTrackbar(TB_CLAHE_CLIP, WINDOW_CTRL, LEGACY_DEFAULTS[TB_CLAHE_CLIP], 100,  self.make_cb(TB_CLAHE_CLIP))
        cv2.createTrackbar(TB_CLAHE_TILE, WINDOW_CTRL, LEGACY_DEFAULTS[TB_CLAHE_TILE], 20,   self.make_cb(TB_CLAHE_TILE))

        cv2.createTrackbar(TB_HYST_LOW,   WINDOW_CTRL, LEGACY_DEFAULTS[TB_HYST_LOW],   1000, self.make_cb(TB_HYST_LOW))
        cv2.createTrackbar(TB_HYST_HIGH,  WINDOW_CTRL, LEGACY_DEFAULTS[TB_HYST_HIGH],  1000, self.make_cb(TB_HYST_HIGH))

        cv2.createTrackbar(TB_MIN_SIZE,   WINDOW_CTRL, LEGACY_DEFAULTS[TB_MIN_SIZE],   5000, self.make_cb(TB_MIN_SIZE))
        cv2.createTrackbar(TB_CLOSE_ON,   WINDOW_CTRL, LEGACY_DEFAULTS[TB_CLOSE_ON],   1,    self.make_cb(TB_CLOSE_ON))
        cv2.createTrackbar(TB_CLOSE_K,    WINDOW_CTRL, LEGACY_DEFAULTS[TB_CLOSE_K],    5,    self.make_cb(TB_CLOSE_K))
        cv2.createTrackbar(TB_ALPHA,      WINDOW_CTRL, LEGACY_DEFAULTS[TB_ALPHA],      100,  self.make_cb(TB_ALPHA))
        cv2.createTrackbar(TB_BRUSH,      WINDOW_CTRL, LEGACY_DEFAULTS[TB_BRUSH],      80,   self.make_cb(TB_BRUSH))
        cv2.createTrackbar(TB_EDIT_MODE,  WINDOW_CTRL, LEGACY_DEFAULTS[TB_EDIT_MODE],  1,    self.make_cb(TB_EDIT_MODE))
        cv2.createTrackbar(TB_ZOOM_MODE,  WINDOW_CTRL, LEGACY_DEFAULTS[TB_ZOOM_MODE],  1,    self.make_cb(TB_ZOOM_MODE))

        cv2.createTrackbar(TB_USE_MODEL,  WINDOW_CTRL, LEGACY_DEFAULTS[TB_USE_MODEL],  1,    self.make_cb(TB_USE_MODEL))
        cv2.createTrackbar(TB_MODEL_TYPE, WINDOW_CTRL, LEGACY_DEFAULTS[TB_MODEL_TYPE], 1,    self.make_cb(TB_MODEL_TYPE))
        cv2.createTrackbar(TB_MODEL_THR,  WINDOW_CTRL, LEGACY_DEFAULTS[TB_MODEL_THR],  1000, self.make_cb(TB_MODEL_THR))

        cv2.createTrackbar(TB_SKEL_ON,    WINDOW_CTRL, LEGACY_DEFAULTS[TB_SKEL_ON],    1,    self.make_cb(TB_SKEL_ON))
        cv2.createTrackbar(TB_SKEL_THICK, WINDOW_CTRL, LEGACY_DEFAULTS[TB_SKEL_THICK], 5,    self.make_cb(TB_SKEL_THICK))
        cv2.createTrackbar(TB_HELP_ITEM,  WINDOW_CTRL, LEGACY_DEFAULTS[TB_HELP_ITEM],  len(HELP_ITEMS) - 1, self.make_cb(TB_HELP_ITEM))

        cv2.setMouseCallback(WINDOW_MAIN, self.on_mouse_main)
        cv2.setMouseCallback(WINDOW_INFO, self.on_mouse_info)

    def get_params(self):
        sigma = max(1, cv2.getTrackbarPos(TB_SIGMA, WINDOW_CTRL))
        black = bool(cv2.getTrackbarPos(TB_BLACK, WINDOW_CTRL))
        use_clahe = bool(cv2.getTrackbarPos(TB_CLAHE_ON, WINDOW_CTRL))
        clip = max(1, cv2.getTrackbarPos(TB_CLAHE_CLIP, WINDOW_CTRL)) / 10.0
        tile = 2*max(1, cv2.getTrackbarPos(TB_CLAHE_TILE, WINDOW_CTRL)) + 2

        low  = cv2.getTrackbarPos(TB_HYST_LOW, WINDOW_CTRL)  / 1000.0
        high = cv2.getTrackbarPos(TB_HYST_HIGH, WINDOW_CTRL) / 1000.0
        if high < low: high = min(1.0, low + 0.01)

        min_size  = cv2.getTrackbarPos(TB_MIN_SIZE, WINDOW_CTRL)
        use_close = bool(cv2.getTrackbarPos(TB_CLOSE_ON, WINDOW_CTRL))
        close_k   = 2*max(1, cv2.getTrackbarPos(TB_CLOSE_K, WINDOW_CTRL)) + 1
        alpha     = cv2.getTrackbarPos(TB_ALPHA, WINDOW_CTRL) / 100.0
        brush     = max(1, cv2.getTrackbarPos(TB_BRUSH, WINDOW_CTRL))
        edit_mode = cv2.getTrackbarPos(TB_EDIT_MODE, WINDOW_CTRL)
        zoom_pan_mode = bool(cv2.getTrackbarPos(TB_ZOOM_MODE, WINDOW_CTRL))

        use_model = bool(cv2.getTrackbarPos(TB_USE_MODEL, WINDOW_CTRL))
        mthr      = cv2.getTrackbarPos(TB_MODEL_THR, WINDOW_CTRL) / 1000.0

        use_skel = bool(cv2.getTrackbarPos(TB_SKEL_ON, WINDOW_CTRL))
        skel_thick = cv2.getTrackbarPos(TB_SKEL_THICK, WINDOW_CTRL)
        skel_thick = max(1, skel_thick)

        model_type = cv2.getTrackbarPos(TB_MODEL_TYPE, WINDOW_CTRL)
        help_item = cv2.getTrackbarPos(TB_HELP_ITEM, WINDOW_CTRL)
        return dict(
            sigma=sigma, black=black, use_clahe=use_clahe, clip=clip, tile=tile,
            low=low, high=high, min_size=min_size, use_close=use_close,
            close_k=close_k, alpha=alpha, brush=brush,
            edit_mode=edit_mode, zoom_pan_mode=zoom_pan_mode,
            use_model=use_model, model_type=model_type, mthr=mthr, view_scale=self.view_scale,
            use_skel=use_skel, skel_thick=skel_thick, help_item=help_item
        )

    # ---------- 로드 ----------
    def load(self, path):
        img = io.imread(path)
        if img.ndim == 3:
            self.rgb = img
            self.gray = np.clip(color.rgb2gray(img), 0, 1)
            self.bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            self.rgb = None
            self.gray = img_as_float(img)
            self.bgr = cv2.cvtColor(to_u8_gray(self.gray), cv2.COLOR_GRAY2BGR)

        self.H, self.W = self.gray.shape[:2]
        self.erase_mask = np.zeros((self.H, self.W), np.uint8)
        self.draw_mask = np.zeros((self.H, self.W), np.uint8)
        self.review_mask = None
        if REVIEW_EXISTING_MASKS or self.viewing_existing_mask:
            mask_path = self.out_path_current()
            if os.path.exists(mask_path):
                review_mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if review_mask is None:
                    print(f"[!] 마스크를 읽을 수 없습니다: {mask_path}")
                else:
                    if review_mask.shape != (self.H, self.W):
                        print(f"[!] 크기 불일치, 화면 표시용으로 리사이즈: {review_mask.shape} -> {(self.H, self.W)}")
                        review_mask = cv2.resize(review_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                    self.review_mask = np.where(review_mask > 0, 255, 0).astype(np.uint8)
            else:
                print(f"[!] 대응 마스크 없음: {mask_path}")
        self.undo_stack.clear()
        self._stroke_tmp = None
        self._erase_before = None
        self._draw_before = None
        self._last_paint_pos = None
        self.reset_view_to_fit()
        self._panning = False
        self._pan_start_mouse = None
        self._pan_start_offset = None
        self._last_scaled_panel_shape = None

        # 캐시 초기화
        self.gray_proc_cache = None; self.gray_proc_key = None
        self.sato_cache = None; self.sato_key = None
        self.model_prob = None; self.model_bin = None
        self.model_cached_for = None; self.model_thr_cached = None

        try:
            cv2.setWindowTitle(WINDOW_MAIN,
                f"{WINDOW_MAIN}   [{self.idx+1}/{len(self.files)}] "
                f"in: {os.path.basename(self.files[self.idx])}  →  out: {os.path.basename(self.out_path_current())}")
        except Exception:
            pass

    def reset_view_to_fit(self):
        if self.W <= 0 or self.H <= 0:
            self.view_scale = VIEW_SCALE
        else:
            panel_w = self.W * 2
            panel_h = self.H
            self.view_scale = min(MAIN_VIEW_W / panel_w, MAIN_VIEW_H / panel_h)
            self.view_scale = float(np.clip(self.view_scale, 0.25, 8.0))
        self.pan_x = 0
        self.pan_y = 0

    # ---------- 저장 ----------
    def save_final(self):
        p = self.get_params()
        if self.viewing_existing_mask and self.review_mask is not None:
            final = self.compose_final(self.review_mask)
        else:
            gray_proc = self.compute_gray_proc_cached(p)
            _, base_sato = self.compute_sato_cached(p, gray_proc)
            base_combined = self.combine_with_model(p, base_sato)
            base_combined = self.apply_skeleton(base_combined, p)
            final = self.compose_final(base_combined)


        out_path = self.out_path_current()
        ok = imwrite_unicode(out_path, final)
        if ok:
            base = os.path.splitext(os.path.basename(self.files[self.idx]))[0]
            self.done.add(base); print(f"[✓] saved: {os.path.abspath(out_path)}")
            self.show_status(f"Saved: {os.path.basename(out_path)}")
            return True
        else:
            print(f"[!] save failed: {os.path.abspath(out_path)}")
            self.show_status(f"Save failed: {os.path.abspath(out_path)}", 4000)
            return False

    def save_empty(self):
        out_path = self.out_path_current()
        if not os.path.exists(out_path):
            empty = np.zeros((self.H, self.W), np.uint8)
            ok = imwrite_unicode(out_path, empty)
            if ok:
                base = os.path.splitext(os.path.basename(self.files[self.idx]))[0]
                self.done.add(base); print(f"[✓] saved empty: {os.path.abspath(out_path)}")
                return True
            else:
                print(f"[!] save empty failed: {os.path.abspath(out_path)}")
                return False
        return True

    # ---------- 캐시 기반 처리 ----------
    def compute_gray_proc_cached(self, p):
        key = (p["use_clahe"], round(p["clip"],3), p["tile"])
        if self.gray_proc_key != key:
            if not p["use_clahe"]:
                self.gray_proc_cache = self.gray
            else:
                g8 = to_u8_gray(self.gray)
                clahe = cv2.createCLAHE(p["clip"], (p["tile"], p["tile"]))
                g8c = clahe.apply(g8)
                self.gray_proc_cache = g8c.astype(np.float32)/255.0
            self.gray_proc_key = key
        return self.gray_proc_cache

    def compute_sato_cached(self, p, gray_src):
        key = (p["sigma"], p["black"], round(p["low"],3), round(p["high"],3),
               int(p["min_size"]), p["use_close"], p["close_k"], self.gray_proc_key)
        if self.sato_key != key:
            sigmas = tuple(range(1, p["sigma"]+1)) or (1,)
            ridge = sato(gray_src, sigmas=sigmas, black_ridges=p["black"])
            rn = (ridge - ridge.min()) / (ridge.max() - ridge.min() + 1e-8)
            mask_bool = apply_hysteresis_threshold(rn, p["low"], p["high"])
            if p["min_size"] > 0:
                mask_bool = remove_small_objects(mask_bool, min_size=int(p["min_size"]))
            m8 = (mask_bool.astype(np.uint8) * 255)
            if p["use_close"]:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (p["close_k"], p["close_k"]))
                m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, k, iterations=1)
            self.sato_cache = (rn, m8)
            self.sato_key = key
        return self.sato_cache

    # ---------- 모델 ----------
    def lazy_load_model(self, model_type):
        path = MODEL_PATH_HC if model_type == 0 else MODEL_PATH_UNETPP
        if self.model_loaded and self.model_type == model_type and os.path.exists(path):
            return True
        if not os.path.exists(path):
            print(f"[!] MODEL_PATH not found: {path}")
            return False
        try:
            model = build_model(model_type, self.device)
            ckpt = torch.load(path, map_location=self.device)
            state = extract_state_dict(ckpt)
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as e:
                print("[!] strict=True model load failed.")
                print("    이 메시지가 UNet++(Model Type 1)에서 나오면, "
                      "체크포인트 구조와 모델 정의가 아직 맞지 않는 것입니다.")
                raise e
            model.eval()
            self.model = model
            self.model_type = model_type
            self.model_loaded = True
            self.model_prob = None
            self.model_bin = None
            self.model_cached_for = None
            self.model_thr_cached = None
            print(f"[✓] Model loaded on {self.device}: {path} ({'HC-Unet++' if model_type == 0 else 'UNet++'})")
            return True
        except Exception as e:
            print(f"[!] Failed to load model: {e}"); return False

    def predict_prob_once(self):
        if self.model_cached_for == self.files[self.idx] and self.model_prob is not None:
            return self.model_prob
        cfg = model_infer_config(self.model_type)
        window_size = cfg["window_size"]
        stride = cfg["stride"]
        transform = Compose([Normalize(mean=cfg["mean"], std=cfg["std"]), ToTensorV2()])
        image_rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        pad_h = (window_size[0] - h % window_size[0]) % window_size[0]
        pad_w = (window_size[1] - w % window_size[1]) % window_size[1]
        padded = np.pad(image_rgb, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
        ph, pw, _ = padded.shape

        out = np.zeros((ph, pw), np.float32)
        cnt = np.zeros((ph, pw), np.float32)
        weight = get_gaussian_kernel(window_size) if cfg["use_gaussian"] else np.ones(window_size, np.float32)

        y_steps = list(range(0, ph - window_size[0] + 1, stride[0]))
        if y_steps[-1] != ph - window_size[0]: y_steps.append(ph - window_size[0])
        x_steps = list(range(0, pw - window_size[1] + 1, stride[1]))
        if x_steps[-1] != pw - window_size[1]: x_steps.append(pw - window_size[1])

        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    crop = padded[y:y+window_size[0], x:x+window_size[1]]
                    inp = transform(image=crop)['image'].unsqueeze(0).to(self.device)
                    prob = torch.sigmoid(self.model(inp)).squeeze().detach().cpu().numpy()
                    out[y:y+window_size[0], x:x+window_size[1]] += prob * weight
                    cnt[y:y+window_size[0], x:x+window_size[1]] += weight

        prob_map = (out / np.maximum(cnt, 1e-6))[:h, :w].astype(np.float32)
        self.model_prob = prob_map
        self.model_cached_for = self.files[self.idx]
        return prob_map

    def ensure_model_bin(self, p):
        if not p["use_model"]:
            self.model_bin = None; return None
        if not self.model_loaded or self.model_type != p["model_type"]:
            if not self.lazy_load_model(p["model_type"]):
                self.model_bin = None; return None
        try:
            prob = self.predict_prob_once()
        except Exception as e:
            print(f"[!] Model inference failed: {e}")
            self.model_bin = None
            return None
        if self.model_thr_cached != p["mthr"] or self.model_bin is None:
            self.model_bin = (prob >= float(np.clip(p["mthr"], 0.0, 1.0))).astype(np.uint8) * 255
            self.model_thr_cached = p["mthr"]
        return self.model_bin

    # ---------- 결합/최종 ----------
    def combine_with_model(self, p, sato_base):
        m = self.ensure_model_bin(p)
        if m is None: return sato_base
        return cv2.bitwise_or(sato_base, m)

    def compose_final(self, base_combined):
        with_draw = cv2.bitwise_or(base_combined, self.draw_mask)
        final = cv2.bitwise_and(with_draw, cv2.bitwise_not(self.erase_mask))
        return np.where(final > 0, 255, 0).astype(np.uint8)

    # ---------- 그리기 ----------
    def draw_text(self, draw, xy, text, font, fill=(30, 30, 30)):
        if draw is not None:
            draw.text(xy, text, font=font, fill=fill)

    def wrap_text(self, text, font, max_width):
        if ImageDraw is None or font is None:
            return [text]
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lines, cur = [], ""
        for ch in text:
            candidate = cur + ch
            if probe.textlength(candidate, font=font) <= max_width or not cur:
                cur = candidate
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    def render_info_window(self, p):
        w, h = 780, 520
        mode_name = "DRAW white crack" if p["edit_mode"] == 1 else "ERASE / RESTORE"
        model_name = "HC-Unet++" if p["model_type"] == 0 else "UNet++"
        model_state = "ON" if p["use_model"] else "OFF"
        skel_state = "ON" if p["use_skel"] else "OFF"
        zoom_state = "ON" if p["zoom_pan_mode"] else "OFF"
        selected = int(np.clip(p.get("help_item", 0), 0, len(HELP_ITEMS) - 1))
        self._help_hit_rows = []

        if Image is not None:
            img = Image.new("RGB", (w, h), (246, 246, 246))
            draw = ImageDraw.Draw(img)
            self.draw_text(draw, (18, 14), "상태 / 도움말", self.font_title, (20, 20, 20))
            self.draw_text(draw, (18, 48), f"편집: {mode_name}   브러쉬: {p['brush']} px", self.font_text)
            self.draw_text(draw, (18, 74), f"모델: {model_state} / {model_name} / 임계값 {p['mthr']:.3f}", self.font_text)
            self.draw_text(draw, (18, 100), f"스켈레톤: {skel_state} / 두께 {p['skel_thick']}   줌: {zoom_state} / {self.view_scale*100:.0f}%", self.font_text)
            self.draw_text(draw, (18, 132), "왼쪽 번호를 클릭하거나 컨트롤 패널의 '20 Help item'을 조절하세요.", self.font_small, (70, 70, 70))

            left_x, row_y, row_h = 18, 166, 22
            for idx, (name, _) in enumerate(HELP_ITEMS):
                y = row_y + idx * row_h
                if y > h - row_h:
                    break
                self._help_hit_rows.append((idx, y, y + row_h))
                fill = (224, 238, 255) if idx == selected else (246, 246, 246)
                draw.rectangle((left_x - 4, y - 2, 260, y + row_h - 2), fill=fill)
                self.draw_text(draw, (left_x, y), f"{idx:02d}. {name}", self.font_small, (15, 80, 150) if idx == selected else (40, 40, 40))

            title, desc = HELP_ITEMS[selected]
            draw.rectangle((285, 160, w - 18, h - 18), outline=(210, 210, 210), width=1)
            self.draw_text(draw, (305, 178), f"{selected:02d}. {title}", self.font_title, (15, 80, 150))
            y = 220
            for line in self.wrap_text(desc, self.font_text, 430):
                self.draw_text(draw, (305, y), line, self.font_text, (35, 35, 35))
                y += 28
            shortcuts = [
                "S 저장 후 다음 / Z 되돌리기 / C 수동 편집 초기화",
                "D 그리기·지우기 전환 / V 줌·이동 전환 / M 모델 전환",
                "줌·이동 ON: 휠 확대·축소, 오른쪽 드래그 화면 이동",
            ]
            y = max(y + 18, 350)
            for line in shortcuts:
                self.draw_text(draw, (305, y), line, self.font_small, (85, 85, 85))
                y += 24
            canvas = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            canvas = np.full((h, w, 3), 245, dtype=np.uint8)
            cv2.putText(canvas, f"Help item: {selected:02d} {HELP_ITEMS[selected][0]}", (18, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(canvas, HELP_ITEMS[selected][1][:80], (18, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_INFO, canvas)

    def clamp_pan(self, scaled_w=None, scaled_h=None):
        if scaled_w is None or scaled_h is None:
            if self._last_scaled_panel_shape is None:
                return
            scaled_h, scaled_w = self._last_scaled_panel_shape[:2]
        self.pan_x = int(np.clip(self.pan_x, 0, max(0, scaled_w - MAIN_VIEW_W)))
        self.pan_y = int(np.clip(self.pan_y, 0, max(0, scaled_h - MAIN_VIEW_H)))

    def make_viewport(self, panel):
        zoom = float(np.clip(self.view_scale, 0.25, 8.0))
        scaled_w = max(1, int(round(panel.shape[1] * zoom)))
        scaled_h = max(1, int(round(panel.shape[0] * zoom)))
        interp = cv2.INTER_CUBIC if zoom >= 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(panel, (scaled_w, scaled_h), interpolation=interp)
        self._last_scaled_panel_shape = scaled.shape
        self.clamp_pan(scaled_w, scaled_h)

        viewport = np.zeros((MAIN_VIEW_H, MAIN_VIEW_W, 3), dtype=np.uint8)
        x0, y0 = self.pan_x, self.pan_y
        x1 = min(x0 + MAIN_VIEW_W, scaled_w)
        y1 = min(y0 + MAIN_VIEW_H, scaled_h)
        crop = scaled[y0:y1, x0:x1]
        viewport[:crop.shape[0], :crop.shape[1]] = crop
        return viewport

    def zoom_at(self, screen_x, screen_y, factor):
        old_zoom = float(self.view_scale)
        new_zoom = float(np.clip(old_zoom * factor, 0.25, 8.0))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        panel_x = (self.pan_x + screen_x) / old_zoom
        panel_y = (self.pan_y + screen_y) / old_zoom
        self.view_scale = new_zoom
        self.pan_x = int(round(panel_x * new_zoom - screen_x))
        self.pan_y = int(round(panel_y * new_zoom - screen_y))
        self.clamp_pan()

    def draw(self):
        if not (0 <= self.idx < len(self.files)):
            print("[*] 작업을 종료합니다. (모든 이미지 완료)")
            cv2.destroyAllWindows(); raise SystemExit

        if not REVIEW_EXISTING_MASKS and not self.viewing_existing_mask and self.mask_exists_for(self.idx):
            moved = self.advance_to_next_unlabeled(direction=+1, include_current=True)
            if not moved:
                print("[*] 작업을 종료합니다. (모든 이미지 완료)")
                cv2.destroyAllWindows(); raise SystemExit
            return

        p = self.get_params()
        self.render_info_window(p)

        if REVIEW_EXISTING_MASKS or self.viewing_existing_mask:
            base_mask = self.review_mask if self.review_mask is not None else np.zeros((self.H, self.W), np.uint8)
            final = self.compose_final(base_mask)
        else:
            gray_proc = self.compute_gray_proc_cached(p)
            _, base_sato = self.compute_sato_cached(p, gray_proc)
            base_combined = self.combine_with_model(p, base_sato)
            base_combined = self.apply_skeleton(base_combined, p)
            final = self.compose_final(base_combined)

        left  = self.bgr
        right = overlay_mask_on_bgr(self.bgr, final, alpha=p["alpha"], scale=1.0)

        panel = np.hstack([left, right])
        cv2.imshow(WINDOW_MAIN, self.make_viewport(panel))

    # ---------- 좌표/편집 ----------
    def mouse_to_image_xy(self, x, y):
        panel_x = (self.pan_x + x) / max(self.view_scale, 1e-6)
        panel_y = (self.pan_y + y) / max(self.view_scale, 1e-6)
        if not (0 <= panel_y < self.H and 0 <= panel_x < self.W * 2):
            return None
        ix = int(panel_x % self.W)
        iy = int(panel_y)
        if 0 <= ix < self.W and 0 <= iy < self.H:
            return (ix, iy)
        return None

    def wheel_delta(self, flags):
        try:
            return cv2.getMouseWheelDelta(flags)
        except AttributeError:
            return 1 if flags > 0 else -1

    def on_mouse_info(self, event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for idx, y0, y1 in self._help_hit_rows:
            if y0 <= my <= y1:
                try:
                    cv2.setTrackbarPos(TB_HELP_ITEM, WINDOW_CTRL, idx)
                except Exception:
                    pass
                return

    def paint_stroke(self, mask, prev_pos, pos, brush, value):
        brush = max(1, int(brush))
        if prev_pos is None:
            if brush == 1:
                mask[pos[1], pos[0]] = value
            else:
                radius = max(1, (brush - 1) // 2)
                cv2.circle(mask, pos, radius, value, -1, lineType=cv2.LINE_8)
            return

        cv2.line(mask, prev_pos, pos, value, thickness=brush, lineType=cv2.LINE_8)

    def on_mouse_main(self, event, mx, my, flags, param):
        p = self.get_params(); brush = p["brush"]

        if event == getattr(cv2, "EVENT_MOUSEWHEEL", 10) and p["zoom_pan_mode"]:
            factor = 1.15 if self.wheel_delta(flags) > 0 else 1.0 / 1.15
            self.zoom_at(mx, my, factor)
            return

        if p["zoom_pan_mode"]:
            if event == cv2.EVENT_RBUTTONDOWN:
                self._panning = True
                self._pan_start_mouse = (mx, my)
                self._pan_start_offset = (self.pan_x, self.pan_y)
                return
            if event == cv2.EVENT_MOUSEMOVE and self._panning and (flags & cv2.EVENT_FLAG_RBUTTON):
                sx, sy = self._pan_start_mouse
                ox, oy = self._pan_start_offset
                self.pan_x = ox - (mx - sx)
                self.pan_y = oy - (my - sy)
                self.clamp_pan()
                return
            if event == cv2.EVENT_RBUTTONUP:
                self._panning = False
                self._pan_start_mouse = None
                self._pan_start_offset = None
                return

        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._stroke_tmp = np.zeros((self.H, self.W), np.uint8)
            self._erase_before = self.erase_mask.copy()
            self._draw_before = self.draw_mask.copy()
            self._last_paint_pos = None

        if (event == cv2.EVENT_LBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
            pos = self.mouse_to_image_xy(mx, my)
            if pos is not None:
                prev = self._last_paint_pos
                self.paint_stroke(self._stroke_tmp, prev, pos, brush, 255)
                if p["edit_mode"] == 1:
                    self.paint_stroke(self.draw_mask, prev, pos, brush, 255)
                    self.paint_stroke(self.erase_mask, prev, pos, brush, 0)
                else:
                    self.paint_stroke(self.erase_mask, prev, pos, brush, 255)
                    self.paint_stroke(self.draw_mask, prev, pos, brush, 0)
                self._last_paint_pos = pos

        if (event == cv2.EVENT_RBUTTONDOWN) or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_RBUTTON)):
            pos = self.mouse_to_image_xy(mx, my)
            if pos is not None:
                prev = self._last_paint_pos
                self.paint_stroke(self._stroke_tmp, prev, pos, brush, 255)
                if p["edit_mode"] == 1:
                    self.paint_stroke(self.draw_mask, prev, pos, brush, 0)
                else:
                    self.paint_stroke(self.erase_mask, prev, pos, brush, 0)
                self._last_paint_pos = pos

        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if self._stroke_tmp is not None and self._erase_before is not None and self._draw_before is not None:
                changed = self._stroke_tmp > 0
                if np.any(changed):
                    prev_erase = self._erase_before[changed].copy()
                    prev_draw = self._draw_before[changed].copy()
                    self.undo_stack.append(("edit", changed, prev_erase, prev_draw))
                    if len(self.undo_stack) > 200: self.undo_stack.pop(0)
            self._stroke_tmp = None; self._erase_before = None
            self._draw_before = None
            self._last_paint_pos = None

    def undo(self):
        if not self.undo_stack: return
        item = self.undo_stack.pop()
        if item[0] == "edit":
            _, changed, prev_erase, prev_draw = item
            self.erase_mask[changed] = prev_erase
            self.draw_mask[changed] = prev_draw

    # ---------- 루프 ----------
    def run(self):
        print("[*] Controls:")
        print("  Mouse: Erase mode → L-drag=지움(erase), R-drag=복원(restore)")
        print("         Draw mode  → L-drag=흰색 크랙 추가, R-drag=직접 그린 부분 제거")
        print("         Zoom/Pan   → Wheel=확대/축소, R-drag=화면 이동")
        print("  Keys : S=Save&Next  N=Next(unlabeled)  P=Prev(unlabeled)  Z=Undo  C=Clear manual edits")
        print("         M=Toggle Model  D=Toggle Draw/Erase  V=Toggle Zoom/Pan  H=Help  Q/Esc=Quit")
        while True:
            self.draw()
            k = cv2.waitKey(16) & 0xFF
            if k in (ord('q'), 27): break
            elif k == ord('s'):
                if self.viewing_existing_mask:
                    if not self.save_final():
                        continue
                    self.viewing_existing_mask = False
                    if not self.advance_to_next_unlabeled(+1): break
                elif self.mask_exists_for(self.idx):
                    print("[*] 이미 저장된 마스크가 있어 저장을 생략합니다.")
                    if not self.advance_to_next_unlabeled(+1): break
                else:
                    if not self.save_final():
                        continue
                    if not self.advance_to_next_unlabeled(+1): break
            elif k == ord('n'):
                if SAVE_EMPTY_ON_SKIP and not self.mask_exists_for(self.idx):
                    if not self.save_empty():
                        continue
                if not self.advance_to_next_unlabeled(+1): break
            elif k == ord('p'):
                if SAVE_EMPTY_ON_SKIP and not self.mask_exists_for(self.idx):
                    if not self.save_empty():
                        continue
                if not self.advance_to_next_unlabeled(-1): break
            elif k == ord('z'):
                self.undo()
            elif k == ord('c'):
                self.erase_mask[:] = 0
                self.draw_mask[:] = 0
                self.undo_stack.clear()
                print("[*] Manual edit masks cleared.")
            elif k == ord('m'):
                cur = cv2.getTrackbarPos(TB_USE_MODEL, WINDOW_CTRL)
                cv2.setTrackbarPos(TB_USE_MODEL, WINDOW_CTRL, 0 if cur==1 else 1)
            elif k == ord('d'):
                cur = cv2.getTrackbarPos(TB_EDIT_MODE, WINDOW_CTRL)
                cv2.setTrackbarPos(TB_EDIT_MODE, WINDOW_CTRL, 0 if cur==1 else 1)
            elif k == ord('v'):
                cur = cv2.getTrackbarPos(TB_ZOOM_MODE, WINDOW_CTRL)
                cv2.setTrackbarPos(TB_ZOOM_MODE, WINDOW_CTRL, 0 if cur==1 else 1)
            elif k == ord('h'):
                self.show_help()

        cv2.destroyAllWindows()

# ============================
# 실행
# ============================
if __name__ == "__main__":
    files = list_images(SRC_DIR)
    if not files: raise SystemExit("입력 폴더에 이미지가 없습니다.")
    if REVIEW_EXISTING_MASKS:
        files = [p for p in files if os.path.exists(
            os.path.join(DST_DIR, os.path.splitext(os.path.basename(p))[0] + ".png")
        )]
        if not files: raise SystemExit("원본 이미지와 파일명이 대응되는 PNG 마스크가 없습니다.")
        print(f"[*] 기존 마스크 검수 모드: 대응되는 이미지/마스크 {len(files)}쌍")
    app = App(files, DST_DIR)
    app.run()
