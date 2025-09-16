# tools/viz.py
from __future__ import annotations
from typing import Tuple, Union, Optional
import cv2

# ---- 共用預設字體與大小（可依需求調整） ----
FONT = cv2.FONT_HERSHEY_SIMPLEX

ID_SCALE = 0.8
ID_THICK = 2

GID_SCALE = 0.9
GID_THICK = 2

STATUS_SCALE = 1.0
STATUS_THICK = 3


def _put_text(
    frame,
    text: str,
    org: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int,
    bg: Optional[Tuple[int, int, int]] = None,
    bg_pad: int = 4,
):
    """
    在框架上畫字（可選文字背景塊）。
    參數：
        frame   : 影像 (H, W, 3)
        text    : 要顯示的字串
        org     : 文字左下角座標 (x, y)
        scale   : 字體縮放
        color   : 文字顏色 (B, G, R)
        thickness: 文字線寬
        bg      : 若提供 (B,G,R) 則在文字底下畫實心背景矩形（增加可讀性）
        bg_pad  : 背景塊的內距（像素）
    """
    x, y = int(org[0]), int(org[1])

    if bg is not None:
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        x1, y1 = x - bg_pad, y - th - bg_pad
        x2, y2 = x + tw + bg_pad, y + baseline + bg_pad
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg, thickness=-1)

    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


def draw_box(
    frame,
    l: int, t: int, r: int, b: int,
    color: Tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2
):
    """
    畫一個偵測框（Bounding Box）。
    參數：
        l, t, r, b : 左、上、右、下 邊界（像素）
        color      : 框線顏色 (B,G,R)
        thickness  : 框線粗細（像素）
    """
    cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, thickness)


def draw_id(
    frame,
    l: int, t: int,
    track_id: Union[int, str],
    color: Tuple[int, int, int] = (0, 200, 0),
    scale: float = ID_SCALE,
    thickness: int = ID_THICK,
    bg: Optional[Tuple[int, int, int]] = None,
):
    """
    在框上方顯示「Local Track ID」。
    參數：
        l, t     : 文字定位以框的左上角為基準，會往上偏 8px
        track_id : 軌跡 ID（int 或 str）
        color    : 文字顏色
        scale    : 文字大小
        thickness: 文字線寬
        bg       : 可選，文字底色 (B,G,R)
    """
    _put_text(
        frame,
        f"ID:{track_id}",
        (int(l), max(0, int(t) - 8)),
        scale=scale, color=color, thickness=thickness, bg=bg
    )


def draw_gid(
    frame,
    l: int, t: int,
    gid: Union[int, str],
    color: Tuple[int, int, int] = (0, 0, 255),
    scale: float = GID_SCALE,
    thickness: int = GID_THICK,
    bg: Optional[Tuple[int, int, int]] = None,
):
    """
    在框上方顯示「Global ID」；若未決可傳入 '?'。
    參數：
        l, t     : 文字定位以框的左上角為基準，會往上偏 28px
        gid      : 全域 ID（int 或 '?'）
        color    : 文字顏色
        scale    : 文字大小
        thickness: 文字線寬
        bg       : 可選，文字底色 (B,G,R)
    """
    _put_text(
        frame,
        f"GID:{gid}",
        (int(l), max(0, int(t) - 28)),
        scale=scale, color=color, thickness=thickness, bg=bg
    )


def draw_status_line(
    frame,
    l: int, b: int,
    text: str,
    color: Tuple[int, int, int] = (255, 255, 255),
    scale: float = STATUS_SCALE,
    thickness: int = STATUS_THICK,
    y_offset: int = 20,
    bg: Optional[Tuple[int, int, int]] = None,
):
    """
    在框的左下角附近顯示狀態列（例如「Inside | OBS 7/30」）。
    參數：
        l, b     : 框的左下角座標
        text     : 要顯示的狀態文字
        color    : 文字顏色
        scale    : 文字大小
        thickness: 文字線寬
        y_offset : 往下位移多少像素（預設 20）
        bg       : 可選，文字底色 (B,G,R)
    """
    _put_text(
        frame,
        text,
        (int(l), int(b) + int(y_offset)),
        scale=scale, color=color, thickness=thickness, bg=bg
    )


def draw_foot(
    frame,
    x: Union[int, float], y: Union[int, float],
    color: Tuple[int, int, int] = (0, 0, 255),
    radius: int = 5,
    thickness: int = -1
):
    """
    在目標「腳點」（bbox 底邊中點）畫實心圓點。
    參數：
        x, y     : 腳點座標（像素）
        color    : 顏色 (B,G,R)
        radius   : 半徑（像素）
        thickness: 負值代表實心
    """
    cv2.circle(frame, (int(x), int(y)), int(radius), color, int(thickness))


def draw_fps(frame, fps: float,
    x: int = 10, y: int = 120,
    color: Tuple[int, int, int] = (0, 255, 0),
    scale: float = 0.9,
    thickness: int = 2,
    bg: Optional[Tuple[int, int, int]] = None,
):
    """
    在畫面固定位置顯示處理 FPS。
    參數：
        fps      : 當前處理 FPS（浮點數）
        x, y     : 文字左下角座標
        color    : 顏色 (B,G,R)
        scale    : 文字大小
        thickness: 線寬
        bg       : 可選，文字底色 (B,G,R)
    """
    _put_text(
        frame,
        f"Proc:{fps:.1f}",
        (int(x), int(y)),
        scale=scale, color=color, thickness=thickness, bg=bg
    )

def status_text_and_color(track_state: dict, obs_n: int):
    """
    回傳 (text, bgr_color)
    - 未 commit：顯示 OBS m/N；若判定歧義 ambiguous=True，顯示 OBS* m/N（黃色）
    - 已 commit：顯示 OK（綠色）
    """
    if track_state.get("committed"):
        return "", (0, 200, 0)
    m = len(track_state.get("obs_feats") or [])
    if track_state.get("ambiguous"):
        return f"OBS* {m}/{obs_n}", (0, 255, 255)   # 黃色提示歧義
    return f"OBS {m}/{obs_n}", (255, 255, 255)      # 白色一般觀察