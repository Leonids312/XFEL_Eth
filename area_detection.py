"""Full-resolution area measurement and cross-section helpers (no Tk state)."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


METHODS = {"threshold": "灰度阈值（原有）", "layout": "按排布：明暗坑 / DIC",
           "pit": "明暗坑：边缘拼接", "dic": "掠入射 DIC：RGB 双边"}


def read_native_rgb(path: Path) -> np.ndarray:
    """Keep native TIFF sample depth; display conversions never enter a measurement."""
    # Match the preview's EXIF orientation (UNCHANGED skips this for JPEG/PNG)
    # while retaining values above 255 in high-bit-depth images.
    pixels = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if pixels is None:
        with Image.open(path) as src:
            if src.mode not in {"RGB", "L", "I", "F", "I;16", "I;16L", "I;16B"}:
                raise ValueError(f"无法以原始通道和位深读取 {path.name}（{src.mode}）。")
            pixels = np.array(src)
            if pixels.ndim == 3:
                return pixels  # PIL uses RGB
    if pixels.ndim == 2:
        return np.broadcast_to(pixels[..., None], (*pixels.shape, 3))
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError(f"不支持的图像通道：{path.name} {pixels.shape}")
    return np.ascontiguousarray(pixels[..., 2::-1])


def spot_bounds(height, width, rows, cols, row, col):
    """Half-open integer grid boundaries, in the original pixel coordinate system."""
    return ((col - 1) * width // cols, (row - 1) * height // rows,
            col * width // cols, row * height // rows)


def resolve_method(method, layout):
    if method == "layout":
        return "pit" if layout == "normal" else "dic"
    return method


def measure_image(path, layout, cells, method, gap, sensitivity, cancelled=lambda: False):
    from dic_detection import detect_dic
    from pit_detection import detect_pit

    if not cells:
        return {}
    rgb = read_native_rgb(path)
    height, width = rgb.shape[:2]
    axis = "horizontal" if layout == "grazing" else "vertical"
    method = resolve_method(method, layout)
    measurements = {}
    for uid, rows, cols, row, col in cells:
        if cancelled():
            return {}
        x0, y0, x1, y1 = spot_bounds(height, width, rows, cols, row, col)
        roi = rgb[y0:y1, x0:x1]
        if roi.size == 0:
            result = {"status": "待复核", "diagnostics": {"reason": "分区尺寸为空，请检查行列数。"}}
        elif method == "dic":
            result = detect_dic(roi, axis=axis, gap_px=gap, sensitivity=sensitivity)
        elif method == "pit":
            result = detect_pit(roi, gap_px=gap, sensitivity=sensitivity)
        else:
            raise ValueError(f"未知轮廓识别方法：{method}")
        status = result["status"]
        diagnostics = {str(k): (v.item() if isinstance(v, np.generic) else v)
                       for k, v in (result.get("diagnostics") or {}).items()
                       if isinstance(v, (str, int, float, bool, np.number))
                       and not (isinstance(v, (float, np.floating)) and not np.isfinite(v))}
        item = {"status": status, "method": method, "gap_px": gap, "sensitivity": sensitivity,
                "diagnostics": diagnostics, "contour": None, "area_px": None, "area_um2": None}
        offset = np.array([x0, y0], dtype=np.int32)
        if status == "自动":
            pts = np.asarray(result["contour"], dtype=np.float32).reshape(-1, 2)
            if not np.isfinite(pts).all() or len(pts) < 3 or np.any(pts < 0) or np.any(pts > [x1-x0-1, y1-y0-1]):
                raise ValueError("识别器返回了越界或无效的闭合轮廓。")
            # Keep the same vertices for area, display, exports and future reloads.
            pts = np.rint(pts).astype(np.int32) + offset
            area = abs(cv2.contourArea(pts))
            if area > 0:
                item.update(contour=pts.tolist(), area_px=area)
            else:
                item.update(status="待复核", diagnostics={"reason": "边界未围成有效面积。"})
        for key in ("paths", "bridges"):
            if result.get(key):
                item[key] = [(np.asarray(p, dtype=np.int32).reshape(-1, 2) + offset).tolist()
                             for p in result[key] if len(p) >= 2]
        if result.get("review_contour") is not None and status == "待复核":
            draft = np.asarray(result["review_contour"], dtype=np.int32).reshape(-1, 2)
            if len(draft) >= 3 and np.all(draft >= 0) and np.all(draft <= [x1-x0-1, y1-y0-1]):
                item["review_contour"] = (draft + offset).tolist()
        measurements[uid] = item
    return measurements


def cross_section(roi, axis, fraction):
    """Average at most five neighbouring scanlines, never blur across the stripe."""
    oriented = roi if axis == "vertical" else roi.swapaxes(0, 1)
    at = round(np.clip(fraction, 0, 1) * (len(oriented) - 1))
    return int(at), oriented[max(0, at - 2):at + 3].mean(axis=0)


def crossings(contour, axis, at, closed=True):
    points = np.asarray(contour, dtype=float).reshape(-1, 2)
    if axis == "horizontal":
        points = points[:, ::-1]
    ends = np.roll(points, -1, axis=0) if closed else points[1:]
    points = points if closed else points[:-1]
    use = (points[:, 1] > at) != (ends[:, 1] > at)
    start, end = points[use], ends[use]
    return start[:, 0] + (at - start[:, 1]) * (end[:, 0] - start[:, 0]) / (end[:, 1] - start[:, 1])
