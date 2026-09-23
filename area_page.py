"""Step 1: Leica image damage-area analysis page."""

from __future__ import annotations

import math
import json
import os
import re
import tkinter as tk
import tkinter.font as tkfont
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from xfel_core import ProjectState, RecentPathStore, attenuation_sort_key


IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(text: str) -> list[tuple[int, float | str]]:
    return [(0, float(part)) if re.fullmatch(r"\d+(?:\.\d+)?", part) else (1, part.casefold())
            for part in re.split(r"(\d+(?:\.\d+)?)", text)]


def expand_range(text: str) -> list[int]:
    values: list[int] = []
    for part in re.split(r"[,，、;；]", text):
        part = part.strip()
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if match:
            start, end = map(int, match.groups())
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        elif part.isdigit():
            values.append(int(part))
    return values


def split_attens(text: str) -> list[str]:
    normalized = text.replace("，", ",").replace("、", ",").replace(";", ",").replace("；", ",")
    # A decimal such as 1.5 is one attenuation; dotted lists such as 0.3.6 are multiple.
    if "," not in normalized and normalized.count(".") == 1:
        left, right = normalized.split(".", 1)
        if left.isdigit() and right.isdigit() and len(right) == 1 and int(left) <= 6:
            return [normalized]
        normalized = normalized.replace(".", ",")
    elif "," not in normalized and normalized.count(".") > 1:
        normalized = normalized.replace(".", ",")
    return [part.strip() for part in normalized.split(",") if part.strip()]


def mapped_serial(row: int, col: int, total_cols: int, column_order: str, row_direction: str) -> int:
    """Reproduce the original Leica program's global-R/row-direction numbering."""
    column_number = col if row_direction == column_order else total_cols - col + 1
    return (row - 1) * total_cols + column_number


def mapped_grazing_serial(serial: int, range_min: int, range_max: int, direction: str) -> int:
    """Map a physical top-to-bottom strip index to its grazing-incidence serial."""
    return serial if direction == "TTB" else range_min + range_max - serial


@dataclass
class Spot:
    atten: str
    serial: int
    row: int
    col: int
    display_row: int
    display_col: int
    uid: str = ""


@dataclass
class ImageEntry:
    path: Path
    rel_path: str
    spots: list[Spot]
    display_rows: int
    display_cols: int
    layout: str = "normal"

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class Candidate:
    contour: list[list[int]]
    area_px: float
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]


@dataclass
class ProcessingParameters:
    gray_low: int = 0
    gray_high: int = 115
    bright_damage: bool = False
    min_area_px: float = 80.0
    max_area_px: float = 100000.0
    median_blur: int = 3
    close_kernel: int = 5


def parse_image_name(
    path: Path,
    root: Path,
    infer_missing_l: bool = True,
    segmentation_mode: str = "normal",
) -> ImageEntry | None:
    stem = path.stem
    explicit = re.search(
        r"atten\s*(?P<atten>[\d.,，、;；]+).*?L\s*(?P<rows>[\d,，、;；-]+).*?R\s*(?P<cols>[\d,，、;；-]+)",
        stem, flags=re.I,
    )
    no_pattern = re.search(
        r"atten\s*(?P<atten>[+-]?\d+(?:\.\d+)?).*?no[_\s-]*(?P<start>\d+)\s*-\s*(?P<end>\d+)",
        stem, flags=re.I,
    )
    spots: list[Spot] = []
    layout = "normal"
    if explicit:
        attens = split_attens(explicit.group("atten"))
        rows = expand_range(explicit.group("rows"))
        cols = expand_range(explicit.group("cols"))
        if not attens or not rows or not cols:
            return None
        display_row = 0
        for atten in attens:
            for row in rows:
                display_row += 1
                for display_col, col in enumerate(cols, 1):
                    spots.append(Spot(atten, 0, row, col, display_row, display_col))
        display_rows, display_cols = len(attens) * len(rows), len(cols)
    elif no_pattern and (infer_missing_l or segmentation_mode == "grazing"):
        atten = no_pattern.group("atten")
        start, end = int(no_pattern.group("start")), int(no_pattern.group("end"))
        serials = list(range(start, end + (1 if end >= start else -1), 1 if end >= start else -1))
        if segmentation_mode == "grazing":
            layout = "grazing"
            for display_row, serial in enumerate(serials, 1):
                spots.append(Spot(atten, 0, 1, serial, display_row, 1))
            display_rows, display_cols = len(serials), 1
        else:
            for display_col, serial in enumerate(serials, 1):
                spots.append(Spot(atten, 0, 1, serial, 1, display_col))
            display_rows, display_cols = 1, len(serials)
    else:
        return None
    rel_path = os.path.relpath(path, root)
    for index, spot in enumerate(spots, 1):
        spot.uid = f"{rel_path}#{index}:{spot.atten}:{spot.row}:{spot.col}"
    return ImageEntry(path, rel_path, spots, display_rows, display_cols, layout)


def read_image_rgb(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detect_candidates(image: np.ndarray, params: ProcessingParameters) -> tuple[list[Candidate], np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = max(1, int(params.median_blur))
    if blur > 1:
        blur += 1 - blur % 2
        gray = cv2.medianBlur(gray, blur)
    if params.bright_damage:
        gray = cv2.bitwise_not(gray)
    low = max(0, min(255, int(params.gray_low)))
    high = max(low, min(255, int(params.gray_high)))
    mask = cv2.inRange(gray, low, high)
    kernel_size = max(1, int(params.close_kernel))
    if kernel_size > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    candidates: list[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not (params.min_area_px <= area <= params.max_area_px):
            continue
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue
        points = [[int(point[0][0]), int(point[0][1])] for point in contour]
        candidates.append(Candidate(points, area, (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]), (x, y, w, h)))
    candidates.sort(key=lambda item: item.area_px, reverse=True)
    return candidates, mask


class AreaAnalysisPage(ttk.Frame):
    def __init__(self, master: tk.Misc, state: ProjectState, status_callback=None, path_store: RecentPathStore | None = None) -> None:
        super().__init__(master)
        self.state = state
        self.status_callback = status_callback or (lambda text: None)
        self.path_store = path_store
        self.root_dir: Path | None = None
        self.entries: list[ImageEntry] = []
        self.current_entry: ImageEntry | None = None
        self.results: dict[str, dict[str, Any]] = {}
        self.directions: dict[str, str] = {}
        self.candidate_cache: dict[str, list[Candidate]] = {}
        self.mask_cache: dict[str, np.ndarray] = {}
        self.image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.params = ProcessingParameters()
        self.scale_um_per_px: float | None = None
        self.scale_line: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.draw_points: list[tuple[int, int]] = []
        self.selected_spot_uid = ""
        self.view_zoom = 1.0
        self.view_pan = [0.0, 0.0]
        self.display_scale = 1.0
        self.display_offset = (0.0, 0.0)
        self.background_key: tuple[Any, ...] | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.drag_start: tuple[int, int] | None = None
        self.pan_start = (0.0, 0.0)
        self.pan_last = (0, 0)
        self.scale_drag_start: tuple[int, int] | None = None
        self._interactive_render_id: str | None = None
        self._quality_render_id: str | None = None

        self.folder_var = tk.StringVar()
        self.image_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="browse")
        self.segmentation_mode_var = tk.StringVar(value="normal")
        self.column_order_var = tk.StringVar(value="LTR")
        self.show_mask_var = tk.BooleanVar(value=False)
        self.show_contours_var = tk.BooleanVar(value=True)
        self.infer_missing_l_var = tk.BooleanVar(value=True)
        self.gray_low_var = tk.IntVar(value=self.params.gray_low)
        self.gray_high_var = tk.IntVar(value=self.params.gray_high)
        self.bright_var = tk.BooleanVar(value=self.params.bright_damage)
        self.min_area_var = tk.DoubleVar(value=self.params.min_area_px)
        self.max_area_var = tk.DoubleVar(value=self.params.max_area_px)
        self.blur_var = tk.IntVar(value=self.params.median_blur)
        self.close_var = tk.IntVar(value=self.params.close_kernel)
        self.scale_length_var = tk.DoubleVar(value=80.0)
        self.contour_color_var = tk.StringVar(value="#22c55e")
        self.contour_width_var = tk.IntVar(value=2)
        self.manual_color_var = tk.StringVar(value="#38bdf8")
        self.manual_width_var = tk.IntVar(value=2)
        self.selected_color_var = tk.StringVar(value="#f59e0b")
        self.selected_width_var = tk.IntVar(value=4)
        self.label_color_var = tk.StringVar(value="#22c55e")
        self.label_font_size_var = tk.IntVar(value=11)
        self.scale_line_color_var = tk.StringVar(value="#ffffff")
        self.scale_line_width_var = tk.IntVar(value=3)
        self.grid_color_var = tk.StringVar(value="#64748b")
        self.grid_width_var = tk.IntVar(value=1)
        self.annotation_font: tkfont.Font | None = None
        self.status_var = tk.StringVar(value="请选择徕卡图像文件夹。")
        self._build_ui()
        self.apply_display_style()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="选择图像文件夹", command=self.choose_folder).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="导入旧面积存档", command=self.import_legacy_project).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Entry(toolbar, textvariable=self.folder_var, width=52, state="readonly").pack(side=tk.LEFT, padx=6)
        ttk.Label(toolbar, text="分割模式").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Radiobutton(
            toolbar, text="正入射", variable=self.segmentation_mode_var, value="normal",
            command=self.on_segmentation_mode_changed,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            toolbar, text="掠入射", variable=self.segmentation_mode_var, value="grazing",
            command=self.on_segmentation_mode_changed,
        ).pack(side=tk.LEFT, padx=(3, 0))
        ttk.Checkbutton(toolbar, text="兼容缺省 L=1 文件名", variable=self.infer_missing_l_var, command=self.reload_folder).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="处理当前", command=self.process_current).pack(side=tk.LEFT, padx=(8, 3))
        ttk.Button(toolbar, text="处理全部", command=self.process_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="面积另存为 Excel", command=self.export_excel).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="复位视图", command=self.reset_view).pack(side=tk.LEFT, padx=3)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(body, padding=6)
        center = ttk.Frame(body, padding=4)
        right = ttk.Frame(body, padding=6)
        body.add(left, weight=0)
        body.add(center, weight=1)
        body.add(right, weight=0)

        ttk.Label(left, text="图像文件", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W)
        self.file_list = tk.Listbox(left, width=31, exportselection=False)
        self.file_list.pack(fill=tk.BOTH, expand=True, pady=(5, 8))
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)
        ttk.Label(left, text="滚轮缩放；右键拖动平移\n手动画轮廓时逐点单击，双击结束", foreground="#475569").pack(anchor=tk.W)

        self.column_order_box = ttk.LabelFrame(left, text="全局 R 列号增长方向（正入射）", padding=6)
        self.column_order_box.pack(fill=tk.X, pady=(8, 5))
        self.column_order_buttons = (
            ttk.Radiobutton(self.column_order_box, text="左 → 右", variable=self.column_order_var, value="LTR", command=self.on_column_order_changed),
            ttk.Radiobutton(self.column_order_box, text="右 → 左", variable=self.column_order_var, value="RTL", command=self.on_column_order_changed),
        )
        self.column_order_buttons[0].pack(side=tk.LEFT)
        self.column_order_buttons[1].pack(side=tk.LEFT, padx=(12, 0))

        self.direction_box = ttk.LabelFrame(left, text="每个衰减倍数、每一行的打点方向", padding=6)
        self.direction_box.pack(fill=tk.BOTH, pady=(0, 3))
        self.direction_tree = ttk.Treeview(self.direction_box, columns=("direction",), show="tree headings", height=8, selectmode="extended")
        self.direction_tree.heading("#0", text="行")
        self.direction_tree.heading("direction", text="方向")
        self.direction_tree.column("#0", width=175, anchor=tk.W)
        self.direction_tree.column("direction", width=68, anchor=tk.CENTER)
        self.direction_tree.pack(fill=tk.BOTH, expand=True)
        self.direction_tree.bind("<Double-1>", lambda _event: self.toggle_direction())
        direction_buttons = ttk.Frame(self.direction_box)
        direction_buttons.pack(fill=tk.X, pady=(5, 0))
        self.direction_first_button = ttk.Button(
            direction_buttons, text="设为左→右", command=lambda: self.set_selected_direction(self._direction_values()[0]),
        )
        self.direction_first_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.direction_second_button = ttk.Button(
            direction_buttons, text="设为右→左", command=lambda: self.set_selected_direction(self._direction_values()[1]),
        )
        self.direction_second_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self.canvas = tk.Canvas(center, bg="#161a20", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._schedule_interactive_render())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<ButtonPress-3>", self._pan_press)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._pan_release)
        self.canvas.bind("<ButtonPress-1>", self._left_press)
        self.canvas.bind("<B1-Motion>", self._left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._left_release)
        self.canvas.bind("<Double-Button-1>", self.finish_polygon)

        notebook = ttk.Notebook(right, width=335)
        notebook.pack(fill=tk.BOTH, expand=True)
        process_tab = ttk.Frame(notebook, padding=8)
        spots_tab = ttk.Frame(notebook, padding=8)
        scale_tab = ttk.Frame(notebook, padding=8)
        style_tab = ttk.Frame(notebook, padding=8)
        notebook.add(process_tab, text="识别参数")
        notebook.add(spots_tab, text="坑位与手动修正")
        notebook.add(scale_tab, text="比例尺")
        notebook.add(style_tab, text="显示样式")
        self._build_process_tab(process_tab)
        self._build_spots_tab(spots_tab)
        self._build_scale_tab(scale_tab)
        self._build_style_tab(style_tab)
        self._update_direction_controls()
        ttk.Label(self, textvariable=self.status_var, anchor=tk.W, padding=(8, 4)).pack(fill=tk.X)

    def _spin(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable, low: float, high: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Spinbox(parent, textvariable=variable, from_=low, to=high, width=12).grid(row=row, column=1, sticky=tk.E, pady=3)

    def _build_process_tab(self, tab: ttk.Frame) -> None:
        self._spin(tab, 0, "灰度下限", self.gray_low_var, 0, 255)
        self._spin(tab, 1, "灰度上限", self.gray_high_var, 0, 255)
        ttk.Checkbutton(tab, text="亮损伤模式（灰度反转）", variable=self.bright_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)
        self._spin(tab, 3, "最小面积 (px²)", self.min_area_var, 0, 1e9)
        self._spin(tab, 4, "最大面积 (px²)", self.max_area_var, 0, 1e9)
        self._spin(tab, 5, "中值滤波核", self.blur_var, 1, 31)
        self._spin(tab, 6, "闭运算核", self.close_var, 1, 51)
        ttk.Checkbutton(tab, text="显示二值图", variable=self.show_mask_var, command=lambda: self.render_canvas(True)).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(8, 2))
        ttk.Checkbutton(tab, text="显示轮廓", variable=self.show_contours_var, command=self.redraw_overlay).grid(row=8, column=0, columnspan=2, sticky=tk.W)
        ttk.Button(tab, text="应用并处理当前", command=self.process_current).grid(row=9, column=0, columnspan=2, sticky=tk.EW, pady=(10, 3))
        ttk.Button(tab, text="应用并处理全部", command=self.process_all).grid(row=10, column=0, columnspan=2, sticky=tk.EW)
        ttk.Label(tab, text="自动识别默认选择每个网格中面积最大的候选轮廓。", wraplength=300, foreground="#475569").grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(12, 0))

    def _build_spots_tab(self, tab: ttk.Frame) -> None:
        self.spot_tree = ttk.Treeview(tab, columns=("atten", "serial", "area", "status"), show="headings", height=20)
        for column, title, width in (("atten", "衰减", 58), ("serial", "序号", 55), ("area", "面积", 92), ("status", "状态", 78)):
            self.spot_tree.heading(column, text=title)
            self.spot_tree.column(column, width=width, anchor=tk.CENTER, stretch=False)
        self.spot_tree.pack(fill=tk.BOTH, expand=True)
        self.spot_tree.bind("<<TreeviewSelect>>", self._spot_select)
        ttk.Button(tab, text="设为无损伤（面积 0）", command=self.mark_no_damage).pack(fill=tk.X, pady=(8, 3))
        ttk.Button(tab, text="图像截断（不统计）", command=self.mark_truncated).pack(fill=tk.X, pady=3)
        ttk.Radiobutton(tab, text="点击自动候选轮廓", variable=self.mode_var, value="pick").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(tab, text="手动画轮廓", variable=self.mode_var, value="draw").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(tab, text="浏览", variable=self.mode_var, value="browse").pack(anchor=tk.W, pady=2)
        ttk.Button(tab, text="清除未完成手动画线", command=self.clear_polygon).pack(fill=tk.X, pady=(6, 0))

    def _build_scale_tab(self, tab: ttk.Frame) -> None:
        ttk.Radiobutton(tab, text="拖动绘制比例尺线", variable=self.mode_var, value="scale").grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(tab, text="实际长度 (µm)").grid(row=1, column=0, sticky=tk.W, pady=(10, 4))
        ttk.Entry(tab, textvariable=self.scale_length_var, width=12).grid(row=1, column=1, sticky=tk.W, pady=(10, 4))
        ttk.Button(tab, text="应用比例尺", command=self.apply_scale).grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=4)
        ttk.Button(tab, text="清除比例尺", command=self.clear_scale).grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=4)
        self.scale_info = ttk.Label(tab, text="未设置比例尺", wraplength=300)
        self.scale_info.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        ttk.Label(tab, text="比例尺换算后，面积将以 µm² 自动同步到第 3 步；未换算时以 px² 同步。", wraplength=300, foreground="#475569").grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

    def _color_control(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
        ttk.Entry(parent, textvariable=variable, width=13).grid(row=row, column=1, sticky=tk.EW, padx=(6, 4), pady=4)
        ttk.Button(parent, text="选择", command=lambda: self.choose_color(variable)).grid(row=row, column=2, pady=4)

    def _build_style_tab(self, tab: ttk.Frame) -> None:
        self._color_control(tab, 0, "自动轮廓颜色", self.contour_color_var)
        self._spin(tab, 1, "自动轮廓线宽", self.contour_width_var, 1, 20)
        self._color_control(tab, 2, "手动轮廓颜色", self.manual_color_var)
        self._spin(tab, 3, "手动轮廓线宽", self.manual_width_var, 1, 20)
        self._color_control(tab, 4, "选中点高亮颜色", self.selected_color_var)
        self._spin(tab, 5, "选中点线宽", self.selected_width_var, 1, 24)
        self._color_control(tab, 6, "注释文字颜色", self.label_color_var)
        self._spin(tab, 7, "注释字号", self.label_font_size_var, 7, 32)
        self._color_control(tab, 8, "比例尺画线颜色", self.scale_line_color_var)
        self._spin(tab, 9, "比例尺画线宽度", self.scale_line_width_var, 1, 20)
        self._color_control(tab, 10, "网格颜色", self.grid_color_var)
        self._spin(tab, 11, "网格线宽", self.grid_width_var, 1, 10)
        ttk.Button(tab, text="应用显示样式", command=self.apply_display_style).grid(row=12, column=0, columnspan=3, sticky=tk.EW, pady=(10, 0))
        ttk.Label(tab, text="样式同时用于第 4 步的整幅徕卡点位详情图。", wraplength=300, foreground="#475569").grid(row=13, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        tab.columnconfigure(1, weight=1)

    def choose_color(self, variable: tk.StringVar) -> None:
        _rgb, color = colorchooser.askcolor(color=variable.get(), parent=self)
        if color:
            variable.set(color)
            self.apply_display_style()

    def _valid_color(self, variable: tk.StringVar, fallback: str) -> str:
        value = variable.get().strip()
        try:
            self.winfo_rgb(value)
        except tk.TclError:
            return fallback
        return value

    def apply_display_style(self) -> None:
        self.annotation_font = tkfont.Font(
            family="Microsoft YaHei UI",
            size=max(7, int(self.label_font_size_var.get())),
            weight="bold",
        )
        self._sync_display_style_to_state()
        self.redraw_overlay()
        self.state.notify("area_display_style")

    def _sync_display_style_to_state(self) -> None:
        self.state.area_display_style = {
            "auto_contour_color": self._valid_color(self.contour_color_var, "#22c55e"),
            "auto_contour_width": int(self.contour_width_var.get()),
            "manual_contour_color": self._valid_color(self.manual_color_var, "#38bdf8"),
            "manual_contour_width": int(self.manual_width_var.get()),
            "selected_color": self._valid_color(self.selected_color_var, "#f59e0b"),
            "selected_width": int(self.selected_width_var.get()),
            "label_color": self._valid_color(self.label_color_var, "#22c55e"),
            "label_font_size": int(self.label_font_size_var.get()),
            "scale_line_color": self._valid_color(self.scale_line_color_var, "#ffffff"),
            "scale_line_width": int(self.scale_line_width_var.get()),
            "grid_color": self._valid_color(self.grid_color_var, "#64748b"),
            "grid_width": int(self.grid_width_var.get()),
            "scale_line": self.scale_line,
        }

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.status_callback(text)
        self.update_idletasks()

    def choose_folder(self) -> None:
        initial_dir = self.path_store.initial_dir("area_image_folder_open", self.root_dir or Path.cwd()) if self.path_store else str(self.root_dir or Path.cwd())
        selected = filedialog.askdirectory(parent=self, initialdir=initial_dir, title="选择徕卡损伤图像文件夹")
        if selected:
            if self.path_store:
                self.path_store.remember_directory("area_image_folder_open", selected)
            self.load_folder(Path(selected))

    def import_legacy_project(self) -> None:
        initial_dir = self.path_store.initial_dir("area_legacy_project_open", self.root_dir or Path.cwd()) if self.path_store else str(self.root_dir or Path.cwd())
        filename = filedialog.askopenfilename(parent=self, initialdir=initial_dir, title="导入原面积程序 JSON 存档", filetypes=[("JSON", "*.json"), ("所有文件", "*.*")])
        if not filename:
            return
        if self.path_store:
            self.path_store.remember_file("area_legacy_project_open", filename)
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            folder = Path(data.get("folder") or data.get("root_dir") or data.get("image_folder") or "")
            if not folder.is_dir():
                initial_folder = self.path_store.initial_dir("area_legacy_image_folder_open", Path(filename).parent) if self.path_store else str(Path(filename).parent)
                selected = filedialog.askdirectory(parent=self, initialdir=initial_folder, title="选择该存档对应的图像文件夹")
                if not selected:
                    return
                if self.path_store:
                    self.path_store.remember_directory("area_legacy_image_folder_open", selected)
                folder = Path(selected)
            self.load_folder(folder, data)
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc), parent=self)
            return
        self.set_status(f"已导入旧面积存档：{filename}，数据已同步到第 3 步。")

    def reload_folder(self) -> None:
        if self.root_dir:
            saved = self.export_config()
            self.load_folder(self.root_dir, saved)

    def on_segmentation_mode_changed(self) -> None:
        """Reload geometry when switching between normal and grazing layouts."""
        self._update_direction_controls()
        if self.root_dir:
            saved = self.export_config()
            # The same UID can describe a different cell geometry after a
            # mode switch, so old contours must not silently move to another
            # physical strip.
            saved["results"] = {}
            self.load_folder(self.root_dir, saved)
            mode_label = "掠入射纵向分割" if self._is_grazing_mode() else "正入射网格分割"
            self.set_status(f"已切换为{mode_label}；旧轮廓结果已清空，请重新处理图像。")

    def _is_grazing_mode(self) -> bool:
        return self.segmentation_mode_var.get() == "grazing"

    def _direction_values(self) -> tuple[str, str]:
        return ("TTB", "BTT") if self._is_grazing_mode() else ("LTR", "RTL")

    def _update_direction_controls(self) -> None:
        grazing = self._is_grazing_mode()
        self.direction_box.configure(
            text="每个衰减倍数的序号方向" if grazing else "每个衰减倍数、每一行的打点方向",
        )
        self.direction_tree.heading("#0", text="衰减倍数" if grazing else "行")
        self.direction_first_button.configure(text="设为上→下" if grazing else "设为左→右")
        self.direction_second_button.configure(text="设为下→上" if grazing else "设为右→左")
        for button in self.column_order_buttons:
            button.configure(state=tk.DISABLED if grazing else tk.NORMAL)

    def load_folder(self, folder: Path, config: dict[str, Any] | None = None) -> None:
        if config:
            self.segmentation_mode_var.set(str(config.get("segmentation_mode", "normal")))
        self._update_direction_controls()
        self.root_dir = folder
        self.folder_var.set(str(folder))
        self.entries.clear()
        self.current_entry = None
        self.image_cache.clear()
        self.candidate_cache.clear()
        self.mask_cache.clear()
        if config is None:
            self.results = {}
            self.directions = {}
            self.column_order_var.set("LTR")
        files = sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: natural_key(p.name))
        skipped = 0
        for path in files:
            entry = parse_image_name(
                path, folder, self.infer_missing_l_var.get(), self.segmentation_mode_var.get(),
            )
            if entry:
                self.entries.append(entry)
            else:
                skipped += 1
        if config:
            self.directions = {str(key): str(value) for key, value in config.get("directions", {}).items()}
            self.column_order_var.set(config.get("column_order", "LTR"))
        self._recompute_serials()
        if config:
            self._apply_config_fields(config)
        self.file_list.delete(0, tk.END)
        for entry in self.entries:
            self.file_list.insert(tk.END, entry.name)
        self.refresh_directions()
        if self.entries:
            self.file_list.selection_set(0)
            self.select_entry(self.entries[0])
        else:
            self.render_canvas(True)
        self.publish_areas()
        mode_label = "掠入射纵向分割" if self._is_grazing_mode() else "正入射网格分割"
        self.set_status(
            f"以{mode_label}读取 {len(self.entries)} 张目标图像，"
            f"跳过 {skipped} 张不匹配文件名的图像。"
        )

    def _recompute_serials(self) -> None:
        max_col: dict[str, int] = {}
        for entry in self.entries:
            if entry.layout == "grazing":
                continue
            for spot in entry.spots:
                max_col[spot.atten] = max(max_col.get(spot.atten, 0), spot.col)
        for entry in self.entries:
            if entry.layout == "grazing":
                by_atten: dict[str, list[Spot]] = {}
                for spot in entry.spots:
                    by_atten.setdefault(spot.atten, []).append(spot)
                for atten, spots in by_atten.items():
                    range_min = min(spot.col for spot in spots)
                    range_max = max(spot.col for spot in spots)
                    direction = self.directions.get(self.grazing_direction_key(atten), "TTB")
                    for spot in spots:
                        spot.serial = mapped_grazing_serial(spot.col, range_min, range_max, direction)
                continue
            for spot in entry.spots:
                total_cols = max_col.get(spot.atten, spot.col)
                row_direction = self.directions.get(self.direction_key(spot.atten, spot.row), "LTR")
                spot.serial = mapped_serial(spot.row, spot.col, total_cols, self.column_order_var.get(), row_direction)

    @staticmethod
    def direction_key(atten: str, row: int) -> str:
        return f"{atten}|{row}"

    @staticmethod
    def grazing_direction_key(atten: str) -> str:
        return f"grazing|{atten}"

    def spot_display_col(self, entry: ImageEntry, spot: Spot) -> int:
        if entry.layout != "grazing" and self.column_order_var.get() == "RTL":
            return entry.display_cols - spot.display_col + 1
        return spot.display_col

    def refresh_directions(self, selected_keys: list[str] | None = None) -> None:
        selected_keys = selected_keys or []
        self.direction_tree.delete(*self.direction_tree.get_children())
        if self._is_grazing_mode():
            attens = sorted(
                {spot.atten for entry in self.entries for spot in entry.spots},
                key=attenuation_sort_key,
            )
            for atten in attens:
                key = self.grazing_direction_key(atten)
                direction = self.directions.get(key, "TTB")
                label = "上→下" if direction == "TTB" else "下→上"
                self.direction_tree.insert("", tk.END, iid=key, text=f"{atten}倍衰减", values=(label,))
                if key in selected_keys:
                    self.direction_tree.selection_add(key)
                    self.direction_tree.see(key)
            return
        pairs = sorted({(spot.atten, spot.row) for entry in self.entries for spot in entry.spots}, key=lambda item: (attenuation_sort_key(item[0]), item[1]))
        for atten, row in pairs:
            key = self.direction_key(atten, row)
            direction = self.directions.get(key, "LTR")
            label = "左→右" if direction == "LTR" else "右→左"
            self.direction_tree.insert("", tk.END, iid=key, text=f"{atten}倍衰减第{row}行", values=(label,))
            if key in selected_keys:
                self.direction_tree.selection_add(key)
                self.direction_tree.see(key)

    def on_column_order_changed(self) -> None:
        if self._is_grazing_mode():
            return
        self._recompute_serials()
        self.refresh_spot_tree()
        self.render_canvas(True)
        self.publish_areas()
        label = "左→右" if self.column_order_var.get() == "LTR" else "右→左"
        self.set_status(f"全局 R 列号增长方向已设为 {label}；序号和第 3 步面积对应已重算。建议重新处理已自动识别的图像。")

    def toggle_direction(self) -> None:
        selected = list(self.direction_tree.selection())
        if not selected:
            target = "衰减倍数" if self._is_grazing_mode() else "行"
            messagebox.showinfo("未选择", f"请先选择一个{target}；可按 Ctrl/Shift 多选，双击也可切换。", parent=self)
            return
        first, second = self._direction_values()
        for key in selected:
            self.directions[key] = second if self.directions.get(key, first) == first else first
        self._directions_changed(selected)

    def set_selected_direction(self, direction: str) -> None:
        selected = list(self.direction_tree.selection())
        if not selected:
            target = "衰减倍数" if self._is_grazing_mode() else "行"
            messagebox.showinfo("未选择", f"请先在列表中选择一个或多个{target}。", parent=self)
            return
        for key in selected:
            self.directions[key] = direction
        self._directions_changed(selected)

    def _directions_changed(self, selected: list[str]) -> None:
        self._recompute_serials()
        self.refresh_directions(selected)
        self.refresh_spot_tree()
        self.redraw_overlay()
        self.publish_areas()
        if self._is_grazing_mode():
            self.set_status("所选衰减倍数的上/下序号方向及第 3 步面积对应已重算。")
        else:
            self.set_status("所选行的打点方向、最终序号和第 3 步面积对应已重算。")

    def _on_file_select(self, _event=None) -> None:
        selected = self.file_list.curselection()
        if selected:
            self.select_entry(self.entries[selected[0]])

    def select_entry(self, entry: ImageEntry) -> None:
        self.current_entry = entry
        self.image_var.set(entry.name)
        self.view_zoom = 1.0
        self.view_pan = [0.0, 0.0]
        self.draw_points.clear()
        self.selected_spot_uid = entry.spots[0].uid if entry.spots else ""
        self.refresh_spot_tree()
        self.render_canvas(True)

    def load_image(self, entry: ImageEntry) -> np.ndarray:
        key = entry.rel_path
        if key not in self.image_cache:
            self.image_cache[key] = read_image_rgb(entry.path)
            while len(self.image_cache) > 2:
                self.image_cache.popitem(last=False)
        else:
            self.image_cache.move_to_end(key)
        return self.image_cache[key]

    def sync_params(self) -> None:
        self.params = ProcessingParameters(
            int(self.gray_low_var.get()), int(self.gray_high_var.get()), bool(self.bright_var.get()),
            float(self.min_area_var.get()), float(self.max_area_var.get()), int(self.blur_var.get()), int(self.close_var.get()),
        )

    def process_current(self) -> None:
        if not self.current_entry:
            return
        try:
            self.sync_params()
            self.process_entry(self.current_entry)
        except Exception as exc:
            messagebox.showerror("处理失败", str(exc), parent=self)
            return
        self.refresh_spot_tree()
        self.render_canvas(True)
        self.publish_areas()
        self.set_status(f"已处理：{self.current_entry.name}，面积已同步到第 3 步。")

    def process_all(self) -> None:
        if not self.entries:
            return
        try:
            self.sync_params()
            for index, entry in enumerate(self.entries, 1):
                self.set_status(f"正在处理 {index}/{len(self.entries)}：{entry.name}")
                self.process_entry(entry)
                if entry is not self.current_entry:
                    self.image_cache.pop(entry.rel_path, None)
        except Exception as exc:
            messagebox.showerror("批量处理失败", str(exc), parent=self)
            return
        self.refresh_spot_tree()
        self.render_canvas(True)
        self.publish_areas()
        self.set_status(f"已处理全部 {len(self.entries)} 张图像，面积已同步到第 3 步。")

    def process_entry(self, entry: ImageEntry) -> None:
        image = self.load_image(entry)
        candidates, mask = detect_candidates(image, self.params)
        self.candidate_cache[entry.rel_path] = candidates
        self.mask_cache[entry.rel_path] = mask
        height, width = image.shape[:2]
        used: set[int] = set()
        for spot in entry.spots:
            current = self.results.get(spot.uid, {})
            if current.get("status") in {"无损伤", "图像截断", "手动轮廓", "手动画"}:
                continue
            display_col = self.spot_display_col(entry, spot)
            x0 = (display_col - 1) * width / entry.display_cols
            x1 = display_col * width / entry.display_cols
            y0 = (spot.display_row - 1) * height / entry.display_rows
            y1 = spot.display_row * height / entry.display_rows
            available = [(index, candidate) for index, candidate in enumerate(candidates)
                         if index not in used and x0 <= candidate.centroid[0] < x1 and y0 <= candidate.centroid[1] < y1]
            if not available:
                self.results[spot.uid] = {"status": "未检出", "area_px": None, "area_um2": None, "contour": None}
                continue
            index, candidate = max(available, key=lambda pair: pair[1].area_px)
            used.add(index)
            self.results[spot.uid] = self._result(candidate.contour, candidate.area_px, "自动")

    def _result(self, contour: list[list[int]] | None, area_px: float, status: str) -> dict[str, Any]:
        area_um2 = area_px * self.scale_um_per_px**2 if self.scale_um_per_px is not None else None
        return {"status": status, "area_px": float(area_px), "area_um2": area_um2, "contour": contour}

    def refresh_spot_tree(self) -> None:
        self.spot_tree.delete(*self.spot_tree.get_children())
        if not self.current_entry:
            return
        for spot in sorted(self.current_entry.spots, key=lambda item: (attenuation_sort_key(item.atten), item.serial)):
            result = self.results.get(spot.uid, {})
            value = result.get("area_um2") if self.scale_um_per_px is not None else result.get("area_px")
            area = "" if value is None else f"{value:.6g}"
            self.spot_tree.insert("", tk.END, iid=spot.uid, values=(spot.atten, spot.serial, area, result.get("status", "未处理")))
        if self.selected_spot_uid and self.spot_tree.exists(self.selected_spot_uid):
            self.spot_tree.selection_set(self.selected_spot_uid)

    def _spot_select(self, _event=None) -> None:
        selection = self.spot_tree.selection()
        self.selected_spot_uid = selection[0] if selection else ""
        self.redraw_overlay()

    def selected_spot(self) -> Spot | None:
        if not self.current_entry:
            return None
        return next((spot for spot in self.current_entry.spots if spot.uid == self.selected_spot_uid), None)

    def mark_no_damage(self) -> None:
        spot = self.selected_spot()
        if not spot:
            messagebox.showinfo("未选择坑位", "请先选择坑位。", parent=self)
            return
        self.results[spot.uid] = self._result(None, 0.0, "无损伤")
        self._changed_area()

    def mark_truncated(self) -> None:
        """Exclude a grazing-incidence stripe that continues beyond the image edge."""
        spot = self.selected_spot()
        if not spot:
            messagebox.showinfo("未选择坑位", "请先选择坑位。", parent=self)
            return
        self.results[spot.uid] = {
            "status": "图像截断", "area_px": None, "area_um2": None, "contour": None,
        }
        self._changed_area()

    def clear_polygon(self) -> None:
        self.draw_points.clear()
        self.redraw_overlay()

    def _changed_area(self) -> None:
        self.refresh_spot_tree()
        self.redraw_overlay()
        self.publish_areas()

    def publish_areas(self) -> None:
        chosen: dict[tuple[str, int], dict[str, Any]] = {}
        priority = {"手动画": 4, "手动轮廓": 4, "无损伤": 4, "图像截断": 4,
                    "自动": 3, "未检出": 2, "未处理": 1}
        for entry in self.entries:
            for spot in entry.spots:
                result = self.results.get(spot.uid, {})
                status = result.get("status", "未处理")
                value = result.get("area_um2") if self.scale_um_per_px is not None else result.get("area_px")
                item = {
                    "atten": spot.atten, "serial": spot.serial, "area": value, "area_px": result.get("area_px"),
                    "status": status, "image_path": str(entry.path), "image_rel_path": entry.rel_path,
                    "image_display_rows": entry.display_rows, "image_display_cols": entry.display_cols,
                    "spot_display_row": spot.display_row, "spot_display_col": self.spot_display_col(entry, spot),
                    "contour": result.get("contour"),
                }
                key = (spot.atten, spot.serial)
                if key not in chosen or priority.get(status, 0) > priority.get(chosen[key]["status"], 0):
                    chosen[key] = item
        unit = "µm²" if self.scale_um_per_px is not None else "px²"
        self.state.merge_area_records(chosen.values(), unit)

    def _geometry(self) -> tuple[float, float, float, int, int]:
        if not self.current_entry:
            return 1.0, 0.0, 0.0, 1, 1
        image = self.load_image(self.current_entry)
        height, width = image.shape[:2]
        canvas_width, canvas_height = max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())
        scale = min(canvas_width / width, canvas_height / height) * self.view_zoom
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        offset_x = (canvas_width - new_width) / 2 + self.view_pan[0]
        offset_y = (canvas_height - new_height) / 2 + self.view_pan[1]
        return scale, offset_x, offset_y, new_width, new_height

    def render_canvas(self, background: bool = True, fast: bool = False) -> None:
        if not fast:
            # A direct high-quality render supersedes any older queued preview
            # so a stale fast frame cannot replace it a moment later.
            if self._interactive_render_id is not None:
                self.after_cancel(self._interactive_render_id)
                self._interactive_render_id = None
            if self._quality_render_id is not None:
                self.after_cancel(self._quality_render_id)
                self._quality_render_id = None
        if not self.current_entry:
            self.canvas.delete("all")
            self.canvas.create_text(20, 20, anchor=tk.NW, fill="#cbd5e1", text="选择图像文件夹后显示徕卡图像")
            return
        scale, ox, oy, new_width, new_height = self._geometry()
        self.display_scale, self.display_offset = scale, (ox, oy)
        mask_key = self.show_mask_var.get() and self.current_entry.rel_path in self.mask_cache
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        key = (
            self.current_entry.rel_path, canvas_width, canvas_height,
            round(self.view_zoom, 5), round(self.view_pan[0], 2), round(self.view_pan[1], 2),
            mask_key, "fast" if fast else "quality",
        )
        if background or key != self.background_key:
            self.canvas.delete("all")
            image = self.load_image(self.current_entry)
            height, width = image.shape[:2]

            # Only rasterize the source rectangle that is visible on screen.
            # The previous implementation resized the entire image on every
            # mouse event; at high zoom that could create a temporary bitmap
            # tens of thousands of pixels wide even though only ~1000 px were
            # visible in the canvas.
            source_left = max(0, int(math.floor((-ox) / scale)) - 2)
            source_top = max(0, int(math.floor((-oy) / scale)) - 2)
            source_right = min(width, int(math.ceil((canvas_width - ox) / scale)) + 2)
            source_bottom = min(height, int(math.ceil((canvas_height - oy) / scale)) + 2)
            if source_right > source_left and source_bottom > source_top:
                if mask_key:
                    mask = self.mask_cache[self.current_entry.rel_path][
                        source_top:source_bottom, source_left:source_right
                    ]
                    crop = np.repeat(mask[:, :, None], 3, axis=2)
                else:
                    crop = image[source_top:source_bottom, source_left:source_right]
                target_width = max(1, round((source_right - source_left) * scale))
                target_height = max(1, round((source_bottom - source_top) * scale))
                resampling = Image.Resampling.BILINEAR if fast else Image.Resampling.LANCZOS
                resized = Image.fromarray(crop).resize((target_width, target_height), resampling)
                self.photo = ImageTk.PhotoImage(resized)
                target_x = ox + source_left * scale
                target_y = oy + source_top * scale
                self.canvas.create_image(
                    target_x, target_y, anchor=tk.NW, image=self.photo, tags=("background",),
                )
            else:
                self.photo = None
            self.background_key = key
        self.redraw_overlay()

    def _schedule_interactive_render(self) -> None:
        """Coalesce dense wheel/drag events into screen-rate preview frames."""
        if self._interactive_render_id is None:
            self._interactive_render_id = self.after(16, self._run_interactive_render)
        if self._quality_render_id is not None:
            self.after_cancel(self._quality_render_id)
        self._quality_render_id = self.after(160, self._run_quality_render)

    def _run_interactive_render(self) -> None:
        self._interactive_render_id = None
        if self.winfo_exists():
            self.render_canvas(True, fast=True)

    def _run_quality_render(self) -> None:
        self._quality_render_id = None
        if self.winfo_exists():
            self.render_canvas(True, fast=False)

    def _to_canvas(self, point: tuple[float, float]) -> tuple[float, float]:
        return self.display_offset[0] + point[0] * self.display_scale, self.display_offset[1] + point[1] * self.display_scale

    def _to_image(self, x: float, y: float) -> tuple[int, int] | None:
        if not self.current_entry:
            return None
        image = self.load_image(self.current_entry)
        ix = (x - self.display_offset[0]) / self.display_scale
        iy = (y - self.display_offset[1]) / self.display_scale
        if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0]:
            return int(round(ix)), int(round(iy))
        return None

    def redraw_overlay(self) -> None:
        self.canvas.delete("overlay")
        entry = self.current_entry
        if not entry:
            return
        image = self.load_image(entry)
        height, width = image.shape[:2]
        grid_color = self._valid_color(self.grid_color_var, "#64748b")
        grid_width = max(1, int(self.grid_width_var.get()))
        for column in range(1, entry.display_cols):
            self.canvas.create_line(*self._to_canvas((column * width / entry.display_cols, 0)), *self._to_canvas((column * width / entry.display_cols, height)), fill=grid_color, width=grid_width, dash=(3, 5), tags="overlay")
        for row in range(1, entry.display_rows):
            self.canvas.create_line(*self._to_canvas((0, row * height / entry.display_rows)), *self._to_canvas((width, row * height / entry.display_rows)), fill=grid_color, width=grid_width, dash=(3, 5), tags="overlay")
        for spot in entry.spots:
            result = self.results.get(spot.uid)
            if not result:
                continue
            contour = result.get("contour")
            selected = spot.uid == self.selected_spot_uid
            is_manual = result.get("status") in {"手动画", "手动轮廓"}
            color = self._valid_color(self.selected_color_var, "#f59e0b") if selected else (self._valid_color(self.manual_color_var, "#38bdf8") if is_manual else self._valid_color(self.contour_color_var, "#22c55e"))
            contour_width = max(1, int(self.selected_width_var.get() if selected else (self.manual_width_var.get() if is_manual else self.contour_width_var.get())))
            if contour and self.show_contours_var.get():
                points = [coordinate for point in contour for coordinate in self._to_canvas((point[0], point[1]))]
                if len(points) >= 6:
                    self.canvas.create_line(*points, *points[:2], fill=color, width=contour_width, tags="overlay")
            display_col = self.spot_display_col(entry, spot)
            x = (display_col - 1) * width / entry.display_cols + 8
            y = (spot.display_row - 1) * height / entry.display_rows + 8
            value = result.get("area_um2") if self.scale_um_per_px is not None else result.get("area_px")
            suffix = "µm²" if self.scale_um_per_px is not None else "px²"
            text = f"{spot.atten}倍 no.{spot.serial}" + (f"\n{value:.5g} {suffix}" if value is not None else "")
            tx, ty = self._to_canvas((x, y))
            label_color = self._valid_color(self.selected_color_var, "#f59e0b") if selected else self._valid_color(self.label_color_var, "#22c55e")
            self.canvas.create_text(round(tx), round(ty), text=text, anchor=tk.NW, fill=label_color, font=self.annotation_font, tags="overlay")
        if self.scale_line:
            self.canvas.create_line(*self._to_canvas(self.scale_line[0]), *self._to_canvas(self.scale_line[1]), fill=self._valid_color(self.scale_line_color_var, "#ffffff"), width=max(1, int(self.scale_line_width_var.get())), tags="overlay")
        if self.draw_points:
            points = [coordinate for point in self.draw_points for coordinate in self._to_canvas(point)]
            if len(points) >= 4:
                self.canvas.create_line(*points, fill=self._valid_color(self.manual_color_var, "#38bdf8"), width=max(1, int(self.manual_width_var.get())), tags="overlay")
            for point in self.draw_points:
                cx, cy = self._to_canvas(point)
                radius = max(3, int(self.manual_width_var.get()) + 1)
                self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, fill=self._valid_color(self.manual_color_var, "#38bdf8"), outline="", tags="overlay")

    def _wheel(self, event: tk.Event) -> None:
        if not self.current_entry:
            return
        old_scale, old_ox, old_oy, _old_width, _old_height = self._geometry()
        image = self.load_image(self.current_entry)
        image_x = (event.x - old_ox) / old_scale
        image_y = (event.y - old_oy) / old_scale
        before = (
            (image_x, image_y)
            if 0 <= image_x < image.shape[1] and 0 <= image_y < image.shape[0]
            else None
        )
        old_zoom = self.view_zoom
        self.view_zoom = min(12.0, max(0.25, self.view_zoom * (1.15 if event.delta > 0 else 1 / 1.15)))
        if before and self.view_zoom != old_zoom:
            new_scale, new_ox, new_oy, _width, _height = self._geometry()
            after_canvas = (
                new_ox + before[0] * new_scale,
                new_oy + before[1] * new_scale,
            )
            self.view_pan[0] += event.x - after_canvas[0]
            self.view_pan[1] += event.y - after_canvas[1]
            final_scale, final_ox, final_oy, _width, _height = self._geometry()
            self.display_scale = final_scale
            self.display_offset = (final_ox, final_oy)
        self.background_key = None
        self._schedule_interactive_render()

    def _pan_press(self, event: tk.Event) -> None:
        self.drag_start = (event.x, event.y)
        self.pan_start = tuple(self.view_pan)
        self.pan_last = (event.x, event.y)

    def _pan_drag(self, event: tk.Event) -> None:
        if self.drag_start:
            new_pan = [
                self.pan_start[0] + event.x - self.drag_start[0],
                self.pan_start[1] + event.y - self.drag_start[1],
            ]
            delta_x = new_pan[0] - self.view_pan[0]
            delta_y = new_pan[1] - self.view_pan[1]
            self.view_pan = new_pan
            # Moving the existing raster and vector items is essentially
            # instantaneous.  A cropped raster refresh follows at screen rate.
            self.canvas.move("all", delta_x, delta_y)
            self.display_offset = (
                self.display_offset[0] + delta_x,
                self.display_offset[1] + delta_y,
            )
            self.pan_last = (event.x, event.y)
            self.background_key = None
            self._schedule_interactive_render()

    def _pan_release(self, _event: tk.Event) -> None:
        if self.drag_start is None:
            return
        self.drag_start = None
        self.background_key = None
        if self._interactive_render_id is not None:
            self.after_cancel(self._interactive_render_id)
            self._interactive_render_id = None
        self.render_canvas(True, fast=True)
        if self._quality_render_id is not None:
            self.after_cancel(self._quality_render_id)
        self._quality_render_id = self.after(80, self._run_quality_render)

    def _left_press(self, event: tk.Event) -> None:
        point = self._to_image(event.x, event.y)
        if point is None:
            return
        mode = self.mode_var.get()
        if mode == "scale":
            self.scale_drag_start = point
            self.scale_line = (point, point)
            self.redraw_overlay()
        elif mode == "draw":
            self.draw_points.append(point)
            # Only overlays are redrawn; the multi-megapixel image is not resized again.
            self.redraw_overlay()
        elif mode == "pick":
            self.pick_candidate(point)

    def _left_drag(self, event: tk.Event) -> None:
        if self.mode_var.get() == "scale" and self.scale_drag_start:
            point = self._to_image(event.x, event.y)
            if point:
                self.scale_line = (self.scale_drag_start, point)
                self.redraw_overlay()

    def _left_release(self, _event: tk.Event) -> None:
        self.scale_drag_start = None

    def reset_view(self) -> None:
        self.view_zoom = 1.0
        self.view_pan = [0.0, 0.0]
        self.render_canvas(True)

    def pick_candidate(self, point: tuple[int, int]) -> None:
        spot = self.selected_spot()
        if not spot or not self.current_entry:
            messagebox.showinfo("未选择坑位", "请先在右侧选择坑位。", parent=self)
            return
        if self.current_entry.rel_path not in self.candidate_cache:
            self.sync_params()
            self.process_entry(self.current_entry)
        best: tuple[float, Candidate] | None = None
        for candidate in self.candidate_cache.get(self.current_entry.rel_path, []):
            contour = np.asarray(candidate.contour, dtype=np.int32).reshape(-1, 1, 2)
            inside = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
            distance = 0.0 if inside >= 0 else math.dist(point, candidate.centroid)
            if best is None or distance < best[0]:
                best = (distance, candidate)
        if best is None or best[0] > 120:
            messagebox.showinfo("未选中", "附近没有候选轮廓。", parent=self)
            return
        self.results[spot.uid] = self._result(best[1].contour, best[1].area_px, "手动轮廓")
        self._changed_area()

    def finish_polygon(self, _event=None) -> None:
        if self.mode_var.get() != "draw" or len(self.draw_points) < 3:
            return
        spot = self.selected_spot()
        if not spot:
            return
        contour = np.asarray(self.draw_points, dtype=np.int32).reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(contour)))
        self.results[spot.uid] = self._result([[x, y] for x, y in self.draw_points], area, "手动画")
        self.draw_points.clear()
        self._changed_area()

    def apply_scale(self) -> None:
        if not self.scale_line:
            messagebox.showinfo("未绘制比例尺", "请选择比例尺模式并在图像上拖动。", parent=self)
            return
        length_px = math.dist(*self.scale_line)
        if length_px <= 0:
            return
        self.scale_um_per_px = float(self.scale_length_var.get()) / length_px
        self._recompute_scaled_areas()
        self.scale_info.configure(text=f"{self.scale_um_per_px:.7g} µm/px；面积系数 {self.scale_um_per_px**2:.7g} µm²/px²")
        self._sync_display_style_to_state()
        self._changed_area()

    def clear_scale(self) -> None:
        self.scale_um_per_px = None
        self.scale_line = None
        self._recompute_scaled_areas()
        self.scale_info.configure(text="未设置比例尺")
        self._sync_display_style_to_state()
        self._changed_area()

    def _recompute_scaled_areas(self) -> None:
        for result in self.results.values():
            area_px = result.get("area_px")
            result["area_um2"] = area_px * self.scale_um_per_px**2 if area_px is not None and self.scale_um_per_px is not None else None

    def export_excel(self) -> None:
        if not self.state.records:
            messagebox.showinfo("没有数据", "尚无面积数据。", parent=self)
            return
        initial_dir = self.path_store.initial_dir("area_excel_save", self.root_dir or Path.cwd()) if self.path_store else str(self.root_dir or Path.cwd())
        filename = filedialog.asksaveasfilename(parent=self, initialdir=initial_dir, title="保存面积表", defaultextension=".xlsx", filetypes=[("Excel 工作簿", "*.xlsx")], initialfile="XFEL_damage_area.xlsx")
        if not filename:
            return
        if self.path_store:
            self.path_store.remember_file("area_excel_save", filename)
        from openpyxl import Workbook
        from openpyxl.styles import Font
        workbook = Workbook()
        workbook.remove(workbook.active)
        for group in self.state.groups():
            sheet = workbook.create_sheet(title=(f"atten_{group}")[:31])
            sheet.append(["序号", "损伤面积", "单位", "状态", "图像文件"])
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for record in self.state.group_records(group):
                sheet.append([record.serial, record.area, record.area_unit, record.area_status, record.image_path])
            sheet.freeze_panes = "A2"
            sheet.column_dimensions["A"].width = 10
            sheet.column_dimensions["B"].width = 16
            sheet.column_dimensions["C"].width = 10
            sheet.column_dimensions["D"].width = 14
            sheet.column_dimensions["E"].width = 60
        workbook.save(filename)
        self.set_status(f"面积表已导出：{filename}")

    def export_config(self) -> dict[str, Any]:
        return {
            "folder": str(self.root_dir) if self.root_dir else "",
            "params": asdict(self.params),
            "scale_um_per_px": self.scale_um_per_px,
            "scale_line": self.scale_line,
            "scale_length": self.scale_length_var.get(),
            "results": self.results,
            "directions": self.directions,
            "segmentation_mode": self.segmentation_mode_var.get(),
            "column_order": self.column_order_var.get(),
            "infer_missing_l": self.infer_missing_l_var.get(),
            "show_contours": self.show_contours_var.get(),
            "display_style": dict(self.state.area_display_style),
            "colors": {"contour": self.contour_color_var.get(), "manual": self.manual_color_var.get()},
        }

    def _apply_config_fields(self, data: dict[str, Any]) -> None:
        params = data.get("params", {})
        params = dict(params)
        aliases = {"bright_pit": "bright_damage", "blur": "median_blur", "close": "close_kernel"}
        for old, new in aliases.items():
            if old in params and new not in params:
                params[new] = params[old]
        self.params = ProcessingParameters(**{key: value for key, value in params.items() if key in ProcessingParameters.__dataclass_fields__})
        self.gray_low_var.set(self.params.gray_low)
        self.gray_high_var.set(self.params.gray_high)
        self.bright_var.set(self.params.bright_damage)
        self.min_area_var.set(self.params.min_area_px)
        self.max_area_var.set(self.params.max_area_px)
        self.blur_var.set(self.params.median_blur)
        self.close_var.set(self.params.close_kernel)
        scale_data = data.get("scale", {})
        self.scale_um_per_px = data.get("scale_um_per_px", scale_data.get("um_per_px"))
        line = data.get("scale_line", scale_data.get("line"))
        self.scale_line = tuple(tuple(map(int, point)) for point in line) if line else None  # type: ignore[assignment]
        self.scale_length_var.set(data.get("scale_length", scale_data.get("actual_um", 80.0)))
        raw_results = dict(data.get("results", {}))
        current_uids = {spot.uid for entry in self.entries for spot in entry.spots}
        if raw_results and not any(uid in current_uids for uid in raw_results):
            converted: dict[str, dict[str, Any]] = {}
            for entry in self.entries:
                for index, spot in enumerate(entry.spots, 1):
                    prefix = f"{entry.rel_path}#{index}:{spot.atten}:r{spot.row}:c{spot.col}:"
                    match = next((value for old_uid, value in raw_results.items() if old_uid.startswith(prefix)), None)
                    if match is not None:
                        converted[spot.uid] = match
            raw_results = converted
        self.results = raw_results
        self.show_contours_var.set(data.get("show_contours", True))
        colors = data.get("colors", {})
        legacy_display = data.get("display", {})
        style = data.get("display_style", {})
        self.contour_color_var.set(style.get("auto_contour_color", legacy_display.get("auto_contour_color", colors.get("contour", "#22c55e"))))
        self.contour_width_var.set(style.get("auto_contour_width", 2))
        self.manual_color_var.set(style.get("manual_contour_color", legacy_display.get("manual_line_color", colors.get("manual", "#38bdf8"))))
        self.manual_width_var.set(style.get("manual_contour_width", legacy_display.get("manual_line_width", 2)))
        self.selected_color_var.set(style.get("selected_color", "#f59e0b"))
        self.selected_width_var.set(style.get("selected_width", 4))
        self.label_color_var.set(style.get("label_color", legacy_display.get("label_color", "#22c55e")))
        self.label_font_size_var.set(style.get("label_font_size", 11))
        self.scale_line_color_var.set(style.get("scale_line_color", legacy_display.get("scale_line_color", "#ffffff")))
        self.scale_line_width_var.set(style.get("scale_line_width", legacy_display.get("scale_line_width", 3)))
        self.grid_color_var.set(style.get("grid_color", "#64748b"))
        self.grid_width_var.set(style.get("grid_width", 1))
        self.apply_display_style()
        if self.scale_um_per_px is not None:
            self.scale_info.configure(text=f"{self.scale_um_per_px:.7g} µm/px；面积系数 {self.scale_um_per_px**2:.7g} µm²/px²")

    def import_config(self, data: dict[str, Any]) -> None:
        folder_text = data.get("folder", "")
        self.segmentation_mode_var.set(str(data.get("segmentation_mode", "normal")))
        self._update_direction_controls()
        self.infer_missing_l_var.set(data.get("infer_missing_l", True))
        if folder_text and Path(folder_text).is_dir():
            self.load_folder(Path(folder_text), data)
        else:
            self._apply_config_fields(data)
