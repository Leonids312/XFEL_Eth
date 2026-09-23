"""Step 3: interactive plots, threshold fitting, point details and recycle bin."""

from __future__ import annotations

import math
import os
from datetime import datetime
from io import BytesIO
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

import numpy as np
from matplotlib import rcParams
from matplotlib import ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import is_color_like
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from PIL import Image, ImageTk

from remembering_toolbar import RememberingNavigationToolbar
from xfel_core import PointRecord, ProjectState, RecentPathStore, ThresholdFit, attenuation_sort_key, fit_threshold


DEFAULT_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c"]
MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*", "+", "x")

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class ThresholdAnalysisPage(ttk.Frame):
    def __init__(self, master: tk.Misc, state: ProjectState, status_callback=None, path_store: RecentPathStore | None = None) -> None:
        super().__init__(master)
        self.state = state
        self.status_callback = status_callback or (lambda text: None)
        self.path_store = path_store
        self.plot_canvases: dict[str, FigureCanvasTkAgg] = {}
        self.figures: dict[str, Figure] = {}
        self.axis_points: dict[Any, list[tuple[PointRecord, float, float]]] = {}
        self.group_styles: dict[str, dict[str, Any]] = {}
        self.plot_preview_context: dict[str, dict[str, Any]] = {}
        self.plot_history: list[dict[str, Any]] = []
        self.active_history_id = ""
        self.last_fit: ThresholdFit | None = None
        self._detail_photo: ImageTk.PhotoImage | None = None

        self.title_var = tk.StringVar(value="XFEL 单脉冲损伤阈值拟合")
        self.xlabel_var = tk.StringVar(value="单脉冲能量 E_pulse (nJ)")
        self.ylabel_var = tk.StringVar(value="损伤面积 S (µm²)")
        self.xscale_var = tk.StringVar(value="log")
        self.yscale_var = tk.StringVar(value="linear")
        self.xmin_var = tk.StringVar()
        self.xmax_var = tk.StringVar()
        self.ymin_var = tk.StringVar(value="0")
        self.ymax_var = tk.StringVar()
        self.title_font_size_var = tk.DoubleVar(value=16.0)
        self.xlabel_font_size_var = tk.DoubleVar(value=13.0)
        self.ylabel_font_size_var = tk.DoubleVar(value=13.0)
        self.xtick_font_size_var = tk.DoubleVar(value=11.0)
        self.ytick_font_size_var = tk.DoubleVar(value=11.0)
        self.legend_font_size_var = tk.DoubleVar(value=10.0)
        self.title_pad_var = tk.DoubleVar(value=8.0)
        self.xlabel_pad_var = tk.DoubleVar(value=6.0)
        self.ylabel_pad_var = tk.DoubleVar(value=6.0)
        self.x_major_step_var = tk.StringVar()
        self.y_major_step_var = tk.StringVar()
        self.x_minor_divisions_var = tk.IntVar(value=5)
        self.y_minor_divisions_var = tk.IntVar(value=5)
        self.show_minor_ticks_var = tk.BooleanVar(value=True)
        self.tick_direction_var = tk.StringVar(value="out")
        self.major_tick_length_var = tk.DoubleVar(value=6.0)
        self.major_tick_width_var = tk.DoubleVar(value=1.0)
        self.minor_tick_length_var = tk.DoubleVar(value=3.5)
        self.minor_tick_width_var = tk.DoubleVar(value=0.8)
        self.xtick_rotation_var = tk.DoubleVar(value=0.0)
        self.ytick_rotation_var = tk.DoubleVar(value=0.0)
        self.show_top_ticks_var = tk.BooleanVar(value=False)
        self.show_right_ticks_var = tk.BooleanVar(value=False)
        self.show_top_labels_var = tk.BooleanVar(value=False)
        self.show_right_labels_var = tk.BooleanVar(value=False)
        self.x_tick_format_var = tk.StringVar(value="自动")
        self.y_tick_format_var = tk.StringVar(value="自动")
        self.x_log_base_var = tk.DoubleVar(value=10.0)
        self.y_log_base_var = tk.DoubleVar(value=10.0)
        self.scientific_min_var = tk.IntVar(value=-3)
        self.scientific_max_var = tk.IntVar(value=4)
        self.axis_line_width_var = tk.DoubleVar(value=1.0)
        self.axis_text_color_var = tk.StringVar(value="#111827")
        self.tick_color_var = tk.StringVar(value="#111827")
        self.spine_color_var = tk.StringVar(value="#111827")
        self.grid_color_var = tk.StringVar(value="#94a3b8")
        self.grid_which_var = tk.StringVar(value="主刻度")
        self.grid_line_style_var = tk.StringVar(value="--")
        self.grid_line_width_var = tk.DoubleVar(value=0.8)
        self.grid_alpha_var = tk.DoubleVar(value=0.35)
        self.marker_size_var = tk.DoubleVar(value=42.0)
        self.dpi_var = tk.IntVar(value=300)
        self.line_color_var = tk.StringVar(value="#dc2626")
        self.line_style_var = tk.StringVar(value="-")
        self.line_width_var = tk.DoubleVar(value=2.0)
        self.show_grid_var = tk.BooleanVar(value=True)
        self.show_legend_var = tk.BooleanVar(value=True)
        self.show_fit_line_var = tk.BooleanVar(value=True)
        self.show_xerr_var = tk.BooleanVar(value=False)
        self.show_yerr_var = tk.BooleanVar(value=False)
        self.range_filter_var = tk.BooleanVar(value=False)
        self.fit_emin_var = tk.StringVar()
        self.fit_emax_var = tk.StringVar()
        self.fit_smin_var = tk.StringVar()
        self.fit_smax_var = tk.StringVar()
        self.attenuation_filter_var = tk.BooleanVar(value=False)
        self.selected_fit_attenuations: set[str] = set()
        self.attenuation_filter_summary_var = tk.StringVar(value="全部衰减倍数参与拟合")
        self.sigma_clip_var = tk.BooleanVar(value=False)
        self.sigma_n_var = tk.DoubleVar(value=3.0)
        self.sigma_iter_var = tk.IntVar(value=3)
        self.result_var = tk.StringVar(value="第 3 步有完整的能量和面积数据后，点击“绘图并拟合阈值”。")
        self._build_ui()
        self.state.subscribe(self._state_changed)

    def _build_ui(self) -> None:
        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        controls_outer = ttk.Frame(body, padding=7)
        plots = ttk.Frame(body, padding=5)
        body.add(controls_outer, weight=0)
        body.add(plots, weight=1)

        control_canvas = tk.Canvas(controls_outer, width=345, highlightthickness=0)
        control_scroll = ttk.Scrollbar(controls_outer, orient=tk.VERTICAL, command=control_canvas.yview)
        controls = ttk.Frame(control_canvas, padding=(2, 2, 8, 12))
        controls.bind("<Configure>", lambda _event: control_canvas.configure(scrollregion=control_canvas.bbox("all")))
        control_canvas.create_window((0, 0), window=controls, anchor=tk.NW)
        control_canvas.configure(yscrollcommand=control_scroll.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        plot_box = ttk.LabelFrame(controls, text="图形设置", padding=7)
        plot_box.pack(fill=tk.X)
        self._entry_row(plot_box, 0, "标题", self.title_var)
        self._entry_row(plot_box, 1, "X 轴标题", self.xlabel_var)
        self._entry_row(plot_box, 2, "Y 轴标题", self.ylabel_var)
        ttk.Label(plot_box, text="X 轴尺度").grid(row=3, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(plot_box, textvariable=self.xscale_var, state="readonly", values=("log", "linear"), width=12).grid(row=3, column=1, sticky=tk.EW, pady=3)
        ranges = ttk.Frame(plot_box)
        ranges.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=3)
        for index, (label, variable) in enumerate((("Xmin", self.xmin_var), ("Xmax", self.xmax_var), ("Ymin", self.ymin_var), ("Ymax", self.ymax_var))):
            ttk.Label(ranges, text=label).grid(row=index // 2, column=(index % 2) * 2, sticky=tk.W)
            ttk.Entry(ranges, textvariable=variable, width=9).grid(row=index // 2, column=(index % 2) * 2 + 1, padx=(3, 6), pady=2)
        self._entry_row(plot_box, 5, "散点大小", self.marker_size_var)
        ttk.Label(plot_box, text="保存 DPI").grid(row=6, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(
            plot_box,
            textvariable=self.dpi_var,
            values=(72, 96, 100, 150, 200, 300, 600, 1200),
            width=12,
        ).grid(row=6, column=1, sticky=tk.EW, pady=3)
        self._entry_row(plot_box, 7, "拟合线颜色", self.line_color_var)
        ttk.Label(plot_box, text="拟合线型").grid(row=8, column=0, sticky=tk.W, pady=3)
        ttk.Combobox(plot_box, textvariable=self.line_style_var, state="readonly", values=("-", "--", "-.", ":"), width=12).grid(row=8, column=1, sticky=tk.EW, pady=3)
        self._entry_row(plot_box, 9, "拟合线宽", self.line_width_var)
        ttk.Checkbutton(plot_box, text="网格", variable=self.show_grid_var).grid(row=10, column=0, sticky=tk.W)
        ttk.Checkbutton(plot_box, text="图例", variable=self.show_legend_var).grid(row=10, column=1, sticky=tk.W)
        ttk.Checkbutton(plot_box, text="显示拟合直线", variable=self.show_fit_line_var).grid(row=11, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(plot_box, text="总图显示 X 误差棒", variable=self.show_xerr_var).grid(row=12, column=0, columnspan=2, sticky=tk.W)
        ttk.Checkbutton(plot_box, text="总图显示 Y 误差棒", variable=self.show_yerr_var).grid(row=13, column=0, columnspan=2, sticky=tk.W)
        ttk.Button(plot_box, text="坐标轴、标题字号与刻度…", command=self.open_axis_settings_dialog).grid(row=14, column=0, columnspan=2, sticky=tk.EW, pady=(7, 0))
        ttk.Button(plot_box, text="逐衰减组设置颜色/点形", command=self.open_style_dialog).grid(row=15, column=0, columnspan=2, sticky=tk.EW, pady=(5, 0))
        ttk.Button(plot_box, text="应用图形设置到当前预览", command=self.apply_plot_settings).grid(row=16, column=0, columnspan=2, sticky=tk.EW, pady=(5, 0))
        plot_box.columnconfigure(1, weight=1)

        fit_box = ttk.LabelFrame(controls, text="拟合筛选", padding=7)
        fit_box.pack(fill=tk.X, pady=(7, 0))
        ttk.Checkbutton(fit_box, text="启用能量/面积范围", variable=self.range_filter_var).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        for index, (label, variable) in enumerate((("E min", self.fit_emin_var), ("E max", self.fit_emax_var), ("S min", self.fit_smin_var), ("S max", self.fit_smax_var))):
            row, column = 1 + index // 2, (index % 2) * 2
            ttk.Label(fit_box, text=label).grid(row=row, column=column, sticky=tk.W, pady=3)
            ttk.Entry(fit_box, textvariable=variable, width=10).grid(row=row, column=column + 1, padx=(3, 7), pady=3)
        ttk.Checkbutton(
            fit_box, text="按衰减倍数筛选", variable=self.attenuation_filter_var,
            command=self._attenuation_filter_toggled,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        ttk.Button(fit_box, text="选择衰减倍数…", command=self.open_attenuation_filter_dialog).grid(
            row=3, column=2, columnspan=2, sticky=tk.EW, padx=(4, 7), pady=(6, 0),
        )
        ttk.Label(
            fit_box, textvariable=self.attenuation_filter_summary_var,
            foreground="#475569", wraplength=305,
        ).grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(3, 2))
        ttk.Separator(fit_box, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=4)
        ttk.Checkbutton(fit_box, text="Sigma 异常值剔除", variable=self.sigma_clip_var).grid(row=6, column=0, columnspan=4, sticky=tk.W, pady=(2, 0))
        ttk.Label(fit_box, text="阈值 ±nσ").grid(row=7, column=0, sticky=tk.W)
        ttk.Entry(fit_box, textvariable=self.sigma_n_var, width=8).grid(row=7, column=1)
        ttk.Label(fit_box, text="最大迭代").grid(row=7, column=2, sticky=tk.W)
        ttk.Entry(fit_box, textvariable=self.sigma_iter_var, width=8).grid(row=7, column=3)

        ttk.Button(controls, text="绘图并拟合阈值", command=self.generate_plots).pack(fill=tk.X, pady=(9, 3))
        ttk.Button(controls, text="保存当前图", command=self.save_current_figure).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="删除点回收站…", command=self.open_recycle_bin).pack(fill=tk.X, pady=3)
        ttk.Label(controls, text="右键单击散点：查看信息或移入回收站", foreground="#475569").pack(anchor=tk.W, pady=(8, 4))
        ttk.Label(controls, textvariable=self.result_var, wraplength=325, justify=tk.LEFT).pack(fill=tk.X, pady=(6, 0))

        plot_body = ttk.Panedwindow(plots, orient=tk.HORIZONTAL)
        plot_body.pack(fill=tk.BOTH, expand=True)
        preview_frame = ttk.Frame(plot_body)
        history_frame = ttk.LabelFrame(plot_body, text="本次运行绘图缓存", padding=5, width=205)
        plot_body.add(preview_frame, weight=1)
        plot_body.add(history_frame, weight=0)
        self.plot_notebook = ttk.Notebook(preview_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)

        self.history_canvas = tk.Canvas(history_frame, width=190, highlightthickness=0, background="#f8fafc")
        history_scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_canvas.yview)
        self.history_list = ttk.Frame(self.history_canvas)
        self.history_window = self.history_canvas.create_window((0, 0), window=self.history_list, anchor=tk.NW)
        self.history_list.bind("<Configure>", lambda _event: self.history_canvas.configure(scrollregion=self.history_canvas.bbox("all")))
        self.history_canvas.bind("<Configure>", lambda event: self.history_canvas.itemconfigure(self.history_window, width=max(170, event.width)))
        self.history_canvas.configure(yscrollcommand=history_scroll.set)
        self.history_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Label(self.history_list, text="每次“绘图并拟合”保存一张总图；\n单衰减散点图不进入缓存。", foreground="#64748b", justify=tk.LEFT).pack(fill=tk.X, padx=3, pady=4)

    @staticmethod
    def _entry_row(parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky=tk.EW, pady=3)

    def open_axis_settings_dialog(self) -> None:
        window = tk.Toplevel(self)
        window.title("坐标轴、标题字号与刻度设置")
        window.geometry("780x690")
        window.minsize(700, 610)
        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        axis_page = ttk.Frame(notebook, padding=12)
        font_page = ttk.Frame(notebook, padding=12)
        tick_page = ttk.Frame(notebook, padding=12)
        notebook.add(axis_page, text="范围、尺度与定位")
        notebook.add(font_page, text="标题与文字")
        notebook.add(tick_page, text="刻度、边框与网格")

        def entry(parent: ttk.Frame, row: int, column: int, label: str, variable: tk.Variable, width: int = 12) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, padx=(3, 5), pady=5)
            ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=column + 1, sticky=tk.EW, padx=(0, 14), pady=5)

        def combo(parent: ttk.Frame, row: int, column: int, label: str, variable: tk.Variable, values: tuple[str, ...]) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=column, sticky=tk.W, padx=(3, 5), pady=5)
            ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=12).grid(
                row=row, column=column + 1, sticky=tk.EW, padx=(0, 14), pady=5,
            )

        ttk.Label(axis_page, text="坐标范围留空表示自动；主刻度间隔留空表示由 Matplotlib 自动确定。", foreground="#475569").grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8),
        )
        combo(axis_page, 1, 0, "X 轴尺度", self.xscale_var, ("linear", "log"))
        combo(axis_page, 1, 2, "Y 轴尺度", self.yscale_var, ("linear", "log"))
        entry(axis_page, 2, 0, "X 最小值", self.xmin_var)
        entry(axis_page, 2, 2, "X 最大值", self.xmax_var)
        entry(axis_page, 3, 0, "Y 最小值", self.ymin_var)
        entry(axis_page, 3, 2, "Y 最大值", self.ymax_var)
        entry(axis_page, 4, 0, "X 主刻度间隔", self.x_major_step_var)
        entry(axis_page, 4, 2, "Y 主刻度间隔", self.y_major_step_var)
        entry(axis_page, 5, 0, "X 次刻度分段数", self.x_minor_divisions_var)
        entry(axis_page, 5, 2, "Y 次刻度分段数", self.y_minor_divisions_var)
        combo(axis_page, 6, 0, "X 刻度格式", self.x_tick_format_var, ("自动", "普通数字", "科学计数"))
        combo(axis_page, 6, 2, "Y 刻度格式", self.y_tick_format_var, ("自动", "普通数字", "科学计数"))
        entry(axis_page, 7, 0, "X 对数底数", self.x_log_base_var)
        entry(axis_page, 7, 2, "Y 对数底数", self.y_log_base_var)
        entry(axis_page, 8, 0, "科学计数下限指数", self.scientific_min_var)
        entry(axis_page, 8, 2, "科学计数上限指数", self.scientific_max_var)
        axis_page.columnconfigure(1, weight=1)
        axis_page.columnconfigure(3, weight=1)

        ttk.Label(font_page, text="标题、坐标轴标题、刻度文字和图例字号可分别设置。", foreground="#475569").grid(
            row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8),
        )
        entry(font_page, 1, 0, "图标题字号", self.title_font_size_var)
        entry(font_page, 1, 2, "图标题间距", self.title_pad_var)
        entry(font_page, 2, 0, "X 轴标题字号", self.xlabel_font_size_var)
        entry(font_page, 2, 2, "X 轴标题间距", self.xlabel_pad_var)
        entry(font_page, 3, 0, "Y 轴标题字号", self.ylabel_font_size_var)
        entry(font_page, 3, 2, "Y 轴标题间距", self.ylabel_pad_var)
        entry(font_page, 4, 0, "X 刻度字号", self.xtick_font_size_var)
        entry(font_page, 4, 2, "Y 刻度字号", self.ytick_font_size_var)
        entry(font_page, 5, 0, "图例字号", self.legend_font_size_var)
        entry(font_page, 5, 2, "X 刻度旋转角", self.xtick_rotation_var)
        entry(font_page, 6, 0, "Y 刻度旋转角", self.ytick_rotation_var)
        entry(font_page, 6, 2, "坐标轴文字颜色", self.axis_text_color_var)
        font_page.columnconfigure(1, weight=1)
        font_page.columnconfigure(3, weight=1)

        ttk.Checkbutton(tick_page, text="显示次刻度", variable=self.show_minor_ticks_var).grid(row=0, column=0, sticky=tk.W, padx=3, pady=5)
        combo(tick_page, 0, 2, "刻度方向", self.tick_direction_var, ("out", "in", "inout"))
        entry(tick_page, 1, 0, "主刻度长度", self.major_tick_length_var)
        entry(tick_page, 1, 2, "主刻度线宽", self.major_tick_width_var)
        entry(tick_page, 2, 0, "次刻度长度", self.minor_tick_length_var)
        entry(tick_page, 2, 2, "次刻度线宽", self.minor_tick_width_var)
        ttk.Checkbutton(tick_page, text="顶部刻度线", variable=self.show_top_ticks_var).grid(row=3, column=0, sticky=tk.W, padx=3, pady=5)
        ttk.Checkbutton(tick_page, text="右侧刻度线", variable=self.show_right_ticks_var).grid(row=3, column=1, sticky=tk.W, padx=3, pady=5)
        ttk.Checkbutton(tick_page, text="顶部刻度标签", variable=self.show_top_labels_var).grid(row=3, column=2, sticky=tk.W, padx=3, pady=5)
        ttk.Checkbutton(tick_page, text="右侧刻度标签", variable=self.show_right_labels_var).grid(row=3, column=3, sticky=tk.W, padx=3, pady=5)
        entry(tick_page, 4, 0, "坐标轴边框线宽", self.axis_line_width_var)
        entry(tick_page, 4, 2, "刻度线/文字颜色", self.tick_color_var)
        entry(tick_page, 5, 0, "坐标轴边框颜色", self.spine_color_var)
        entry(tick_page, 5, 2, "网格颜色", self.grid_color_var)
        combo(tick_page, 6, 0, "网格范围", self.grid_which_var, ("主刻度", "主+次刻度"))
        combo(tick_page, 6, 2, "网格线型", self.grid_line_style_var, ("-", "--", "-.", ":"))
        entry(tick_page, 7, 0, "网格线宽", self.grid_line_width_var)
        entry(tick_page, 7, 2, "网格透明度", self.grid_alpha_var)
        tick_page.columnconfigure(1, weight=1)
        tick_page.columnconfigure(3, weight=1)

        buttons = ttk.Frame(window, padding=(10, 4, 10, 10))
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="应用到当前预览", command=self.apply_plot_settings).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="关闭", command=window.destroy).pack(side=tk.RIGHT, padx=(0, 7))

    def set_status(self, text: str) -> None:
        self.status_callback(text)

    def _state_changed(self, reason: str) -> None:
        if reason in {"deleted", "restored"} and self.figures:
            self.after_idle(self.generate_plots)

    def _plot_records(self) -> list[PointRecord]:
        records = [record for record in self.state.records.values()
                   if not record.deleted and record.energy is not None and record.area is not None
                   and np.isfinite(record.energy) and np.isfinite(record.area) and record.energy > 0]
        return sorted(records, key=lambda record: (attenuation_sort_key(record.atten), record.serial))

    def _update_attenuation_filter_summary(self) -> None:
        if not self.attenuation_filter_var.get():
            self.attenuation_filter_summary_var.set("全部衰减倍数参与拟合")
            return
        selected = sorted(self.selected_fit_attenuations, key=attenuation_sort_key)
        if not selected:
            self.attenuation_filter_summary_var.set("尚未选择参与拟合的衰减倍数")
            return
        self.attenuation_filter_summary_var.set(
            "参与拟合：" + "、".join(f"{group} 倍" for group in selected)
        )

    def _attenuation_filter_toggled(self) -> None:
        if self.attenuation_filter_var.get() and not self.selected_fit_attenuations:
            self.selected_fit_attenuations = set(self.state.groups())
        self._update_attenuation_filter_summary()
        self.state.notify("threshold_attenuation_filter")

    def open_attenuation_filter_dialog(self) -> None:
        groups = self.state.groups()
        if not groups:
            messagebox.showinfo("没有衰减组", "请先在第 3 步创建或导入衰减组数据。", parent=self)
            return
        window = tk.Toplevel(self)
        window.title("选择参与阈值拟合的衰减倍数")
        window.geometry("360x460")
        window.minsize(320, 300)
        window.transient(self.winfo_toplevel())
        ttk.Label(
            window,
            text="勾选参与拟合的衰减组；未选组仍显示在总图中，但以浅色标记。",
            wraplength=320, justify=tk.LEFT, padding=(10, 10, 10, 5),
        ).pack(fill=tk.X)

        scroll_host = ttk.Frame(window, padding=(10, 0, 10, 0))
        scroll_host.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(scroll_host, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=canvas.yview)
        body = ttk.Frame(canvas, padding=(6, 5, 6, 5))
        window_id = canvas.create_window((0, 0), window=body, anchor=tk.NW)
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window_id, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        selected = self.selected_fit_attenuations if self.selected_fit_attenuations else set(groups)
        variables = {group: tk.BooleanVar(value=group in selected) for group in groups}
        for row, group in enumerate(groups):
            valid_count = sum(
                1 for record in self.state.group_records(group, include_deleted=False)
                if record.energy is not None and record.area is not None
            )
            ttk.Checkbutton(
                body, text=f"{group} 倍衰减（{valid_count} 个可绘图点）",
                variable=variables[group],
            ).grid(row=row, column=0, sticky=tk.W, pady=4)

        buttons = ttk.Frame(window, padding=10)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="全选", command=lambda: [variable.set(True) for variable in variables.values()]).pack(side=tk.LEFT)
        ttk.Button(buttons, text="全不选", command=lambda: [variable.set(False) for variable in variables.values()]).pack(side=tk.LEFT, padx=5)

        def apply_selection() -> None:
            self.selected_fit_attenuations = {group for group, variable in variables.items() if variable.get()}
            self.attenuation_filter_var.set(True)
            self._update_attenuation_filter_summary()
            self.state.notify("threshold_attenuation_filter")
            window.destroy()

        ttk.Button(buttons, text="应用", command=apply_selection).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="取消", command=window.destroy).pack(side=tk.RIGHT, padx=(0, 5))

    @staticmethod
    def _attenuation_selection_mask(
        records: list[PointRecord], selected: set[str],
    ) -> np.ndarray:
        return np.asarray(
            [str(record.atten) in selected for record in records], dtype=bool,
        )

    def _fit_range_mask(self, records: list[PointRecord]) -> np.ndarray:
        mask = np.ones(len(records), dtype=bool)
        if self.attenuation_filter_var.get():
            mask &= self._attenuation_selection_mask(records, self.selected_fit_attenuations)
        if self.range_filter_var.get():
            limits = {
                "emin": self._optional_float(self.fit_emin_var.get()), "emax": self._optional_float(self.fit_emax_var.get()),
                "smin": self._optional_float(self.fit_smin_var.get()), "smax": self._optional_float(self.fit_smax_var.get()),
            }
            for index, record in enumerate(records):
                mask[index] = bool(mask[index]) and (
                    (limits["emin"] is None or record.energy >= limits["emin"])
                    and (limits["emax"] is None or record.energy <= limits["emax"])
                    and (limits["smin"] is None or record.area >= limits["smin"])
                    and (limits["smax"] is None or record.area <= limits["smax"])
                )
        return mask

    @staticmethod
    def _optional_float(text: str) -> float | None:
        return float(text) if str(text).strip() else None

    @staticmethod
    def _format_threshold(fit: ThresholdFit, value_digits: int = 7, error_digits: int = 5) -> str:
        value = f"{fit.threshold:.{value_digits}g}"
        error = f"{fit.threshold_err:.{error_digits}g}" if np.isfinite(fit.threshold_err) else "无法估计"
        return f"{value} ± {error} nJ"

    @classmethod
    def _format_fit_result(
        cls,
        fit: ThresholdFit,
        used_count: int,
        record_count: int,
        range_count: int,
    ) -> str:
        fit_error = f"{fit.threshold_fit_err:.4g}" if np.isfinite(fit.threshold_fit_err) else "无法估计"
        return (
            "拟合模型：S = a·ln(E_pulse) + b\n"
            f"a = {fit.a:.7g} ± {fit.a_err:.4g}\n"
            f"b = {fit.b:.7g} ± {fit.b_err:.4g}\n"
            f"R² = {fit.r_squared:.6f}\n"
            f"E_th = {cls._format_threshold(fit)}\n"
            f"总误差分量：X={fit.threshold_x_err:.4g}，"
            f"Y={fit.threshold_y_err:.4g}，拟合={fit_error} nJ\n"
            f"参与拟合 {used_count}/{record_count} 点；筛选内 {range_count} 点；"
            f"残差 σ={fit.sigma:.5g}。"
        )

    def generate_plots(self) -> None:
        try:
            self._plot_dpi()
            self._validate_axis_settings()
            records = self._plot_records()
            if len(records) < 2:
                raise ValueError("有效点少于 2 个。请先在第 3 步补齐单脉冲能量和损伤面积。")
            range_mask = self._fit_range_mask(records)
            if np.count_nonzero(range_mask) < 2:
                raise ValueError("拟合筛选后的有效点少于 2 个；请检查衰减倍数及能量/面积筛选范围。")
            energies = np.asarray([record.energy for record in records], dtype=float)[range_mask]
            areas = np.asarray([record.area for record in records], dtype=float)[range_mask]
            energy_errors = np.asarray(
                [0.0 if record.energy_rel_err is None else record.energy_rel_err for record in records],
                dtype=float,
            )[range_mask]
            area_errors = np.asarray(
                [0.0 if record.area_rel_err is None else record.area_rel_err for record in records],
                dtype=float,
            )[range_mask]
            sigma = float(self.sigma_n_var.get()) if self.sigma_clip_var.get() else None
            fit = fit_threshold(
                energies, areas, sigma, int(self.sigma_iter_var.get()),
                energy_rel_err=energy_errors, area_rel_err=area_errors,
            )
        except Exception as exc:
            messagebox.showerror("无法拟合", str(exc), parent=self)
            return
        try:
            self.last_fit = fit
            self._clear_plots()
            used_mask = np.zeros(len(records), dtype=bool)
            used_mask[np.flatnonzero(range_mask)] = fit.used_mask
            result_text = self._format_fit_result(
                fit,
                int(np.count_nonzero(used_mask)),
                len(records),
                int(np.count_nonzero(range_mask)),
            )
            combined_figure = self._create_combined_plot(records, fit, used_mask)
            for group in sorted({record.atten for record in records}, key=attenuation_sort_key):
                self._create_group_plot(group, [record for record in records if record.atten == group])
            self._add_plot_history(combined_figure, records, fit, used_mask, range_mask, result_text)
        except Exception as exc:
            self._clear_plots()
            messagebox.showerror("无法绘图", str(exc), parent=self)
            return
        self.result_var.set(result_text)
        self.set_status(f"阈值拟合完成：E_th = {self._format_threshold(fit)}。")

    def _clear_plots(self) -> None:
        self._save_active_history_view()
        for tab_id in self.plot_notebook.tabs():
            try:
                widget = self.nametowidget(tab_id)
                self.plot_notebook.forget(tab_id)
                widget.destroy()
            except tk.TclError:
                pass
        self.plot_canvases.clear()
        self.figures.clear()
        self.axis_points.clear()
        self.plot_preview_context.clear()

    def _plot_dpi(self) -> int:
        try:
            dpi = int(self.dpi_var.get())
        except (tk.TclError, TypeError, ValueError) as exc:
            raise ValueError("保存 DPI 必须是整数。") from exc
        if not 50 <= dpi <= 1200:
            raise ValueError("保存 DPI 应在 50–1200 之间。")
        return dpi

    def _style(self, group: str, index: int = 0) -> dict[str, Any]:
        if group not in self.group_styles:
            self.group_styles[group] = {"color": DEFAULT_COLORS[index % len(DEFAULT_COLORS)], "marker": MARKERS[index % len(MARKERS)]}
        return self.group_styles[group]

    def _create_plot_tab(self, key: str, title: str, figure: Figure) -> FigureCanvasTkAgg:
        frame = ttk.Frame(self.plot_notebook, padding=6)
        toolbar_frame = ttk.Frame(frame)
        toolbar_frame.pack(fill=tk.X)
        canvas_host = ttk.Frame(frame)
        canvas_host.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self._restore_logical_figure_dpi(figure)
        canvas = FigureCanvasTkAgg(figure, master=canvas_host)
        toolbar = RememberingNavigationToolbar(
            canvas, toolbar_frame, path_store=self.path_store,
            path_key="threshold_toolbar_figure_save", pack_toolbar=False,
            dpi_getter=self._plot_dpi,
        )
        toolbar.update()
        toolbar.pack(side=tk.LEFT)
        widget = canvas.get_tk_widget()

        self.plot_preview_context[key] = {
            "host": canvas_host,
            "widget": widget,
            "figure": figure,
            "canvas": canvas,
            "toolbar": toolbar,
            "aspect": float(figure.get_figwidth() / figure.get_figheight()),
        }

        def resize_preview(_event=None) -> None:
            self._resize_plot_preview(key)

        canvas.mpl_connect("button_press_event", self._on_plot_click)
        canvas.draw()
        self.plot_canvases[key] = canvas
        self.figures[key] = figure
        # Size the still-hidden canvas from the already visible notebook before
        # adding the tab.  This avoids the 180 px placeholder flash and makes
        # the very first generated preview start at its final fitted size.
        self._resize_plot_preview(key)
        # TkAgg normally applies the Windows device-pixel ratio only after the
        # widget is mapped.  Doing it while still hidden prevents a one-frame
        # logical-DPI image from appearing before the final high-DPI canvas.
        update_ratio = getattr(canvas, "_update_device_pixel_ratio", None)
        if callable(update_ratio):
            update_ratio()
        widget.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.plot_notebook.add(frame, text=title)
        canvas_host.bind("<Configure>", resize_preview, add="+")
        self.after_idle(lambda plot_key=key: self._resize_plot_preview(plot_key))
        return canvas

    @staticmethod
    def _restore_logical_figure_dpi(figure: Figure) -> None:
        """Prevent Windows display scaling from accumulating on cache switches.

        FigureCanvasBase records the figure's *current* DPI as its original DPI.
        A cached figure may currently have an already device-scaled DPI, so
        constructing another Tk canvas from it would multiply the Windows
        scaling factor again.  Keep one stable logical DPI for every figure.
        """
        logical_dpi = float(getattr(figure, "_xfel_logical_dpi", figure.dpi))
        figure._xfel_logical_dpi = logical_dpi  # type: ignore[attr-defined]
        figure._set_dpi(logical_dpi, forward=False)
        figure._original_dpi = logical_dpi

    @staticmethod
    def _light_color(color: str, amount: float = 0.78) -> str:
        text = color.lstrip("#")
        if len(text) != 6:
            return "#dbe4ee"
        try:
            rgb = [int(text[index:index + 2], 16) for index in (0, 2, 4)]
        except ValueError:
            return "#dbe4ee"
        mixed = [round(channel + (255 - channel) * amount) for channel in rgb]
        return "#" + "".join(f"{channel:02x}" for channel in mixed)

    def _create_combined_plot(self, records: list[PointRecord], fit: ThresholdFit, used_mask: np.ndarray) -> Figure:
        figure = Figure(figsize=(10.2, 7.2), dpi=100)
        figure.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.91)
        axis = figure.add_subplot(111)
        self.axis_points[axis] = []
        groups = sorted({record.atten for record in records}, key=attenuation_sort_key)
        for index, group in enumerate(groups):
            style = self._style(group, index)
            group_pairs = [(record_index, record) for record_index, record in enumerate(records) if record.atten == group]
            group_has_used_points = any(bool(used_mask[record_index]) for record_index, _record in group_pairs)
            for used_value in (False, True):
                pairs = [(i, record) for i, record in group_pairs if bool(used_mask[i]) == used_value]
                if not pairs:
                    continue
                x = np.asarray([record.energy for _, record in pairs], dtype=float)
                y = np.asarray([record.area for _, record in pairs], dtype=float)
                marker = style["marker"]
                # A group that has only excluded points still belongs in the
                # legend, but its label does not call out the exclusion state.
                label = f"{group} 倍" if used_value or not group_has_used_points else "_nolegend_"
                point_color = style["color"] if used_value else self._light_color(style["color"])
                xerr = np.asarray([abs(record.energy * (record.energy_rel_err or 0)) for _, record in pairs]) if self.show_xerr_var.get() else None
                yerr = np.asarray([abs(record.area * (record.area_rel_err or 0)) for _, record in pairs]) if self.show_yerr_var.get() else None
                if xerr is not None or yerr is not None:
                    if xerr is not None:
                        xerr = np.minimum(xerr, x * 0.999)
                    artist = axis.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, linestyle="none", color=point_color,
                                           ecolor=point_color, capsize=2, markersize=math.sqrt(self.marker_size_var.get()), alpha=0.9, label=label)
                    artist.lines[0]._xfel_group = group  # type: ignore[attr-defined]
                    artist.lines[0]._xfel_excluded = not used_value  # type: ignore[attr-defined]
                else:
                    artist = axis.scatter(x, y, s=self.marker_size_var.get(), marker=marker, color=point_color, alpha=0.9, label=label)
                    artist._xfel_group = group  # type: ignore[attr-defined]
                    artist._xfel_excluded = not used_value  # type: ignore[attr-defined]
                self.axis_points[axis].extend((record, float(record.energy), float(record.area)) for _, record in pairs)
        minimum, maximum = float(np.min([record.energy for record in records])), float(np.max([record.energy for record in records]))
        if self.xscale_var.get() == "log":
            xline = np.geomspace(max(minimum * 0.7, np.finfo(float).tiny), maximum * 1.3, 500)
        else:
            xline = np.linspace(max(0, minimum * 0.7), maximum * 1.3, 500)
        fit_label = (
            f"S={fit.a:.4g}ln(E){fit.b:+.4g}; "
            f"E_th={self._format_threshold(fit, value_digits=4, error_digits=3)}"
        )
        fit_line, = axis.plot(
            xline, fit.a * np.log(xline) + fit.b,
            color=self.line_color_var.get(), linestyle=self.line_style_var.get(),
            linewidth=self.line_width_var.get(),
            label=fit_label if self.show_fit_line_var.get() else "_nolegend_",
            visible=self.show_fit_line_var.get(),
        )
        fit_line._xfel_fit_line = True  # type: ignore[attr-defined]
        fit_line._xfel_fit_label = fit_label  # type: ignore[attr-defined]
        self._format_axis(axis, self.title_var.get())
        self._create_plot_tab("combined", "总图 + 阈值拟合", figure)
        return figure

    def _create_group_plot(self, group: str, records: list[PointRecord]) -> None:
        figure = Figure(figsize=(10.2, 7.2), dpi=100)
        figure.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.91)
        axis = figure.add_subplot(111)
        style = self._style(group)
        x = [record.energy for record in records]
        y = [record.area for record in records]
        # Requirement: individual attenuation figures are scatter-only, without fit or error bars.
        axis.scatter(x, y, s=self.marker_size_var.get(), marker=style["marker"], color=style["color"], alpha=0.85, label=f"{group} 倍")
        self.axis_points[axis] = [(record, float(record.energy), float(record.area)) for record in records]
        self._format_axis(axis, f"{group} 倍衰减：损伤面积–单脉冲能量")
        self._create_plot_tab(f"group::{group}", f"{group} 倍散点图", figure)

    def _add_plot_history(
        self,
        figure: Figure,
        records: list[PointRecord],
        fit: ThresholdFit,
        used_mask: np.ndarray,
        range_mask: np.ndarray,
        result_text: str,
    ) -> None:
        history_id = f"plot_{datetime.now().strftime('%H%M%S_%f')}"
        axis = figure.axes[0]
        entry = {
            "id": history_id,
            "created_at": datetime.now().strftime("%H:%M:%S"),
            "figure": figure,
            "axis_points": list(self.axis_points.get(axis, [])),
            "threshold": float(fit.threshold),
            "threshold_err": float(fit.threshold_err),
            "fit": fit,
            "result_text": result_text,
            "total": len(records),
            "used": int(np.count_nonzero(used_mask)),
            "in_range": int(np.count_nonzero(range_mask)),
            "original_view": self._capture_figure_view(figure),
            "current_view": self._capture_figure_view(figure),
            # Individual attenuation plots still do not enter the history
            # cache as Figure objects/cards.  Their source records are enough
            # to recreate the group tabs whenever this combined plot returns.
            "records": list(records),
            "photo": None,
        }
        self._render_history_thumbnail(entry)
        self.plot_history.append(entry)
        self.active_history_id = history_id
        self._seed_toolbar_history("combined", entry["original_view"], entry["current_view"])
        self._refresh_history_sidebar()
        self.history_canvas.after_idle(lambda: self.history_canvas.yview_moveto(1.0))

    def _render_history_thumbnail(self, entry: dict[str, Any]) -> None:
        buffer = BytesIO()
        entry["figure"].savefig(buffer, format="png", dpi=70, facecolor="white")
        buffer.seek(0)
        with Image.open(buffer) as opened:
            image = opened.convert("RGB")
            image.thumbnail((172, 118), Image.Resampling.LANCZOS)
            entry["photo"] = ImageTk.PhotoImage(image)

    def _refresh_history_sidebar(self) -> None:
        for child in self.history_list.winfo_children()[1:]:
            child.destroy()
        for index, entry in enumerate(self.plot_history, 1):
            active = str(entry["id"]) == self.active_history_id
            button = tk.Button(
                self.history_list,
                image=entry["photo"],
                text=(
                    f"{index}. {entry['created_at']}\n"
                    f"E_th={self._format_threshold(entry['fit'], value_digits=4, error_digits=3)}\n"
                    f"{entry['used']}/{entry['total']} 点"
                ),
                compound=tk.TOP,
                justify=tk.LEFT,
                wraplength=178,
                relief=tk.SUNKEN if active else tk.RAISED,
                borderwidth=2 if active else 1,
                background="#dbeafe" if active else "#ffffff",
                command=lambda history_id=str(entry["id"]): self._activate_plot_history(history_id),
            )
            button.pack(fill=tk.X, padx=3, pady=4)

    def _activate_plot_history(self, history_id: str) -> None:
        entry = next((item for item in self.plot_history if str(item["id"]) == history_id), None)
        if entry is None:
            return
        self._clear_plots()
        figure = entry["figure"]
        original_view = entry.setdefault("original_view", self._capture_figure_view(figure))
        current_view = entry.setdefault("current_view", self._capture_figure_view(figure))
        self._restore_figure_view(figure, original_view)
        axis = figure.axes[0]
        self.axis_points[axis] = list(entry["axis_points"])
        self._create_plot_tab("combined", "缓存总图 + 阈值拟合", figure)
        records = list(entry.get("records", []))
        for group in sorted({record.atten for record in records}, key=attenuation_sort_key):
            self._create_group_plot(group, [record for record in records if record.atten == group])
        self._seed_toolbar_history("combined", original_view, current_view)
        self.active_history_id = history_id
        self.last_fit = entry["fit"]
        self.result_var.set(entry["result_text"])
        self._refresh_history_sidebar()
        self.set_status(
            f"第 4 步已切换到 {entry['created_at']} 生成的绘图缓存："
            f"E_th = {self._format_threshold(entry['fit'])}。"
        )

    @staticmethod
    def _capture_figure_view(figure: Figure) -> list[dict[str, Any]]:
        return [
            {
                "xscale": axis.get_xscale(), "yscale": axis.get_yscale(),
                "xlim": tuple(map(float, axis.get_xlim())), "ylim": tuple(map(float, axis.get_ylim())),
            }
            for axis in figure.axes
        ]

    @staticmethod
    def _restore_figure_view(figure: Figure, views: list[dict[str, Any]]) -> None:
        for axis, view in zip(figure.axes, views):
            xscale = str(view.get("xscale", axis.get_xscale()))
            yscale = str(view.get("yscale", axis.get_yscale()))
            # Reapplying an unchanged scale resets custom tick locators and
            # formatters.  Toolbar navigation normally changes only limits.
            if axis.get_xscale() != xscale:
                axis.set_xscale(xscale)
            if axis.get_yscale() != yscale:
                axis.set_yscale(yscale)
            if "xlim" in view:
                axis.set_xlim(*view["xlim"])
            if "ylim" in view:
                axis.set_ylim(*view["ylim"])

    def _save_active_history_view(self) -> None:
        context = self.plot_preview_context.get("combined")
        if context is None or not self.active_history_id:
            return
        entry = next((item for item in self.plot_history if str(item["id"]) == self.active_history_id), None)
        if entry is not None and entry.get("figure") is context.get("figure"):
            entry["current_view"] = self._capture_figure_view(context["figure"])

    def _seed_toolbar_history(self, key: str, original_view: list[dict[str, Any]], current_view: list[dict[str, Any]]) -> None:
        context = self.plot_preview_context.get(key)
        if context is None:
            return
        figure = context["figure"]
        toolbar = context["toolbar"]
        self._restore_figure_view(figure, original_view)
        toolbar._nav_stack.clear()
        toolbar.push_current()
        self._restore_figure_view(figure, current_view)
        if current_view != original_view:
            toolbar.push_current()
        context["canvas"].draw_idle()

    def clear_plot_history(self) -> None:
        self._clear_plots()
        self.plot_history.clear()
        self.active_history_id = ""
        self._refresh_history_sidebar()

    def _validate_axis_settings(self) -> None:
        positive = {
            "图标题字号": self.title_font_size_var, "X 轴标题字号": self.xlabel_font_size_var,
            "Y 轴标题字号": self.ylabel_font_size_var, "X 刻度字号": self.xtick_font_size_var,
            "Y 刻度字号": self.ytick_font_size_var, "图例字号": self.legend_font_size_var,
            "主刻度长度": self.major_tick_length_var, "主刻度线宽": self.major_tick_width_var,
            "次刻度长度": self.minor_tick_length_var, "次刻度线宽": self.minor_tick_width_var,
            "坐标轴边框线宽": self.axis_line_width_var, "网格线宽": self.grid_line_width_var,
            "散点大小": self.marker_size_var, "拟合线宽": self.line_width_var,
        }
        for label, variable in positive.items():
            if float(variable.get()) <= 0:
                raise ValueError(f"{label}必须大于 0。")
        for label, variable in (("X 对数底数", self.x_log_base_var), ("Y 对数底数", self.y_log_base_var)):
            if float(variable.get()) <= 1:
                raise ValueError(f"{label}必须大于 1。")
        for label, variable in (("X 次刻度分段数", self.x_minor_divisions_var), ("Y 次刻度分段数", self.y_minor_divisions_var)):
            if int(variable.get()) < 2:
                raise ValueError(f"{label}至少为 2。")
        for label, variable in (("X 主刻度间隔", self.x_major_step_var), ("Y 主刻度间隔", self.y_major_step_var)):
            if str(variable.get()).strip() and float(variable.get()) <= 0:
                raise ValueError(f"{label}必须大于 0，或留空使用自动刻度。")
        if not 0 <= float(self.grid_alpha_var.get()) <= 1:
            raise ValueError("网格透明度必须在 0–1 之间。")
        if int(self.scientific_min_var.get()) > int(self.scientific_max_var.get()):
            raise ValueError("科学计数下限指数不能大于上限指数。")
        colors = {
            "坐标轴文字颜色": self.axis_text_color_var.get(), "刻度颜色": self.tick_color_var.get(),
            "坐标轴边框颜色": self.spine_color_var.get(), "网格颜色": self.grid_color_var.get(),
            "拟合线颜色": self.line_color_var.get(),
        }
        colors.update({f"{group} 倍散点颜色": style.get("color", "") for group, style in self.group_styles.items()})
        for label, color in colors.items():
            if not is_color_like(color):
                raise ValueError(f"{label}“{color}”不是有效颜色。")
        for minimum, maximum, label in (
            (self.xmin_var, self.xmax_var, "X"), (self.ymin_var, self.ymax_var, "Y"),
        ):
            low, high = self._optional_float(minimum.get()), self._optional_float(maximum.get())
            if low is not None and high is not None and low >= high:
                raise ValueError(f"{label} 轴最小值必须小于最大值。")

    def _configure_tick_locator(
        self,
        axis_object,
        scale: str,
        major_step_text: str,
        minor_divisions: int,
        log_base: float,
        number_format: str,
    ) -> None:
        if scale == "log":
            axis_object.set_major_locator(mticker.LogLocator(base=log_base))
            if self.show_minor_ticks_var.get():
                integer_base = max(2, int(round(log_base)))
                subs = np.arange(2, integer_base, dtype=float) if integer_base > 2 else "auto"
                axis_object.set_minor_locator(mticker.LogLocator(base=log_base, subs=subs))
            else:
                axis_object.set_minor_locator(mticker.NullLocator())
            axis_object.set_major_formatter(mticker.LogFormatterSciNotation(base=log_base))
            return

        if str(major_step_text).strip():
            axis_object.set_major_locator(mticker.MultipleLocator(float(major_step_text)))
        else:
            axis_object.set_major_locator(mticker.AutoLocator())
        if self.show_minor_ticks_var.get():
            axis_object.set_minor_locator(mticker.AutoMinorLocator(int(minor_divisions)))
        else:
            axis_object.set_minor_locator(mticker.NullLocator())
        formatter = mticker.ScalarFormatter(useMathText=number_format == "科学计数")
        if number_format == "普通数字":
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
        elif number_format == "科学计数":
            formatter.set_scientific(True)
            formatter.set_powerlimits((int(self.scientific_min_var.get()), int(self.scientific_max_var.get())))
        axis_object.set_major_formatter(formatter)

    def _update_axis_legend(self, axis) -> None:
        legend = axis.get_legend()
        if self.show_legend_var.get():
            axis.legend(fontsize=float(self.legend_font_size_var.get()))
        elif legend is not None:
            legend.remove()

    def _apply_axis_settings(self, axis, title: str, *, manage_legend: bool = True) -> None:
        self._validate_axis_settings()
        xscale, yscale = self.xscale_var.get(), self.yscale_var.get()
        axis.set_xscale(xscale, base=float(self.x_log_base_var.get())) if xscale == "log" else axis.set_xscale("linear")
        axis.set_yscale(yscale, base=float(self.y_log_base_var.get())) if yscale == "log" else axis.set_yscale("linear")
        axis.set_title(
            title, fontsize=float(self.title_font_size_var.get()), color=self.axis_text_color_var.get(),
            pad=float(self.title_pad_var.get()),
        )
        axis.set_xlabel(
            self.xlabel_var.get(), fontsize=float(self.xlabel_font_size_var.get()),
            color=self.axis_text_color_var.get(), labelpad=float(self.xlabel_pad_var.get()),
        )
        axis.set_ylabel(
            self.ylabel_var.get(), fontsize=float(self.ylabel_font_size_var.get()),
            color=self.axis_text_color_var.get(), labelpad=float(self.ylabel_pad_var.get()),
        )
        xmin, xmax = self._optional_float(self.xmin_var.get()), self._optional_float(self.xmax_var.get())
        ymin, ymax = self._optional_float(self.ymin_var.get()), self._optional_float(self.ymax_var.get())
        if xscale == "log":
            xmin = None if xmin is not None and xmin <= 0 else xmin
            xmax = None if xmax is not None and xmax <= 0 else xmax
        if yscale == "log":
            ymin = None if ymin is not None and ymin <= 0 else ymin
            ymax = None if ymax is not None and ymax <= 0 else ymax
        if xmin is not None or xmax is not None:
            current = axis.get_xlim()
            axis.set_xlim(xmin if xmin is not None else current[0], xmax if xmax is not None else current[1])
        if ymin is not None or ymax is not None:
            current = axis.get_ylim()
            axis.set_ylim(ymin if ymin is not None else current[0], ymax if ymax is not None else current[1])

        self._configure_tick_locator(
            axis.xaxis, xscale, self.x_major_step_var.get(), int(self.x_minor_divisions_var.get()),
            float(self.x_log_base_var.get()), self.x_tick_format_var.get(),
        )
        self._configure_tick_locator(
            axis.yaxis, yscale, self.y_major_step_var.get(), int(self.y_minor_divisions_var.get()),
            float(self.y_log_base_var.get()), self.y_tick_format_var.get(),
        )
        common = {
            "direction": self.tick_direction_var.get(), "color": self.tick_color_var.get(),
            "labelcolor": self.tick_color_var.get(),
        }
        axis.tick_params(
            axis="x", which="major", length=float(self.major_tick_length_var.get()),
            width=float(self.major_tick_width_var.get()), labelsize=float(self.xtick_font_size_var.get()),
            labelrotation=float(self.xtick_rotation_var.get()), top=self.show_top_ticks_var.get(),
            bottom=True, labeltop=self.show_top_labels_var.get(), labelbottom=True, **common,
        )
        axis.tick_params(
            axis="y", which="major", length=float(self.major_tick_length_var.get()),
            width=float(self.major_tick_width_var.get()), labelsize=float(self.ytick_font_size_var.get()),
            labelrotation=float(self.ytick_rotation_var.get()), right=self.show_right_ticks_var.get(),
            left=True, labelright=self.show_right_labels_var.get(), labelleft=True, **common,
        )
        axis.tick_params(
            axis="both", which="minor", length=float(self.minor_tick_length_var.get()),
            width=float(self.minor_tick_width_var.get()), top=self.show_top_ticks_var.get(),
            right=self.show_right_ticks_var.get(), bottom=True, left=True,
            direction=self.tick_direction_var.get(), color=self.tick_color_var.get(),
        )
        for spine in axis.spines.values():
            spine.set_linewidth(float(self.axis_line_width_var.get()))
            spine.set_color(self.spine_color_var.get())
        axis.grid(False, which="both")
        if self.show_grid_var.get():
            axis.grid(
                True, which="both" if self.grid_which_var.get() == "主+次刻度" else "major",
                linestyle=self.grid_line_style_var.get(), color=self.grid_color_var.get(),
                linewidth=float(self.grid_line_width_var.get()), alpha=float(self.grid_alpha_var.get()),
            )
        if manage_legend:
            self._update_axis_legend(axis)

    def apply_plot_settings(self) -> None:
        selected = self.plot_notebook.select()
        if not selected:
            messagebox.showinfo("没有预览图", "请先绘图或选择右侧绘图缓存。", parent=self)
            return
        index = self.plot_notebook.index(selected)
        key = list(self.figures)[index]
        figure = self.figures[key]
        axis = figure.axes[0]
        try:
            marker_size = float(self.marker_size_var.get())
            target_title = self.title_var.get() if key == "combined" else axis.get_title()
            self._apply_axis_settings(axis, target_title, manage_legend=False)
        except Exception as exc:
            messagebox.showerror("图形设置无效", str(exc), parent=self)
            return

        for collection in axis.collections:
            label = str(collection.get_label())
            group = getattr(collection, "_xfel_group", None)
            if group is None:
                group = next((candidate for candidate in self.group_styles if label.startswith(f"{candidate} 倍")), None)
            if group is None:
                continue
            style = self.group_styles[group]
            color = self._light_color(style["color"]) if bool(getattr(collection, "_xfel_excluded", False)) else style["color"]
            if hasattr(collection, "set_sizes"):
                collection.set_sizes([marker_size])
            if hasattr(collection, "set_facecolor"):
                collection.set_facecolor(color)
            if hasattr(collection, "set_edgecolor"):
                collection.set_edgecolor(color)
            if hasattr(collection, "set_paths"):
                marker = MarkerStyle(style["marker"])
                collection.set_paths([marker.get_path().transformed(marker.get_transform())])
        for line in axis.lines:
            label = str(line.get_label())
            group = getattr(line, "_xfel_group", None)
            if group in self.group_styles:
                style = self.group_styles[group]
                color = self._light_color(style["color"]) if bool(getattr(line, "_xfel_excluded", False)) else style["color"]
                line.set_color(color)
                line.set_marker(style["marker"])
                line.set_markersize(math.sqrt(marker_size))
            elif bool(getattr(line, "_xfel_fit_line", False)) or label.startswith("S="):
                line.set_color(self.line_color_var.get())
                line.set_linestyle(self.line_style_var.get())
                line.set_linewidth(float(self.line_width_var.get()))
                visible = bool(self.show_fit_line_var.get())
                fit_label = str(getattr(line, "_xfel_fit_label", label))
                line._xfel_fit_line = True  # type: ignore[attr-defined]
                line._xfel_fit_label = fit_label  # type: ignore[attr-defined]
                line.set_visible(visible)
                line.set_label(fit_label if visible else "_nolegend_")
        self._update_axis_legend(axis)
        figure.canvas.draw_idle()
        view = self._capture_figure_view(figure)
        self._seed_toolbar_history(key, view, view)
        entry = next((item for item in self.plot_history if item.get("figure") is figure), None)
        if entry is not None:
            entry["original_view"] = view
            entry["current_view"] = view
            self._render_history_thumbnail(entry)
            self._refresh_history_sidebar()
        self.set_status("第 4 步图形设置已应用到当前预览；未生成新的绘图缓存。")

    def _format_axis(self, axis, title: str) -> None:
        self._apply_axis_settings(axis, title)

    def _resize_plot_preview(self, key: str) -> None:
        context = self.plot_preview_context.get(key)
        if context is None:
            return
        host = context["host"]
        widget = context["widget"]
        figure = context["figure"]
        canvas = context["canvas"]
        aspect = max(0.1, float(context["aspect"]))
        host_width, host_height = host.winfo_width(), host.winfo_height()
        if host_width <= 100 or host_height <= 100:
            # New and non-selected tabs are not mapped yet.  The notebook is
            # already laid out, so it provides a stable pre-display size.
            available_width = max(180, self.plot_notebook.winfo_width() - 24)
            available_height = max(150, self.plot_notebook.winfo_height() - 88)
        else:
            available_width = max(180, host_width - 12)
            available_height = max(150, host_height - 12)
        base_width = max(180, min(available_width, 1170))
        base_height = max(150, min(available_height, 810))
        if base_width / base_height > aspect:
            fitted_width = round(base_height * aspect)
            fitted_height = base_height
        else:
            fitted_width = base_width
            fitted_height = round(base_width / aspect)
        target_width = fitted_width
        target_height = fitted_height
        widget.configure(width=max(180, target_width), height=max(150, target_height))
        figure.set_size_inches(max(180, target_width) / figure.dpi, max(150, target_height) / figure.dpi, forward=False)
        canvas.draw_idle()

    def _nearest_record(self, event) -> PointRecord | None:
        if event.inaxes is None or event.x is None or event.y is None:
            return None
        candidates = self.axis_points.get(event.inaxes, [])
        nearest: tuple[float, PointRecord] | None = None
        for record, x, y in candidates:
            display = event.inaxes.transData.transform((x, y))
            distance = math.hypot(display[0] - event.x, display[1] - event.y)
            if nearest is None or distance < nearest[0]:
                nearest = distance, record
        return nearest[1] if nearest and nearest[0] <= 14 else None

    def _on_plot_click(self, event) -> None:
        if event.button != 3:
            return
        record = self._nearest_record(event)
        if record is None:
            return
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="查看该点全部信息与徕卡轮廓图", command=lambda: self.show_point_detail(record.uid))
        menu.add_command(label="移入删除点回收站", command=lambda: self.delete_point(record.uid))
        menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())

    def delete_point(self, uid: str) -> None:
        self.state.delete_point(uid)
        self.set_status(f"点 {uid} 已移入回收站，可随时还原。")

    def show_point_detail(self, uid: str) -> None:
        record = self.state.records[uid]
        window = tk.Toplevel(self)
        window.title(f"点信息：{record.atten} 倍 no.{record.serial}")
        window.update_idletasks()
        detail_width = min(1040, max(760, round(window.winfo_screenwidth() * 0.84)))
        detail_height = min(720, max(560, round(window.winfo_screenheight() * 0.80)))
        window.geometry(f"{detail_width}x{detail_height}")
        body = ttk.Panedwindow(window, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=0)
        body.add(right, weight=1)
        tree = ttk.Treeview(left, columns=("value",), show="tree headings", height=25)
        tree.heading("#0", text="字段")
        tree.heading("value", text="值")
        tree.column("#0", width=135)
        tree.column("value", width=245)
        tree.pack(fill=tk.BOTH, expand=True)
        values = {
            "衰减倍数": record.atten, "序号": record.serial, "GMD (A)": record.gmd, "PD (V·s)": record.pd,
            "单脉冲能量 (nJ)": record.energy, "损伤面积": record.area, "面积单位": record.area_unit,
            "能量相对误差": record.energy_rel_err, "面积相对误差": record.area_rel_err,
            "面积识别状态": record.area_status, "像素面积": record.area_px, "GMD HDF5": record.source_hdf5,
            "GMD 偏移量": record.gmd_shift, "徕卡图像": record.image_path, "删除状态": record.deleted,
        }
        for key, value in values.items():
            tree.insert("", tk.END, text=key, values=("" if value is None else value,))
        image_canvas = tk.Canvas(right, bg="#111827", highlightthickness=0)
        image_canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            right,
            text="显示该点所在的完整徕卡图；目标点用第 1 步设置的高亮颜色标出。",
            foreground="#475569",
        ).pack(anchor=tk.W, pady=(5, 0))

        source_image: Image.Image | None = None
        path = Path(record.image_path) if record.image_path else None
        image_error = ""
        if path is None or not path.exists():
            image_error = "原始徕卡图像文件不存在或尚未关联。"
        else:
            try:
                with Image.open(path) as opened:
                    source_image = opened.convert("RGB")
            except Exception as exc:
                image_error = f"无法读取图像：{exc}"

        image_records: list[PointRecord] = []
        if path is not None:
            target_path = self._normal_path(path)
            image_records = [
                item for item in self.state.records.values()
                if item.image_path and self._normal_path(Path(item.image_path)) == target_path
            ]
        if record not in image_records:
            image_records.append(record)

        render_job: str | None = None

        def render(_event=None) -> None:
            nonlocal render_job
            if render_job is not None:
                try:
                    window.after_cancel(render_job)
                except tk.TclError:
                    pass
            render_job = window.after(
                45,
                lambda: self._render_detail_image(image_canvas, record, source_image, image_records, image_error),
            )

        image_canvas.bind("<Configure>", render)
        render()

    @staticmethod
    def _normal_path(path: Path) -> str:
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    @staticmethod
    def _style_color(style: dict[str, Any], key: str, fallback: str) -> str:
        value = str(style.get(key, fallback))
        return value if value else fallback

    def _render_detail_image(
        self,
        canvas: tk.Canvas,
        record: PointRecord,
        source_image: Image.Image | None,
        image_records: list[PointRecord],
        image_error: str = "",
    ) -> None:
        canvas.delete("all")
        if source_image is None:
            canvas.create_text(20, 20, anchor=tk.NW, fill="white", text=image_error or "原始徕卡图像不可用。")
            return

        canvas_width = max(100, canvas.winfo_width())
        canvas_height = max(100, canvas.winfo_height())
        scale = min((canvas_width - 20) / source_image.width, (canvas_height - 20) / source_image.height)
        scale = max(scale, 0.01)
        display_width = max(1, round(source_image.width * scale))
        display_height = max(1, round(source_image.height * scale))
        image = source_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self._detail_photo = ImageTk.PhotoImage(image)
        canvas._detail_photo = self._detail_photo  # type: ignore[attr-defined]
        origin_x = (canvas_width - display_width) / 2
        origin_y = (canvas_height - display_height) / 2
        canvas.create_image(origin_x, origin_y, image=canvas._detail_photo, anchor=tk.NW)  # type: ignore[attr-defined]

        style = self.state.area_display_style
        auto_color = self._style_color(style, "auto_contour_color", "#22c55e")
        manual_color = self._style_color(style, "manual_contour_color", "#38bdf8")
        selected_color = self._style_color(style, "selected_color", "#f59e0b")
        label_color = self._style_color(style, "label_color", "#22c55e")
        grid_color = self._style_color(style, "grid_color", "#64748b")
        contour_width = max(1, round(float(style.get("auto_contour_width", 2))))
        manual_width = max(1, round(float(style.get("manual_contour_width", 2))))
        selected_width = max(1, round(float(style.get("selected_width", 4))))
        grid_width = max(1, round(float(style.get("grid_width", 1))))
        label_font_size = max(7, round(float(style.get("label_font_size", 11))))

        rows = max([item.image_display_rows or 0 for item in image_records] + [1])
        cols = max([item.image_display_cols or 0 for item in image_records] + [1])
        cell_width = display_width / cols
        cell_height = display_height / rows

        for column in range(1, cols):
            x = origin_x + column * cell_width
            canvas.create_line(x, origin_y, x, origin_y + display_height, fill=grid_color, width=grid_width, dash=(2, 3))
        for row in range(1, rows):
            y = origin_y + row * cell_height
            canvas.create_line(origin_x, y, origin_x + display_width, y, fill=grid_color, width=grid_width, dash=(2, 3))

        for item in image_records:
            selected = item.uid == record.uid
            manual = item.area_status in {"手动画", "手动轮廓"}
            color = selected_color if selected else (manual_color if manual else auto_color)
            width = selected_width if selected else (manual_width if manual else contour_width)
            contour = item.contour or []
            if len(contour) >= 2:
                points = [(origin_x + float(x) * scale, origin_y + float(y) * scale) for x, y in contour]
                flat_points = [coordinate for point in (points + [points[0]]) for coordinate in point]
                canvas.create_line(*flat_points, fill=color, width=width, joinstyle=tk.ROUND, capstyle=tk.ROUND)

            display_row = int(item.spot_display_row) - 1 if item.spot_display_row is not None else None
            display_col = int(item.spot_display_col) - 1 if item.spot_display_col is not None else None
            if (display_row is None or display_col is None) and contour:
                center_x = sum(float(point[0]) for point in contour) / len(contour)
                center_y = sum(float(point[1]) for point in contour) / len(contour)
                display_col = min(cols - 1, max(0, int(center_x / (source_image.width / cols))))
                display_row = min(rows - 1, max(0, int(center_y / (source_image.height / rows))))
            if display_row is not None and display_col is not None and 0 <= display_row < rows and 0 <= display_col < cols:
                left = origin_x + display_col * cell_width
                top = origin_y + display_row * cell_height
                label = f"{item.atten}倍 no.{item.serial}"
                if item.area is not None:
                    label += f"\n{item.area:.5g} {item.area_unit}"
                canvas.create_text(
                    left + 5,
                    top + 4,
                    anchor=tk.NW,
                    text=label,
                    fill=selected_color if selected else label_color,
                    font=("Microsoft YaHei UI", label_font_size, "bold"),
                )
                if selected:
                    canvas.create_rectangle(
                        left + 1,
                        top + 1,
                        left + cell_width - 1,
                        top + cell_height - 1,
                        outline=selected_color,
                        width=selected_width,
                        dash=(8, 4),
                    )

        scale_line = style.get("scale_line")
        if scale_line and len(scale_line) == 2:
            try:
                (x1, y1), (x2, y2) = scale_line
                canvas.create_line(
                    origin_x + float(x1) * scale,
                    origin_y + float(y1) * scale,
                    origin_x + float(x2) * scale,
                    origin_y + float(y2) * scale,
                    fill=self._style_color(style, "scale_line_color", "#ffffff"),
                    width=max(1, round(float(style.get("scale_line_width", 3)))),
                )
            except (TypeError, ValueError):
                pass

    def open_recycle_bin(self) -> None:
        window = tk.Toplevel(self)
        window.title("删除点回收站")
        window.geometry("760x470")
        tree = ttk.Treeview(window, columns=("atten", "serial", "energy", "area", "deleted_at"), show="headings", selectmode="extended")
        for column, title, width in (("atten", "衰减", 80), ("serial", "序号", 80), ("energy", "能量", 130), ("area", "面积", 130), ("deleted_at", "删除时间", 190)):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor=tk.CENTER)
        tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def refresh() -> None:
            tree.delete(*tree.get_children())
            for record in sorted((item for item in self.state.records.values() if item.deleted), key=lambda item: (attenuation_sort_key(item.atten), item.serial)):
                tree.insert("", tk.END, iid=record.uid, values=(record.atten, record.serial, record.energy, record.area, record.deleted_at))

        def restore_selected() -> None:
            for uid in tree.selection():
                self.state.restore_point(uid)
            refresh()

        def restore_all() -> None:
            for record in self.state.records.values():
                if record.deleted:
                    self.state.restore_point(record.uid)
            refresh()

        buttons = ttk.Frame(window, padding=(8, 0, 8, 8))
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="还原选中", command=restore_selected).pack(side=tk.LEFT)
        ttk.Button(buttons, text="全部还原", command=restore_all).pack(side=tk.LEFT, padx=5)
        refresh()

    def open_style_dialog(self) -> None:
        groups = self.state.groups()
        if not groups:
            return
        window = tk.Toplevel(self)
        window.title("逐衰减组图形样式")
        window.geometry("520x420")
        body = ttk.Frame(window, padding=10)
        body.pack(fill=tk.BOTH, expand=True)
        for row, group in enumerate(groups):
            style = self._style(group, row)
            ttk.Label(body, text=f"{group} 倍", width=14).grid(row=row, column=0, sticky=tk.W, pady=4)
            color_var = tk.StringVar(value=style["color"])
            marker_var = tk.StringVar(value=style["marker"])
            entry = ttk.Entry(body, textvariable=color_var, width=12)
            entry.grid(row=row, column=1, padx=4)

            def choose(g=group, variable=color_var) -> None:
                _rgb, color = colorchooser.askcolor(color=variable.get(), parent=window)
                if color:
                    variable.set(color)
                    self.group_styles.setdefault(g, {})["color"] = color

            ttk.Button(body, text="选颜色", command=choose).grid(row=row, column=2, padx=4)
            combo = ttk.Combobox(body, textvariable=marker_var, values=MARKERS, state="readonly", width=7)
            combo.grid(row=row, column=3, padx=4)
            color_var.trace_add("write", lambda *_args, g=group, v=color_var: self.group_styles.setdefault(g, {}).__setitem__("color", v.get()))
            marker_var.trace_add("write", lambda *_args, g=group, v=marker_var: self.group_styles.setdefault(g, {}).__setitem__("marker", v.get()))
        ttk.Button(body, text="应用并重绘", command=lambda: (window.destroy(), self.generate_plots())).grid(row=len(groups), column=0, columnspan=4, pady=12)

    def save_current_figure(self) -> None:
        selected = self.plot_notebook.select()
        if not selected:
            return
        try:
            dpi = self._plot_dpi()
        except ValueError as exc:
            messagebox.showerror("DPI 无效", str(exc), parent=self)
            return
        index = self.plot_notebook.index(selected)
        key = list(self.figures)[index]
        initial_dir = self.path_store.initial_dir("threshold_figure_save", Path.cwd()) if self.path_store else str(Path.cwd())
        filename = filedialog.asksaveasfilename(parent=self, initialdir=initial_dir, title="保存当前图", defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")], initialfile=f"XFEL_{key.replace('::', '_')}.png")
        if filename:
            if self.path_store:
                self.path_store.remember_file("threshold_figure_save", filename)
            # Preserve the margins chosen with "Configure subplots" exactly.
            self.figures[key].savefig(filename, dpi=dpi)
            self.set_status(f"图已保存：{filename}")

    def export_config(self) -> dict[str, Any]:
        return {
            "title": self.title_var.get(), "xlabel": self.xlabel_var.get(), "ylabel": self.ylabel_var.get(),
            "xscale": self.xscale_var.get(), "yscale": self.yscale_var.get(), "xmin": self.xmin_var.get(), "xmax": self.xmax_var.get(),
            "ymin": self.ymin_var.get(), "ymax": self.ymax_var.get(), "marker_size": self.marker_size_var.get(),
            "dpi": self.dpi_var.get(),
            "title_font_size": self.title_font_size_var.get(), "xlabel_font_size": self.xlabel_font_size_var.get(),
            "ylabel_font_size": self.ylabel_font_size_var.get(), "xtick_font_size": self.xtick_font_size_var.get(),
            "ytick_font_size": self.ytick_font_size_var.get(), "legend_font_size": self.legend_font_size_var.get(),
            "title_pad": self.title_pad_var.get(), "xlabel_pad": self.xlabel_pad_var.get(), "ylabel_pad": self.ylabel_pad_var.get(),
            "x_major_step": self.x_major_step_var.get(), "y_major_step": self.y_major_step_var.get(),
            "x_minor_divisions": self.x_minor_divisions_var.get(), "y_minor_divisions": self.y_minor_divisions_var.get(),
            "show_minor_ticks": self.show_minor_ticks_var.get(), "tick_direction": self.tick_direction_var.get(),
            "major_tick_length": self.major_tick_length_var.get(), "major_tick_width": self.major_tick_width_var.get(),
            "minor_tick_length": self.minor_tick_length_var.get(), "minor_tick_width": self.minor_tick_width_var.get(),
            "xtick_rotation": self.xtick_rotation_var.get(), "ytick_rotation": self.ytick_rotation_var.get(),
            "show_top_ticks": self.show_top_ticks_var.get(), "show_right_ticks": self.show_right_ticks_var.get(),
            "show_top_labels": self.show_top_labels_var.get(), "show_right_labels": self.show_right_labels_var.get(),
            "x_tick_format": self.x_tick_format_var.get(), "y_tick_format": self.y_tick_format_var.get(),
            "x_log_base": self.x_log_base_var.get(), "y_log_base": self.y_log_base_var.get(),
            "scientific_min": self.scientific_min_var.get(), "scientific_max": self.scientific_max_var.get(),
            "axis_line_width": self.axis_line_width_var.get(), "axis_text_color": self.axis_text_color_var.get(),
            "tick_color": self.tick_color_var.get(), "spine_color": self.spine_color_var.get(), "grid_color": self.grid_color_var.get(),
            "grid_which": self.grid_which_var.get(), "grid_line_style": self.grid_line_style_var.get(),
            "grid_line_width": self.grid_line_width_var.get(), "grid_alpha": self.grid_alpha_var.get(),
            "line_color": self.line_color_var.get(), "line_style": self.line_style_var.get(), "line_width": self.line_width_var.get(),
            "show_grid": self.show_grid_var.get(), "show_legend": self.show_legend_var.get(),
            "show_fit_line": self.show_fit_line_var.get(),
            "show_xerr": self.show_xerr_var.get(), "show_yerr": self.show_yerr_var.get(),
            "range_filter": self.range_filter_var.get(), "fit_emin": self.fit_emin_var.get(), "fit_emax": self.fit_emax_var.get(),
            "fit_smin": self.fit_smin_var.get(), "fit_smax": self.fit_smax_var.get(),
            "attenuation_filter": self.attenuation_filter_var.get(),
            "fit_attenuations": sorted(self.selected_fit_attenuations, key=attenuation_sort_key),
            "sigma_clip": self.sigma_clip_var.get(), "sigma_n": self.sigma_n_var.get(), "sigma_iter": self.sigma_iter_var.get(),
            "group_styles": self.group_styles,
        }

    def import_config(self, data: dict[str, Any]) -> None:
        self.clear_plot_history()
        mapping = {
            "title": self.title_var, "xlabel": self.xlabel_var, "ylabel": self.ylabel_var,
            "xscale": self.xscale_var, "yscale": self.yscale_var,
            "xmin": self.xmin_var, "xmax": self.xmax_var, "ymin": self.ymin_var, "ymax": self.ymax_var,
            "marker_size": self.marker_size_var, "dpi": self.dpi_var,
            "title_font_size": self.title_font_size_var, "xlabel_font_size": self.xlabel_font_size_var,
            "ylabel_font_size": self.ylabel_font_size_var, "xtick_font_size": self.xtick_font_size_var,
            "ytick_font_size": self.ytick_font_size_var, "legend_font_size": self.legend_font_size_var,
            "title_pad": self.title_pad_var, "xlabel_pad": self.xlabel_pad_var, "ylabel_pad": self.ylabel_pad_var,
            "x_major_step": self.x_major_step_var, "y_major_step": self.y_major_step_var,
            "x_minor_divisions": self.x_minor_divisions_var, "y_minor_divisions": self.y_minor_divisions_var,
            "show_minor_ticks": self.show_minor_ticks_var, "tick_direction": self.tick_direction_var,
            "major_tick_length": self.major_tick_length_var, "major_tick_width": self.major_tick_width_var,
            "minor_tick_length": self.minor_tick_length_var, "minor_tick_width": self.minor_tick_width_var,
            "xtick_rotation": self.xtick_rotation_var, "ytick_rotation": self.ytick_rotation_var,
            "show_top_ticks": self.show_top_ticks_var, "show_right_ticks": self.show_right_ticks_var,
            "show_top_labels": self.show_top_labels_var, "show_right_labels": self.show_right_labels_var,
            "x_tick_format": self.x_tick_format_var, "y_tick_format": self.y_tick_format_var,
            "x_log_base": self.x_log_base_var, "y_log_base": self.y_log_base_var,
            "scientific_min": self.scientific_min_var, "scientific_max": self.scientific_max_var,
            "axis_line_width": self.axis_line_width_var, "axis_text_color": self.axis_text_color_var,
            "tick_color": self.tick_color_var, "spine_color": self.spine_color_var, "grid_color": self.grid_color_var,
            "grid_which": self.grid_which_var, "grid_line_style": self.grid_line_style_var,
            "grid_line_width": self.grid_line_width_var, "grid_alpha": self.grid_alpha_var,
            "line_color": self.line_color_var, "line_style": self.line_style_var,
            "line_width": self.line_width_var, "show_grid": self.show_grid_var, "show_legend": self.show_legend_var,
            "show_fit_line": self.show_fit_line_var,
            "show_xerr": self.show_xerr_var, "show_yerr": self.show_yerr_var, "range_filter": self.range_filter_var,
            "attenuation_filter": self.attenuation_filter_var,
            "fit_emin": self.fit_emin_var, "fit_emax": self.fit_emax_var, "fit_smin": self.fit_smin_var, "fit_smax": self.fit_smax_var,
            "sigma_clip": self.sigma_clip_var, "sigma_n": self.sigma_n_var, "sigma_iter": self.sigma_iter_var,
        }
        for key, variable in mapping.items():
            if key in data:
                variable.set(data[key])
        if "show_fit_line" not in data:
            self.show_fit_line_var.set(True)
        if "attenuation_filter" not in data:
            self.attenuation_filter_var.set(False)
        self.selected_fit_attenuations = {str(group) for group in data.get("fit_attenuations", [])}
        self._update_attenuation_filter_summary()
        self.group_styles = {str(group): dict(style) for group, style in data.get("group_styles", {}).items()}
