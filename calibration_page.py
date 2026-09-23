"""Step 2: GMD-PD calibration cache and recursive PD-minimum inspection."""

from __future__ import annotations

import math
import copy
import re
import threading
import queue
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import h5py
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from tksheet import Sheet

from remembering_toolbar import RememberingNavigationToolbar
from xfel_core import PD_KEY, ProjectState, RecentPathStore, fit_gmd_pd, load_json, parse_offsets


def read_pd_minimum(path: str | Path) -> tuple[float, int]:
    """Return the minimum physical PD value (-mso:Area) and finite sample count."""
    with h5py.File(path, "r") as handle:
        if PD_KEY not in handle:
            raise KeyError(f"缺少数据集 {PD_KEY}")
        raw = np.asarray(handle[PD_KEY], dtype=float).reshape(-1)
    finite = (-raw)[np.isfinite(raw)]
    if not len(finite):
        raise ValueError("没有有限数值")
    return float(np.min(finite)), int(len(finite))


def read_physical_pd(path: str | Path) -> np.ndarray:
    """Read finite physical PD values (-mso:Area) from a calibration file."""
    with h5py.File(path, "r") as handle:
        if PD_KEY not in handle:
            raise KeyError(f"缺少数据集 {PD_KEY}")
        raw = np.asarray(handle[PD_KEY], dtype=float).reshape(-1)
    values = (-raw)[np.isfinite(raw)]
    if not len(values):
        raise ValueError("没有有限 PD 数值")
    return values


class GMDPDCalibrationPage(ttk.Frame):
    """Independent calibration workflow shared with the energy-formula page."""

    def __init__(self, master: tk.Misc, state: ProjectState, status_callback=None, path_store: RecentPathStore | None = None) -> None:
        super().__init__(master)
        self.state = state
        self.status_callback = status_callback or (lambda text: None)
        self.path_store = path_store
        self.calibration_cache: list[dict[str, Any]] = []
        self.selected_calibration_id = ""
        self.calibration_folder_var = tk.StringVar()
        self.single_calibration_file_var = tk.StringVar()
        self.offsets_var = tk.StringVar(value="0, ±1, ±2")
        self.fit_result_var = tk.StringVar(value="尚未拟合 GMD–PD。")
        self.fit_xmin_var = tk.StringVar()
        self.fit_xmax_var = tk.StringVar()
        self.fit_ymin_var = tk.StringVar()
        self.fit_ymax_var = tk.StringVar()
        self.pd_baseline_var = tk.StringVar(value="0")
        self.pd_baseline_sigma_var = tk.StringVar(value="0")
        self.pd_background_file_var = tk.StringVar(value="请先在上方缓存区选择一个文件")
        self.pd_background_result_var = tk.StringVar(value="尚未计算 PD 背底。")
        self.last_pd_background: dict[str, Any] = {}
        self.gas_attenuation_data: list[list[Any]] = [["", ""] for _ in range(12)]
        self.gas_attenuation_fit: dict[str, Any] = {}
        self.gas_attenuation_result_var = tk.StringVar(value="尚未拟合气体衰减常数。")
        self.gas_attenuation_window: tk.Toplevel | None = None
        self.normalized_window: tk.Toplevel | None = None
        self.pd_background_window: tk.Toplevel | None = None
        self._cache_sort_column = "file"
        self._cache_sort_reverse = False
        self._fit_running = False
        self._fit_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._preserved_fit_view: tuple[str, int, tuple[float, float], tuple[float, float]] | None = None
        self._build_ui()

    def set_status(self, text: str) -> None:
        self.status_callback(text)

    def _build_ui(self) -> None:
        self.page_canvas = tk.Canvas(self, highlightthickness=0)
        page_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.page_canvas.yview)
        self.page_canvas.configure(yscrollcommand=page_scroll.set)
        self.page_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        page_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        fit_page = ttk.Frame(self.page_canvas, padding=7)
        self._page_window = self.page_canvas.create_window((0, 0), window=fit_page, anchor=tk.NW)
        fit_page.bind(
            "<Configure>",
            lambda _event: self.page_canvas.configure(scrollregion=self.page_canvas.bbox("all")),
        )
        self.page_canvas.bind(
            "<Configure>",
            lambda event: self.page_canvas.itemconfigure(self._page_window, width=max(900, event.width)),
        )
        self._build_fit_page(fit_page)

    def _build_fit_page(self, page: ttk.Frame) -> None:
        toolbar = ttk.LabelFrame(page, text="文件夹批量或单文件拟合 GMD–PD", padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Entry(toolbar, textvariable=self.calibration_folder_var, width=80, state="readonly").grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(toolbar, text="选择标定文件夹", command=self.choose_calibration_folder).grid(row=0, column=1, padx=5)
        ttk.Label(toolbar, text="尝试偏移").grid(row=0, column=2, padx=(12, 3))
        ttk.Entry(toolbar, textvariable=self.offsets_var, width=18).grid(row=0, column=3)
        self.batch_fit_button = ttk.Button(toolbar, text="遍历拟合并加入缓存", command=self.run_calibration_folder)
        self.batch_fit_button.grid(row=0, column=4, padx=(6, 3))
        ttk.Button(toolbar, text="将选中缓存应用于公式", command=self.apply_selected_calibration).grid(row=0, column=5, padx=3)
        ttk.Button(toolbar, text="删除选中缓存（可多选）", command=self.delete_selected_calibration).grid(row=0, column=6, padx=3)
        ttk.Entry(toolbar, textvariable=self.single_calibration_file_var, width=80, state="readonly").grid(
            row=1, column=0, columnspan=4, sticky=tk.EW, pady=(6, 0),
        )
        ttk.Button(toolbar, text="选择单个 HDF5", command=self.choose_single_calibration_file).grid(
            row=1, column=4, padx=(6, 3), pady=(6, 0),
        )
        self.single_fit_button = ttk.Button(toolbar, text="拟合单文件并加入缓存", command=self.run_single_calibration_file)
        self.single_fit_button.grid(row=1, column=5, columnspan=2, sticky=tk.W, padx=3, pady=(6, 0))
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(toolbar, textvariable=self.fit_result_var, foreground="#334155", wraplength=1450).grid(row=2, column=0, columnspan=9, sticky=tk.W, pady=(7, 0))
        analysis_buttons = ttk.Frame(toolbar)
        analysis_buttons.grid(row=3, column=0, columnspan=9, sticky=tk.W, pady=(7, 0))
        ttk.Label(analysis_buttons, text="独立分析工具（PD/GMD 背底不参与计算）：").pack(side=tk.LEFT)
        ttk.Button(analysis_buttons, text="气体衰减常数测定…", command=self.open_gas_attenuation_window).pack(side=tk.LEFT, padx=3)
        ttk.Button(analysis_buttons, text="标准化斜率统计…", command=self.open_normalized_slope_window).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            analysis_buttons, text="从项目读取第2步数据…", command=self.import_step2_from_project,
        ).pack(side=tk.LEFT, padx=(12, 3))

        # This is the main calibration/cache layer.  It deliberately has a
        # generous fixed requested height because the whole Step 2 page is
        # vertically scrollable.
        body = ttk.Panedwindow(page, orient=tk.HORIZONTAL, height=1380)
        self.calibration_body = body
        body.pack(fill=tk.X, pady=(7, 0))
        cache_panel = ttk.Frame(body, padding=(0, 0, 5, 0))
        cache_panel.pack_propagate(False)
        detail_panel = ttk.Panedwindow(body, orient=tk.VERTICAL)
        body.add(cache_panel, weight=2)
        body.add(detail_panel, weight=1)

        # ttk.Panedwindow weights distribute spare space but do not guarantee
        # the requested initial ratio when child requested widths differ.  Set
        # the first sash once after the real page width is known.
        initial_ratio_applied = False

        def set_initial_pane_ratio(event: tk.Event) -> None:
            nonlocal initial_ratio_applied
            if initial_ratio_applied or event.width < 600:
                return
            body.sashpos(0, round(event.width * 2 / 3))
            initial_ratio_applied = True

        body.bind("<Configure>", set_initial_pane_ratio, add="+")

        ttk.Label(cache_panel, text="拟合缓存区", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W)
        cache_table = ttk.Frame(cache_panel)
        cache_table.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.calibration_cache_tree = ttk.Treeview(
            cache_table,
            columns=("time", "file", "file_mtime", "pd_min", "pprime", "offset", "r2", "slope", "normalized", "intercept", "n", "active"),
            show="headings", height=22, selectmode="extended",
        )
        for column, title, width in (
            ("time", "缓存时间", 150), ("file", "文件目录", 260), ("file_mtime", "文件修改时间", 165),
            ("pd_min", "PD 最低值 (V·s)", 165),
            ("pprime", "Pprime (Torr)", 135),
            ("offset", "偏移", 60), ("r2", "R²", 105), ("slope", "Bfit", 145),
            ("normalized", "标准化斜率", 180), ("intercept", "拟合截距", 150),
            ("n", "N", 65), ("active", "公式", 60),
        ):
            arrow = " ↑" if column == self._cache_sort_column else ""
            self.calibration_cache_tree.heading(column, text=title + arrow, command=lambda name=column: self.sort_calibration_cache(name))
            self.calibration_cache_tree.column(column, width=width, minwidth=width, anchor=tk.CENTER, stretch=False)
        cache_y = ttk.Scrollbar(cache_table, orient=tk.VERTICAL, command=self.calibration_cache_tree.yview)
        cache_x = ttk.Scrollbar(cache_table, orient=tk.HORIZONTAL, command=self.calibration_cache_tree.xview)
        self.calibration_cache_tree.configure(yscrollcommand=cache_y.set, xscrollcommand=cache_x.set)
        self.calibration_cache_tree.grid(row=0, column=0, sticky=tk.NSEW)
        cache_y.grid(row=0, column=1, sticky=tk.NS)
        cache_x.grid(row=1, column=0, sticky=tk.EW)
        cache_table.rowconfigure(0, weight=1)
        cache_table.columnconfigure(0, weight=1)
        self.calibration_cache_tree.bind("<<TreeviewSelect>>", self._on_calibration_cache_select)
        self.calibration_cache_tree.bind("<Delete>", lambda _event: self.delete_selected_calibration())
        self.calibration_cache_tree.bind("<Double-1>", self._edit_cache_cell)

        plot_frame = ttk.Frame(detail_panel, padding=8, height=430)
        plot_frame.pack_propagate(False)
        summary_frame = ttk.Frame(detail_panel, padding=(0, 6, 0, 0))
        detail_panel.add(plot_frame, weight=2)
        detail_panel.add(summary_frame, weight=1)
        self.calibration_figure = Figure(figsize=(6.4, 4.0), dpi=100)
        self.calibration_figure.subplots_adjust(left=0.15, right=0.97, bottom=0.17, top=0.91)
        self.calibration_axis = self.calibration_figure.add_subplot(111)
        self._empty_plot("拟合后在左侧缓存区选择记录")
        canvas_host = ttk.Frame(plot_frame)
        canvas_host.pack(fill=tk.BOTH, expand=True)
        self.calibration_canvas = FigureCanvasTkAgg(self.calibration_figure, master=canvas_host)
        plot_toolbar = RememberingNavigationToolbar(
            self.calibration_canvas, plot_frame, path_store=self.path_store,
            path_key="energy_calibration_plot_save", pack_toolbar=False,
        )
        plot_toolbar.update()
        plot_toolbar.pack(fill=tk.X, before=canvas_host)
        canvas_widget = self.calibration_canvas.get_tk_widget()
        canvas_widget.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        def resize_preview(event) -> None:
            width = max(160, min(event.width - 12, 860))
            height = max(140, min(event.height - 12, 500))
            if width / height > 1.6:
                width = round(height * 1.6)
            else:
                height = round(width / 1.6)
            canvas_widget.configure(width=width, height=height)

        canvas_host.bind("<Configure>", resize_preview)
        self.calibration_canvas.draw()

        range_frame = ttk.LabelFrame(summary_frame, text="当前缓存图的独立拟合范围", padding=6)
        range_frame.pack(fill=tk.X, pady=(0, 5))
        for column, (label, variable) in enumerate((
            ("X min", self.fit_xmin_var), ("X max", self.fit_xmax_var),
            ("Y min", self.fit_ymin_var), ("Y max", self.fit_ymax_var),
        )):
            ttk.Label(range_frame, text=label).grid(row=0, column=column * 2, padx=(2, 3))
            ttk.Entry(range_frame, textvariable=variable, width=13).grid(row=0, column=column * 2 + 1, padx=(0, 7))
        ttk.Button(range_frame, text="应用拟合范围", command=self.apply_fit_range).grid(row=0, column=8, padx=(5, 3))
        ttk.Label(range_frame, text="留空表示不限；未选点浅色显示。双击缓存表的 Pprime 单元格可填写气压。", foreground="#64748b").grid(row=1, column=0, columnspan=9, sticky=tk.W, pady=(5, 0))

        ttk.Label(summary_frame, text="当前文件的全部偏移结果", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W)
        self.calibration_offsets_tree = ttk.Treeview(
            summary_frame,
            columns=("offset", "n", "slope", "intercept", "r", "r2", "stderr", "state"),
            show="headings", height=6,
        )
        for column, title, width in (
            ("offset", "偏移", 60), ("n", "N", 60), ("slope", "斜率", 125),
            ("intercept", "截距", 125), ("r", "r", 90), ("r2", "R²", 90),
            ("stderr", "斜率标准误", 125), ("state", "状态", 95),
        ):
            self.calibration_offsets_tree.heading(column, text=title)
            self.calibration_offsets_tree.column(column, width=width, anchor=tk.CENTER)
        self.calibration_offsets_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.calibration_offsets_tree.bind("<<TreeviewSelect>>", self._on_calibration_offset_select)

    @staticmethod
    def fit_gas_attenuation_data(pressures: np.ndarray, slopes: np.ndarray) -> dict[str, Any]:
        """Fit ln(r)=ln(A)+kP and return the OLS 1-sigma parameter errors."""
        pressures = np.asarray(pressures, dtype=float)
        slopes = np.asarray(slopes, dtype=float)
        mask = np.isfinite(pressures) & np.isfinite(slopes) & (slopes > 0)
        pressures, slopes = pressures[mask], slopes[mask]
        if len(pressures) < 3:
            raise ValueError("至少需要 3 组有限且斜率大于 0 的 P–斜率数据。")
        if np.any(pressures < 0):
            raise ValueError("气压 P 不能为负数。")
        if np.ptp(pressures) <= 0:
            raise ValueError("气压 P 不能全部相同。")
        log_slopes = np.log(slopes)
        design = np.column_stack([pressures, np.ones_like(pressures)])
        params, _, rank, _ = np.linalg.lstsq(design, log_slopes, rcond=None)
        if rank < 2:
            raise ValueError("数据不足以确定气体衰减常数。")
        k, log_a = map(float, params)
        residual = log_slopes - design @ params
        dof = len(pressures) - 2
        residual_variance = float(np.sum(residual**2) / dof)
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        sigma_k = float(math.sqrt(max(0.0, covariance[0, 0])))
        sigma_log_a = float(math.sqrt(max(0.0, covariance[1, 1])))
        fitted_log = design @ params
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((log_slopes - np.mean(log_slopes)) ** 2))
        return {
            "k": k,
            "A": float(math.exp(log_a)),
            "sigma_k": sigma_k,
            "sigma_A": float(math.exp(log_a) * sigma_log_a),
            "r_squared_log": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
            "n": int(len(pressures)),
            "pressures": pressures.tolist(),
            "slopes": slopes.tolist(),
            "fitted_slopes": np.exp(fitted_log).tolist(),
        }

    def open_gas_attenuation_window(self) -> None:
        if self.gas_attenuation_window is not None and self.gas_attenuation_window.winfo_exists():
            self.gas_attenuation_window.lift()
            self.gas_attenuation_window.focus_force()
            return
        window = tk.Toplevel(self)
        self.gas_attenuation_window = window
        window.title("气体衰减常数测定")
        window.geometry("1120x720")
        window.minsize(900, 600)
        body = ttk.Panedwindow(window, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        data_frame = ttk.LabelFrame(body, text="P–斜率数据", padding=8, width=360)
        plot_frame = ttk.LabelFrame(body, text="指数衰减拟合", padding=8)
        body.add(data_frame, weight=0)
        body.add(plot_frame, weight=1)
        ttk.Label(
            data_frame,
            text="拟合形式：ln(r)=ln(A)+kP。可从 Excel/Origin 直接粘贴两列；sigma_k 为斜率标准误（1σ）。",
            wraplength=330, foreground="#475569", justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(0, 6))
        self.gas_attenuation_sheet = Sheet(
            data_frame, headers=["P (Torr)", "对应斜率 r"], show_row_index=True,
            data=[list(row[:2]) for row in self.gas_attenuation_data],
            paste_can_expand_x=False, paste_can_expand_y=True,
        )
        self.gas_attenuation_sheet.pack(fill=tk.BOTH, expand=True)
        self.gas_attenuation_sheet.enable_bindings(
            "single_select", "drag_select", "select_all", "row_select", "column_select",
            "ctrl_select", "copy", "paste", "delete", "edit_cell", "arrowkeys",
            "undo", "redo", "column_width_resize", "row_height_resize",
        )
        self.gas_attenuation_sheet.set_column_widths([125, 150])
        buttons = ttk.Frame(data_frame)
        buttons.pack(fill=tk.X, pady=(7, 0))
        ttk.Button(buttons, text="增加空行", command=self._add_gas_attenuation_row).pack(side=tk.LEFT)
        ttk.Button(buttons, text="清空", command=self._clear_gas_attenuation_data).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="拟合并绘图", command=self.fit_gas_attenuation).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="应用 k 和 sigma_k", command=self.apply_gas_attenuation_fit).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            data_frame, textvariable=self.gas_attenuation_result_var,
            wraplength=330, justify=tk.LEFT, foreground="#334155",
        ).pack(fill=tk.X, pady=(8, 0))

        self.gas_attenuation_figure = Figure(figsize=(7.0, 5.6), dpi=100)
        self.gas_attenuation_figure.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.90)
        self.gas_attenuation_axis = self.gas_attenuation_figure.add_subplot(111)
        self.gas_attenuation_axis.text(
            0.5, 0.5, "输入 P 和对应斜率后点击“拟合并绘图”",
            ha="center", va="center", transform=self.gas_attenuation_axis.transAxes,
        )
        self.gas_attenuation_canvas = FigureCanvasTkAgg(self.gas_attenuation_figure, master=plot_frame)
        gas_toolbar = RememberingNavigationToolbar(
            self.gas_attenuation_canvas, plot_frame, path_store=self.path_store,
            path_key="calibration_gas_attenuation_plot_save", pack_toolbar=False,
        )
        gas_toolbar.update()
        gas_toolbar.pack(fill=tk.X)
        self.gas_attenuation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.gas_attenuation_canvas.draw()
        if self.gas_attenuation_fit:
            self._draw_gas_attenuation_fit(self.gas_attenuation_fit)

        def close_window() -> None:
            self._store_gas_attenuation_sheet()
            window.destroy()
            self.gas_attenuation_window = None
            self.gas_attenuation_sheet = None

        window.protocol("WM_DELETE_WINDOW", close_window)

    def _store_gas_attenuation_sheet(self) -> None:
        sheet = getattr(self, "gas_attenuation_sheet", None)
        if sheet is not None:
            try:
                if sheet.winfo_exists():
                    self.gas_attenuation_data = [list(row[:2]) for row in sheet.get_sheet_data()]
            except tk.TclError:
                pass

    def _add_gas_attenuation_row(self) -> None:
        data = [list(row[:2]) for row in self.gas_attenuation_sheet.get_sheet_data()]
        data.append(["", ""])
        self.gas_attenuation_sheet.set_sheet_data(data, reset_col_positions=False, reset_row_positions=False)

    def _clear_gas_attenuation_data(self) -> None:
        self.gas_attenuation_data = [["", ""] for _ in range(12)]
        self.gas_attenuation_fit = {}
        self.gas_attenuation_sheet.set_sheet_data(self.gas_attenuation_data, reset_col_positions=False)
        self.gas_attenuation_result_var.set("尚未拟合气体衰减常数。")
        self.gas_attenuation_axis.clear()
        self.gas_attenuation_axis.text(
            0.5, 0.5, "输入 P 和对应斜率后点击“拟合并绘图”",
            ha="center", va="center", transform=self.gas_attenuation_axis.transAxes,
        )
        self.gas_attenuation_canvas.draw_idle()

    def fit_gas_attenuation(self) -> None:
        self._store_gas_attenuation_sheet()
        pressures, slopes = [], []
        invalid_rows = []
        for row_index, row in enumerate(self.gas_attenuation_data, 1):
            left = str(row[0]).strip() if len(row) > 0 else ""
            right = str(row[1]).strip() if len(row) > 1 else ""
            if not left and not right:
                continue
            try:
                pressures.append(float(left))
                slopes.append(float(right))
            except ValueError:
                invalid_rows.append(row_index)
        if invalid_rows:
            messagebox.showerror("数据无效", f"第 {', '.join(map(str, invalid_rows[:12]))} 行不是完整数值。", parent=self.gas_attenuation_window)
            return
        try:
            result = self.fit_gas_attenuation_data(np.asarray(pressures), np.asarray(slopes))
        except Exception as exc:
            messagebox.showerror("无法拟合", str(exc), parent=self.gas_attenuation_window)
            return
        self.gas_attenuation_fit = result
        self.gas_attenuation_result_var.set(
            f"k = {result['k']:.10g} ± {result['sigma_k']:.4g} (1σ)\n"
            f"A = {result['A']:.10g} ± {result['sigma_A']:.4g}\n"
            f"R²[ln(r)] = {result['r_squared_log']:.7g}；N = {result['n']}"
        )
        self._draw_gas_attenuation_fit(result)
        self.state.notify("gas_attenuation_fit")

    def _draw_gas_attenuation_fit(self, result: dict[str, Any]) -> None:
        axis = self.gas_attenuation_axis
        axis.clear()
        pressures = np.asarray(result["pressures"], dtype=float)
        slopes = np.asarray(result["slopes"], dtype=float)
        axis.scatter(pressures, slopes, s=40, color="#2563eb", alpha=0.85, label=f"实验数据 (N={len(pressures)})")
        span = float(np.ptp(pressures))
        margin = max(span * 0.05, max(abs(float(np.mean(pressures))), 1.0) * 1e-4)
        xline = np.linspace(float(np.min(pressures) - margin), float(np.max(pressures) + margin), 400)
        yline = float(result["A"]) * np.exp(float(result["k"]) * xline)
        axis.plot(
            xline, yline, color="#dc2626", linewidth=2.0,
            label=f"r={float(result['A']):.6g}·exp({float(result['k']):+.6g}P)",
        )
        axis.set_title("气体衰减常数拟合")
        axis.set_xlabel("气压 P (Torr)")
        axis.set_ylabel("对应斜率 r")
        axis.grid(alpha=0.28)
        axis.legend()
        self.gas_attenuation_canvas.draw_idle()

    def apply_gas_attenuation_fit(self) -> None:
        if not self.gas_attenuation_fit:
            messagebox.showinfo("尚未拟合", "请先输入数据并完成拟合。", parent=self.gas_attenuation_window or self)
            return
        self.state.constants["k"] = float(self.gas_attenuation_fit["k"])
        self.state.constants["sigk"] = float(self.gas_attenuation_fit["sigma_k"])
        normalized_values = self._recalculate_normalized_slope_cache(float(self.gas_attenuation_fit["k"]))
        if (
            normalized_values and self.normalized_window is not None
            and self.normalized_window.winfo_exists()
        ):
            self._show_normalized_slope_distribution(normalized_values, float(self.gas_attenuation_fit["k"]))
        self.state.notify("gas_attenuation_applied")
        self.set_status(
            f"气体衰减常数 k={self.gas_attenuation_fit['k']:.10g}、"
            f"sigk={self.gas_attenuation_fit['sigma_k']:.6g} 已应用到第 3 步命名常数；"
            f"同时重算了 {len(normalized_values)} 条有效标准化斜率。"
        )

    def open_normalized_slope_window(self) -> None:
        if self.normalized_window is not None and self.normalized_window.winfo_exists():
            self.normalized_window.lift()
            return
        window = tk.Toplevel(self)
        self.normalized_window = window
        window.title("标准化斜率统计")
        window.geometry("880x650")
        controls = ttk.Frame(window, padding=8)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="统计全部已填写 Pprime 的缓存并更新分布图", command=self.calculate_normalized_slopes).pack(side=tk.LEFT)
        ttk.Label(controls, text="标准化斜率 = Bfit / exp(k·Pprime)", foreground="#475569").pack(side=tk.LEFT, padx=10)
        self.normalized_figure = Figure(figsize=(7.2, 5.1), dpi=100)
        self.normalized_figure.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.88)
        self.normalized_axis = self.normalized_figure.add_subplot(111)
        self.normalized_axis.text(0.5, 0.5, "点击上方按钮生成分布图", ha="center", va="center", transform=self.normalized_axis.transAxes)
        self.normalized_canvas = FigureCanvasTkAgg(self.normalized_figure, master=window)
        normalized_toolbar = RememberingNavigationToolbar(
            self.normalized_canvas, window, path_store=self.path_store,
            path_key="calibration_normalized_slope_plot_save", pack_toolbar=False,
        )
        normalized_toolbar.update()
        normalized_toolbar.pack(fill=tk.X)
        self.normalized_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.normalized_canvas.draw()
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_analysis_window("normalized"))

    def open_pd_background_window(self) -> None:
        if self.pd_background_window is not None and self.pd_background_window.winfo_exists():
            self.pd_background_window.lift()
            return
        window = tk.Toplevel(self)
        self.pd_background_window = window
        window.title("PD 背底计算")
        window.geometry("940x700")
        controls = ttk.Frame(window, padding=8)
        controls.pack(fill=tk.X)
        ttk.Label(controls, text="示波器 baseline (V·s)").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(controls, textvariable=self.pd_baseline_var, width=16).grid(row=0, column=1, padx=(4, 10))
        ttk.Label(controls, text="baseline 标准不确定度 (V·s)").grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(controls, textvariable=self.pd_baseline_sigma_var, width=16).grid(row=0, column=3, padx=(4, 10))
        ttk.Button(controls, text="用选中缓存计算背底", command=self.calculate_pd_background).grid(row=0, column=4, padx=3)
        ttk.Button(controls, text="应用背底及其误差", command=self.apply_pd_background).grid(row=0, column=5, padx=3)
        ttk.Label(controls, textvariable=self.pd_background_file_var, foreground="#475569", wraplength=890).grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=(7, 0))
        ttk.Label(controls, textvariable=self.pd_background_result_var, foreground="#334155", wraplength=890).grid(row=2, column=0, columnspan=6, sticky=tk.W, pady=(3, 0))
        self.background_figure = Figure(figsize=(7.8, 5.0), dpi=100)
        self.background_figure.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.88)
        self.background_axis = self.background_figure.add_subplot(111)
        self.background_axis.text(0.5, 0.5, "选择主页面缓存并输入 baseline", ha="center", va="center", transform=self.background_axis.transAxes)
        self.background_canvas = FigureCanvasTkAgg(self.background_figure, master=window)
        background_toolbar = RememberingNavigationToolbar(
            self.background_canvas, window, path_store=self.path_store,
            path_key="calibration_pd_background_plot_save", pack_toolbar=False,
        )
        background_toolbar.update()
        background_toolbar.pack(fill=tk.X)
        self.background_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.background_canvas.draw()
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_analysis_window("background"))

    def _close_analysis_window(self, kind: str) -> None:
        if kind == "normalized" and self.normalized_window is not None:
            self.normalized_window.destroy()
            self.normalized_window = None
        elif kind == "background" and self.pd_background_window is not None:
            self.pd_background_window.destroy()
            self.pd_background_window = None

    def choose_calibration_folder(self) -> None:
        fallback = Path(self.calibration_folder_var.get()) if self.calibration_folder_var.get() else Path.cwd()
        initial_dir = self.path_store.initial_dir("energy_calibration_folder_open", fallback) if self.path_store else str(fallback)
        folder = filedialog.askdirectory(parent=self, initialdir=initial_dir, title="选择要递归拟合的 GMD–PD HDF5 文件夹")
        if folder:
            self.calibration_folder_var.set(folder)
            if self.path_store:
                self.path_store.remember_directory("energy_calibration_folder_open", folder)

    def choose_single_calibration_file(self) -> None:
        fallback = Path(self.single_calibration_file_var.get()).parent if self.single_calibration_file_var.get() else Path.cwd()
        initial_dir = self.path_store.initial_dir("calibration_single_hdf5_open", fallback) if self.path_store else str(fallback)
        filename = filedialog.askopenfilename(
            parent=self, initialdir=initial_dir, title="选择单个 GMD–PD HDF5",
            filetypes=[("HDF5", "*.hdf5 *.h5"), ("所有文件", "*.*")],
        )
        if filename:
            self.single_calibration_file_var.set(filename)
            if self.path_store:
                self.path_store.remember_file("calibration_single_hdf5_open", filename)

    @staticmethod
    def calibration_config_from_project(project: dict[str, Any]) -> dict[str, Any]:
        """Extract only Step 2 data from current and legacy integrated projects."""
        calibration = project.get("calibration_page")
        if isinstance(calibration, dict):
            return copy.deepcopy(calibration)
        energy = project.get("energy_page")
        if isinstance(energy, dict) and any(
            key in energy for key in ("calibration_cache", "calibration_file", "offsets")
        ):
            # Version <= 6 kept the then-calibration section under Step 3.
            keys = (
                "calibration_folder", "calibration_file", "single_calibration_file",
                "offsets", "calibration_cache", "selected_calibration_id",
                "pd_baseline", "pd_baseline_sigma", "last_pd_background",
                "gas_attenuation_data", "gas_attenuation_fit",
            )
            return {key: copy.deepcopy(energy[key]) for key in keys if key in energy}
        raise ValueError("所选 JSON 中没有可识别的第 2 步标定数据。")

    @staticmethod
    def merge_calibration_configs(
        current: dict[str, Any], incoming: dict[str, Any],
    ) -> tuple[dict[str, Any], int, int]:
        """Use incoming Step 2 settings while retaining unique current caches."""
        merged = copy.deepcopy(current)
        merged.update(copy.deepcopy(incoming))
        existing = [copy.deepcopy(item) for item in current.get("calibration_cache", [])]
        imported = [copy.deepcopy(item) for item in incoming.get("calibration_cache", [])]
        known_ids = {str(item.get("cache_id", "")) for item in existing if item.get("cache_id")}
        known_fingerprints = {
            (str(item.get("source", "")), str(item.get("created_at", "")), str(item.get("best_offset", 0)))
            for item in existing
        }
        fingerprint_ids = {
            (str(item.get("source", "")), str(item.get("created_at", "")), str(item.get("best_offset", 0))): str(item.get("cache_id", ""))
            for item in existing
        }
        added = 0
        skipped = 0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        incoming_selected = str(incoming.get("selected_calibration_id", ""))
        resolved_selected = incoming_selected
        for index, item in enumerate(imported, 1):
            fingerprint = (
                str(item.get("source", "")), str(item.get("created_at", "")),
                str(item.get("best_offset", 0)),
            )
            cache_id = str(item.get("cache_id", ""))
            if fingerprint in known_fingerprints:
                if cache_id == incoming_selected:
                    resolved_selected = fingerprint_ids.get(fingerprint, cache_id)
                skipped += 1
                continue
            if not cache_id or cache_id in known_ids:
                original_id = cache_id
                cache_id = f"cal_import_{stamp}_{index}"
                item["cache_id"] = cache_id
                if original_id == incoming_selected:
                    resolved_selected = cache_id
            existing.append(item)
            known_ids.add(cache_id)
            known_fingerprints.add(fingerprint)
            fingerprint_ids[fingerprint] = cache_id
            added += 1
        merged["calibration_cache"] = existing
        merged["selected_calibration_id"] = resolved_selected or str(current.get("selected_calibration_id", ""))
        return merged, added, skipped

    def _close_analysis_windows(self) -> None:
        self._store_gas_attenuation_sheet()
        for attribute in ("gas_attenuation_window", "normalized_window", "pd_background_window"):
            window = getattr(self, attribute, None)
            try:
                if window is not None and window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass
            setattr(self, attribute, None)
        self.gas_attenuation_sheet = None

    def import_step2_from_project(self) -> None:
        fallback = Path.cwd()
        initial_dir = self.path_store.initial_dir("calibration_project_open", fallback) if self.path_store else str(fallback)
        filename = filedialog.askopenfilename(
            parent=self, initialdir=initial_dir, title="读取其他项目的第 2 步数据",
            filetypes=[("XFEL 项目", "*.xfel.json *.json"), ("所有文件", "*.*")],
        )
        if not filename:
            return
        try:
            project = load_json(filename)
            if not isinstance(project, dict):
                raise ValueError("项目文件顶层不是 JSON 对象。")
            incoming = self.calibration_config_from_project(project)
            imported_count = len(incoming.get("calibration_cache", []))
        except Exception as exc:
            messagebox.showerror("无法读取第 2 步数据", str(exc), parent=self)
            return
        if self.path_store:
            self.path_store.remember_file("calibration_project_open", filename)

        append = False
        if self.calibration_cache:
            answer = messagebox.askyesnocancel(
                "导入第 2 步数据",
                f"来源项目包含 {imported_count} 条拟合缓存。\n\n"
                "选择“是”：追加并去重缓存，同时采用来源项目的第 2 步参数和分析数据。\n"
                "选择“否”：用来源项目完整替换当前第 2 步数据。\n"
                "选择“取消”：不导入。\n\n"
                "该操作不会修改当前项目的第 1、3、4 步数据，也不会自动改写第 3 步命名常数。",
                parent=self,
            )
            if answer is None:
                return
            append = bool(answer)

        added = imported_count
        skipped = 0
        config = incoming
        if append:
            config, added, skipped = self.merge_calibration_configs(self.export_config(), incoming)
        self.import_config(config)
        self.state.notify("calibration_project_import")
        mode = "追加" if append else "替换"
        detail = f"新增 {added} 条，跳过重复 {skipped} 条" if append else f"读取 {imported_count} 条缓存"
        message = (
            f"已从 {Path(filename).name} {mode}第 2 步全部数据：{detail}。"
            "第 1、3、4 步及当前命名常数未改动。"
        )
        self.fit_result_var.set(message)
        self.set_status(message)

    @staticmethod
    def _fit_result_cache_item(
        path: Path,
        requested_offsets: list[int],
        result,
        sequence: int,
        pd_minimum: float,
        pd_min_count: int,
    ) -> dict[str, Any]:
        return {
            "cache_id": f"cal_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{sequence}",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(path),
            "file_mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "requested_offsets": requested_offsets,
            "best_offset": int(result.offset),
            "selected_offset": int(result.offset),
            "pd_minimum": float(pd_minimum),
            "pd_min_count": int(pd_min_count),
            "pprime": None,
            "normalized_slope": None,
            "normalized_slope_stderr": None,
            "fit_ranges": {},
            "all_results": result.all_results,
            "curves": [
                {**{key: value for key, value in curve.items() if key not in {"gmd", "pd"}},
                 "gmd": np.asarray(curve["gmd"], dtype=float).tolist(),
                 "pd": np.asarray(curve["pd"], dtype=float).tolist()}
                for curve in result.curves
            ],
        }

    def run_calibration_folder(self) -> None:
        if self._fit_running:
            return
        folder_text = self.calibration_folder_var.get().strip()
        if not folder_text:
            self.choose_calibration_folder()
            folder_text = self.calibration_folder_var.get().strip()
        root = Path(folder_text)
        if not root.is_dir():
            return
        try:
            requested_offsets = parse_offsets(self.offsets_var.get())
        except Exception as exc:
            messagebox.showerror("偏移范围无效", str(exc), parent=self)
            return
        files = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".hdf5", ".h5"}),
            key=lambda path: self.natural_path_key(str(path)),
        )
        if not files:
            messagebox.showinfo("没有 HDF5", "该文件夹及其子文件夹中没有 .hdf5/.h5 文件。", parent=self)
            return
        self._start_calibration_fit(files, requested_offsets, "递归批量")

    def run_single_calibration_file(self) -> None:
        if self._fit_running:
            return
        path_text = self.single_calibration_file_var.get().strip()
        if not path_text:
            self.choose_single_calibration_file()
            path_text = self.single_calibration_file_var.get().strip()
        path = Path(path_text)
        if not path.is_file() or path.suffix.lower() not in {".hdf5", ".h5"}:
            if path_text:
                messagebox.showerror("文件无效", "请选择存在的 .hdf5 或 .h5 文件。", parent=self)
            return
        try:
            requested_offsets = parse_offsets(self.offsets_var.get())
        except Exception as exc:
            messagebox.showerror("偏移范围无效", str(exc), parent=self)
            return
        if self.path_store:
            self.path_store.remember_file("calibration_single_hdf5_open", path)
        self._start_calibration_fit([path], requested_offsets, "单文件")

    def _start_calibration_fit(self, files: list[Path], requested_offsets: list[int], operation: str) -> None:
        self._fit_running = True
        self.batch_fit_button.configure(state=tk.DISABLED)
        self.single_fit_button.configure(state=tk.DISABLED)
        self.fit_result_var.set(f"正在{operation}拟合 {len(files)} 个 HDF5 文件，并同步读取 PD 最低值……")
        self.set_status(f"第 2 步正在{operation}拟合 {len(files)} 个 GMD–PD 文件。")
        self.after(60, self._poll_calibration_fit)

        def worker() -> None:
            items: list[dict[str, Any]] = []
            errors: list[tuple[str, str]] = []
            for sequence, path in enumerate(files, len(self.calibration_cache) + 1):
                try:
                    result = fit_gmd_pd(path, requested_offsets)
                    pd_minimum, pd_min_count = read_pd_minimum(path)
                    items.append(self._fit_result_cache_item(
                        path, requested_offsets, result, sequence, pd_minimum, pd_min_count,
                    ))
                except Exception as exc:
                    errors.append((str(path), str(exc)))
            self._fit_queue.put({"items": items, "errors": errors, "total": len(files)})

        threading.Thread(target=worker, daemon=True, name="xfel-gmd-pd-batch-fit").start()

    def _poll_calibration_fit(self) -> None:
        if not self.winfo_exists():
            return
        try:
            result = self._fit_queue.get_nowait()
        except queue.Empty:
            if self._fit_running:
                self.after(60, self._poll_calibration_fit)
            return
        self._finish_calibration_fit(result)

    def _finish_calibration_fit(self, batch: dict[str, Any]) -> None:
        self._fit_running = False
        self.batch_fit_button.configure(state=tk.NORMAL)
        self.single_fit_button.configure(state=tk.NORMAL)
        items = list(batch.get("items", []))
        errors = list(batch.get("errors", []))
        self.calibration_cache.extend(items)
        select_id = str(items[-1]["cache_id"]) if items else None
        self._sort_calibration_cache_in_place()
        self._refresh_calibration_cache_tree(select_id=select_id)
        self.state.notify("calibration_cache")
        result_text = f"批量拟合完成：成功 {len(items)}/{int(batch.get('total', 0))} 个文件，失败 {len(errors)} 个。"
        valid_minima = [item for item in items if item.get("pd_minimum") is not None]
        if valid_minima:
            lowest = min(valid_minima, key=lambda item: float(item["pd_minimum"]))
            result_text += (
                f" 全局最低 PD={float(lowest['pd_minimum']):.12g} V·s；"
                f"来源：{self.source_label(str(lowest.get('source', '')))}。"
            )
        self.fit_result_var.set(result_text)
        self.set_status(result_text)
        if errors:
            examples = "\n".join(f"{self.source_label(path)}：{error}" for path, error in errors[:12])
            messagebox.showwarning("部分文件未加入缓存", f"失败 {len(errors)} 个文件：\n\n{examples}", parent=self)

    @staticmethod
    def source_label(source: str) -> str:
        path = Path(source)
        return str(Path(*path.parts[-3:])) if len(path.parts) >= 3 else str(path)

    @staticmethod
    def natural_path_key(value: str) -> tuple[tuple[int, Any], ...]:
        """Case-insensitive path key with digit runs compared numerically."""
        normalized = str(value).replace("/", "\\").casefold()
        return tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", normalized)
            if part
        )

    def _cache_sort_value(self, item: dict[str, Any], column: str) -> Any:
        offset = int(item.get("selected_offset", item.get("best_offset", 0)))
        result = self._calibration_result(item, offset) or {}
        source = str(item.get("source", ""))
        mapping = {
            "time": item.get("created_at"),
            "file": (self.natural_path_key(self.source_label(source)), self.natural_path_key(source)),
            "file_mtime": item.get("file_mtime"),
            "pd_min": item.get("pd_minimum"), "pprime": item.get("pprime"),
            "offset": offset, "r2": result.get("r_squared"), "slope": result.get("slope"),
            "normalized": item.get("normalized_slope"), "intercept": result.get("intercept"),
            "n": result.get("n"),
            "active": str(item.get("cache_id", "")) == str(self.state.gmd_pd_fit.get("cache_id", "")),
        }
        return mapping.get(column)

    def _sort_calibration_cache_in_place(self) -> None:
        column = self._cache_sort_column
        present = [item for item in self.calibration_cache if self._cache_sort_value(item, column) is not None]
        missing = [item for item in self.calibration_cache if self._cache_sort_value(item, column) is None]
        present.sort(key=lambda item: self._cache_sort_value(item, column), reverse=self._cache_sort_reverse)
        self.calibration_cache = present + missing

    def _update_cache_sort_headings(self) -> None:
        titles = {
            "time": "缓存时间", "file": "文件目录", "file_mtime": "文件修改时间",
            "pd_min": "PD 最低值 (V·s)",
            "pprime": "Pprime (Torr)", "offset": "偏移", "r2": "R²", "slope": "Bfit",
            "normalized": "标准化斜率", "intercept": "拟合截距", "n": "N", "active": "公式",
        }
        for name, title in titles.items():
            arrow = " ↓" if name == self._cache_sort_column and self._cache_sort_reverse else (
                " ↑" if name == self._cache_sort_column else ""
            )
            self.calibration_cache_tree.heading(
                name, text=title + arrow, command=lambda key=name: self.sort_calibration_cache(key),
            )

    def _refresh_calibration_cache_tree(self, select_id: str | None = None) -> None:
        active_id = str(self.state.gmd_pd_fit.get("cache_id", ""))
        previous_selection = set(self.calibration_cache_tree.selection())
        self.calibration_cache_tree.delete(*self.calibration_cache_tree.get_children())
        for item in self.calibration_cache:
            offset = int(item.get("selected_offset", item.get("best_offset", 0)))
            result = self._calibration_result(item, offset)
            if result is None:
                continue
            cache_id = str(item["cache_id"])
            created = str(item.get("created_at", "")).replace("T", " ")
            modified = str(item.get("file_mtime", "")).replace("T", " ")
            pd_minimum = item.get("pd_minimum")
            pprime = item.get("pprime")
            normalized = item.get("normalized_slope")
            self.calibration_cache_tree.insert(
                "", tk.END, iid=cache_id,
                values=(created, self.source_label(str(item.get("source", ""))), modified,
                        "" if pd_minimum is None else f"{float(pd_minimum):.12g}",
                        "" if pprime is None else f"{float(pprime):.10g}", f"{offset:+d}",
                        f"{float(result['r_squared']):.6g}", f"{float(result['slope']):.7g}",
                        "" if normalized is None else f"{float(normalized):.10g}",
                        f"{float(result['intercept']):.7g}", int(result["n"]), "当前" if cache_id == active_id else ""),
            )
        target = select_id or self.selected_calibration_id or active_id
        if target and self.calibration_cache_tree.exists(target):
            self.calibration_cache_tree.selection_set(target)
            self.calibration_cache_tree.focus(target)
            self.calibration_cache_tree.see(target)
            self._display_calibration_cache(target)
        elif previous_selection:
            existing = [iid for iid in previous_selection if self.calibration_cache_tree.exists(iid)]
            if existing:
                self.calibration_cache_tree.selection_set(existing)

    def sort_calibration_cache(self, column: str) -> None:
        if self._cache_sort_column == column:
            self._cache_sort_reverse = not self._cache_sort_reverse
        else:
            self._cache_sort_column = column
            self._cache_sort_reverse = False

        self._sort_calibration_cache_in_place()
        self._update_cache_sort_headings()
        self._refresh_calibration_cache_tree(select_id=self.selected_calibration_id or None)

    @staticmethod
    def _calibration_result(item: dict[str, Any], offset: int) -> dict[str, Any] | None:
        return next((result for result in item.get("all_results", []) if int(result.get("offset", 999999)) == int(offset)), None)

    def _selected_cache_item(self) -> dict[str, Any] | None:
        return next((item for item in self.calibration_cache if str(item.get("cache_id")) == self.selected_calibration_id), None)

    def _on_calibration_cache_select(self, _event=None) -> None:
        selection = self.calibration_cache_tree.selection()
        if selection:
            focused = self.calibration_cache_tree.focus()
            self._display_calibration_cache(focused if focused in selection else selection[-1])

    def _edit_cache_cell(self, event: tk.Event) -> None:
        if self.calibration_cache_tree.identify_region(event.x, event.y) != "cell":
            return
        row_id = self.calibration_cache_tree.identify_row(event.y)
        column_id = self.calibration_cache_tree.identify_column(event.x)
        if not row_id or column_id != "#5":
            return
        item = next((record for record in self.calibration_cache if str(record.get("cache_id")) == row_id), None)
        if item is None:
            return
        bbox = self.calibration_cache_tree.bbox(row_id, column_id)
        if not bbox:
            return
        x, y, width, height = bbox
        entry = ttk.Entry(self.calibration_cache_tree)
        entry.place(x=x, y=y, width=width, height=height)
        if item.get("pprime") is not None:
            entry.insert(0, f"{float(item['pprime']):.10g}")
        entry.focus_set()
        entry.select_range(0, tk.END)
        finished = False

        def commit(_event=None) -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            text = entry.get().strip()
            try:
                item["pprime"] = None if not text else float(text)
            except ValueError:
                messagebox.showerror("气压无效", "Pprime 必须是以 Torr 为单位的数值。", parent=self)
                entry.destroy()
                return
            item["normalized_slope"] = None
            item["normalized_slope_stderr"] = None
            entry.destroy()
            self.selected_calibration_id = row_id
            self._refresh_calibration_cache_tree(select_id=row_id)
            self.state.notify("calibration_pprime")

        def cancel(_event=None) -> str:
            nonlocal finished
            finished = True
            entry.destroy()
            return "break"

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", cancel)

    def _display_calibration_cache(self, cache_id: str) -> None:
        item = next((record for record in self.calibration_cache if str(record.get("cache_id")) == str(cache_id)), None)
        if item is None:
            return
        self.selected_calibration_id = str(cache_id)
        self.pd_background_file_var.set(
            f"当前选择：{self.source_label(str(item.get('source', '')))}"
        )
        self.calibration_offsets_tree.delete(*self.calibration_offsets_tree.get_children())
        best_offset = int(item.get("best_offset", 0))
        active_cache = str(self.state.gmd_pd_fit.get("cache_id", ""))
        active_offset = self.state.gmd_pd_fit.get("offset")
        for result in item.get("all_results", []):
            offset = int(result["offset"])
            states = []
            if offset == best_offset:
                states.append("本次最优")
            is_active = str(cache_id) == active_cache and active_offset is not None and offset == int(active_offset)
            if is_active:
                states.append("当前公式")
            self.calibration_offsets_tree.insert(
                "", tk.END, iid=f"{cache_id}::offset::{offset}",
                values=(f"{offset:+d}", int(result["n"]), f"{float(result['slope']):.10g}",
                        f"{float(result['intercept']):.10g}", f"{float(result['r']):.7g}",
                        f"{float(result['r_squared']):.7g}", f"{float(result['slope_stderr']):.7g}", " / ".join(states)),
                tags=("active",) if is_active else (("best",) if offset == best_offset else ()),
            )
        self.calibration_offsets_tree.tag_configure("best", background="#dcfce7")
        self.calibration_offsets_tree.tag_configure("active", background="#dbeafe")
        default_offset = int(item.get("selected_offset", best_offset))
        iid = f"{cache_id}::offset::{default_offset}"
        if self.calibration_offsets_tree.exists(iid):
            self.calibration_offsets_tree.selection_set(iid)
            self._display_calibration_offset(item, default_offset)

    def _on_calibration_offset_select(self, _event=None) -> None:
        item = self._selected_cache_item()
        selection = self.calibration_offsets_tree.selection()
        if item is None or not selection:
            return
        try:
            offset = int(selection[0].rsplit("::", 1)[1])
        except (ValueError, IndexError):
            return
        self._display_calibration_offset(item, offset)

    def _display_calibration_offset(self, item: dict[str, Any], offset: int) -> None:
        curve = next((entry for entry in item.get("curves", []) if int(entry.get("offset", 999999)) == offset), None)
        result = self._calibration_result(item, offset)
        if curve is None or result is None:
            return
        gmd = np.asarray(curve.get("gmd", []), dtype=float)
        pd = np.asarray(curve.get("pd", []), dtype=float)
        item["selected_offset"] = int(offset)
        range_data = dict(item.get("fit_ranges", {}).get(str(offset), {}))
        self.fit_xmin_var.set("" if range_data.get("xmin") is None else f"{float(range_data['xmin']):.12g}")
        self.fit_xmax_var.set("" if range_data.get("xmax") is None else f"{float(range_data['xmax']):.12g}")
        self.fit_ymin_var.set("" if range_data.get("ymin") is None else f"{float(range_data['ymin']):.12g}")
        self.fit_ymax_var.set("" if range_data.get("ymax") is None else f"{float(range_data['ymax']):.12g}")
        included = self._calibration_range_mask(gmd, pd, range_data)
        self.calibration_axis.clear()
        if np.any(~included):
            self.calibration_axis.scatter(gmd[~included], pd[~included], s=22, color="#cbdcf4", alpha=0.9, label="_nolegend_")
        if np.any(included):
            self.calibration_axis.scatter(gmd[included], pd[included], s=22, color="#2563eb", alpha=0.78, label=f"data (n={np.count_nonzero(included)})")
        if np.any(included):
            # Draw an axes-clipped infinite line instead of limiting the fit
            # segment to the selected samples.  It therefore reaches both
            # plot borders and keeps doing so after pan/zoom or range changes.
            fit_line = self.calibration_axis.axline(
                (0.0, float(result["intercept"])), slope=float(result["slope"]),
                color="red", linewidth=1.8,
                label=(
                    f"PD = {float(result['slope']):.6g}×GMD"
                    f"{float(result['intercept']):+.6g}\n"
                    f"offset {offset:+d}, R²={float(result['r_squared']):.6g}"
                ),
            )
            # Clip against a slightly inset axes rectangle so the line reaches
            # close to the plot frame without visually touching the border.
            inset = 0.018
            fit_line.set_clip_path(Rectangle(
                (inset, inset), 1.0 - 2 * inset, 1.0 - 2 * inset,
                transform=self.calibration_axis.transAxes,
            ))
        self.calibration_axis.set_xlabel("GMD (A)")
        self.calibration_axis.set_ylabel("PD = -mso:Area (V·s)")
        self.calibration_axis.set_title(f"{Path(str(item.get('source', ''))).name} — offset {offset:+d}")
        self.calibration_axis.grid(alpha=0.25)
        self.calibration_axis.legend()
        preserved = self._preserved_fit_view
        if preserved is not None and preserved[0] == str(item.get("cache_id", "")) and preserved[1] == int(offset):
            self.calibration_axis.set_xlim(*preserved[2])
            self.calibration_axis.set_ylim(*preserved[3])
        self.calibration_canvas.draw_idle()
        self.fit_result_var.set(
            f"缓存 {Path(str(item.get('source', ''))).name}；当前查看偏移 {offset:+d}；"
            f"PD={float(result['slope']):.9g}×GMD{float(result['intercept']):+.9g}；R²={float(result['r_squared']):.7g}。"
        )

    @staticmethod
    def _calibration_range_mask(gmd: np.ndarray, pd: np.ndarray, limits: dict[str, Any]) -> np.ndarray:
        mask = np.isfinite(gmd) & np.isfinite(pd)
        if limits.get("xmin") is not None:
            mask &= gmd >= float(limits["xmin"])
        if limits.get("xmax") is not None:
            mask &= gmd <= float(limits["xmax"])
        if limits.get("ymin") is not None:
            mask &= pd >= float(limits["ymin"])
        if limits.get("ymax") is not None:
            mask &= pd <= float(limits["ymax"])
        return mask

    @staticmethod
    def _linear_fit(gmd: np.ndarray, pd: np.ndarray, offset: int) -> dict[str, Any]:
        if len(gmd) < 3 or np.ptp(gmd) == 0:
            raise ValueError("拟合范围内至少需要 3 个且 GMD 不完全相同的数据点。")
        design = np.column_stack([gmd, np.ones_like(gmd)])
        params, _, rank, _ = np.linalg.lstsq(design, pd, rcond=None)
        if rank < 2:
            raise ValueError("拟合范围内的 GMD 数据不足以确定斜率。")
        slope, intercept = map(float, params)
        residual = pd - design @ params
        ss_res = float(np.sum(residual**2))
        ss_tot = float(np.sum((pd - np.mean(pd)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot else float("nan")
        correlation = float(np.corrcoef(gmd, pd)[0, 1])
        dof = len(gmd) - 2
        denominator = float(np.sum((gmd - np.mean(gmd)) ** 2))
        slope_stderr = math.sqrt((ss_res / dof) / denominator) if dof > 0 and denominator > 0 else float("nan")
        return {
            "offset": int(offset), "slope": slope, "intercept": intercept, "r": correlation,
            "r_squared": r_squared, "slope_stderr": slope_stderr, "n": int(len(gmd)),
        }

    @staticmethod
    def _optional_limit(text: str) -> float | None:
        return None if not str(text).strip() else float(text)

    def apply_fit_range(self) -> None:
        item = self._selected_cache_item()
        if item is None:
            messagebox.showinfo("未选择缓存", "请先选择一条拟合缓存。", parent=self)
            return
        offset = int(item.get("selected_offset", item.get("best_offset", 0)))
        curve = next((entry for entry in item.get("curves", []) if int(entry.get("offset", 999999)) == offset), None)
        if curve is None:
            return
        previous_view = None
        if self.calibration_axis.collections or self.calibration_axis.lines:
            previous_view = (self.calibration_axis.get_xlim(), self.calibration_axis.get_ylim())
        try:
            limits = {
                "xmin": self._optional_limit(self.fit_xmin_var.get()), "xmax": self._optional_limit(self.fit_xmax_var.get()),
                "ymin": self._optional_limit(self.fit_ymin_var.get()), "ymax": self._optional_limit(self.fit_ymax_var.get()),
            }
            if limits["xmin"] is not None and limits["xmax"] is not None and limits["xmin"] > limits["xmax"]:
                raise ValueError("X min 不能大于 X max。")
            if limits["ymin"] is not None and limits["ymax"] is not None and limits["ymin"] > limits["ymax"]:
                raise ValueError("Y min 不能大于 Y max。")
            gmd = np.asarray(curve.get("gmd", []), dtype=float)
            pd = np.asarray(curve.get("pd", []), dtype=float)
            mask = self._calibration_range_mask(gmd, pd, limits)
            updated = self._linear_fit(gmd[mask], pd[mask], offset)
        except Exception as exc:
            messagebox.showerror("无法应用拟合范围", str(exc), parent=self)
            return
        item.setdefault("fit_ranges", {})[str(offset)] = limits
        item["selected_offset"] = offset
        item["normalized_slope"] = None
        item["normalized_slope_stderr"] = None
        results = item.setdefault("all_results", [])
        for index, result in enumerate(results):
            if int(result.get("offset", 999999)) == offset:
                results[index] = updated
                break
        else:
            results.append(updated)
        was_active_formula = (
            str(self.state.gmd_pd_fit.get("cache_id", "")) == str(item.get("cache_id", ""))
            and int(self.state.gmd_pd_fit.get("offset", 999999)) == offset
        )
        if was_active_formula:
            # The cache now contains a different fit, while formula constants
            # deliberately change only through the explicit "apply" action.
            self.state.gmd_pd_fit["cache_id"] = ""
        preserve_token = None
        if previous_view is not None:
            preserve_token = (str(item["cache_id"]), offset, previous_view[0], previous_view[1])
            self._preserved_fit_view = preserve_token
        self._refresh_calibration_cache_tree(select_id=str(item["cache_id"]))
        iid = f"{item['cache_id']}::offset::{offset}"
        if self.calibration_offsets_tree.exists(iid):
            self.calibration_offsets_tree.selection_set(iid)
        self._display_calibration_offset(item, offset)
        if preserve_token is not None:
            def release_preserved_view() -> None:
                if self._preserved_fit_view == preserve_token:
                    self._preserved_fit_view = None

            # Programmatic Treeview selections emit queued redraw events.  All
            # of them use the preserved view; release it after that short burst.
            self.after(100, release_preserved_view)
        self.state.notify("calibration_range_updated")
        suffix = "；原公式未自动改动，请重新点击‘应用于公式’" if was_active_formula else ""
        self.set_status(f"已用独立 XY 范围覆盖该缓存 offset {offset:+d} 的拟合参数{suffix}。")

    def apply_selected_calibration(self) -> None:
        item = self._selected_cache_item()
        if item is None:
            messagebox.showinfo("未选择缓存", "请先在缓存区选择一次拟合记录。", parent=self)
            return
        selection = self.calibration_offsets_tree.selection()
        offset = int(item.get("selected_offset", item.get("best_offset", 0)))
        if selection:
            try:
                offset = int(selection[0].rsplit("::", 1)[1])
            except (ValueError, IndexError):
                pass
        result = self._calibration_result(item, offset)
        if result is None:
            return
        if item.get("pprime") is None:
            messagebox.showinfo("尚未填写 Pprime", "请先双击缓存表中的 Pprime (Torr) 单元格填写标定气压。", parent=self)
            return
        try:
            coefficient = float(self.state.constants["k"])
            scale = math.exp(coefficient * float(item["pprime"]))
            normalized = float(result["slope"]) / scale
            normalized_stderr = float(result["slope_stderr"]) / abs(scale)
        except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            messagebox.showerror("无法计算标准化斜率", str(exc), parent=self)
            return
        if not math.isfinite(normalized):
            messagebox.showerror("无法计算标准化斜率", "计算结果不是有限数值。", parent=self)
            return
        item["normalized_slope"] = normalized
        item["normalized_slope_stderr"] = normalized_stderr if math.isfinite(normalized_stderr) else None
        self.state.constants["normalized_slope"] = normalized
        self.state.constants["sig_normalized_slope"] = normalized_stderr if math.isfinite(normalized_stderr) else 0.0
        self.state.constants["Pprime"] = float(item["pprime"])
        self.state.gmd_pd_fit = {
            "cache_id": item["cache_id"], "source": item.get("source", ""), "offset": offset,
            "slope": float(result["slope"]), "intercept": float(result["intercept"]),
            "r": float(result["r"]), "r_squared": float(result["r_squared"]),
            "slope_stderr": float(result["slope_stderr"]), "n": int(result["n"]),
            "pprime": float(item["pprime"]),
            "normalized_slope": normalized,
            "sig_normalized_slope": self.state.constants["sig_normalized_slope"],
            "all_results": item.get("all_results", []),
        }
        self._refresh_calibration_cache_tree(select_id=str(item["cache_id"]))
        self.state.notify("calibration_applied")
        self.set_status(
            f"标准化斜率 {normalized:.10g}、标准误差及 Pprime={float(item['pprime']):.7g} Torr "
            "已应用到第 3 步；拟合截距及 PD/GMD 背底均不参与计算。"
        )

    def delete_selected_calibration(self) -> None:
        selected_ids = set(self.calibration_cache_tree.selection())
        if not selected_ids and self.selected_calibration_id:
            selected_ids.add(self.selected_calibration_id)
        if not selected_ids:
            messagebox.showinfo("未选择缓存", "请先选择要删除的拟合记录。", parent=self)
            return
        if not messagebox.askyesno("删除拟合缓存", f"确定删除选中的 {len(selected_ids)} 条拟合缓存吗？", parent=self):
            return
        self.calibration_cache = [record for record in self.calibration_cache if str(record.get("cache_id", "")) not in selected_ids]
        if str(self.state.gmd_pd_fit.get("cache_id", "")) in selected_ids:
            self.state.gmd_pd_fit["cache_id"] = ""
        self.selected_calibration_id = ""
        self.calibration_offsets_tree.delete(*self.calibration_offsets_tree.get_children())
        self._empty_plot("选择其他缓存记录")
        self.calibration_canvas.draw_idle()
        self._refresh_calibration_cache_tree()
        self.state.notify("calibration_cache_deleted")

    def calculate_normalized_slopes(self) -> None:
        try:
            coefficient = float(self.state.constants["k"])
        except (KeyError, TypeError, ValueError):
            messagebox.showerror("缺少常数", "命名常数 k 无效，无法计算标准化斜率。", parent=self)
            return
        values = self._recalculate_normalized_slope_cache(coefficient)
        self.state.notify("normalized_slopes")
        if not values:
            messagebox.showinfo("没有可统计数据", "请先为至少一条缓存填写 Pprime。", parent=self)
            return
        self._show_normalized_slope_distribution(values, coefficient)

    def _recalculate_normalized_slope_cache(self, coefficient: float) -> list[float]:
        values: list[float] = []
        for item in self.calibration_cache:
            pprime = item.get("pprime")
            offset = int(item.get("selected_offset", item.get("best_offset", 0)))
            result = self._calibration_result(item, offset)
            if pprime is None or result is None:
                item["normalized_slope"] = None
                item["normalized_slope_stderr"] = None
                continue
            try:
                scale = math.exp(coefficient * float(pprime))
                normalized = float(result["slope"]) / scale
                normalized_stderr = float(result["slope_stderr"]) / abs(scale)
            except (OverflowError, ZeroDivisionError, TypeError, ValueError):
                item["normalized_slope"] = None
                item["normalized_slope_stderr"] = None
                continue
            item["normalized_slope"] = normalized if math.isfinite(normalized) else None
            item["normalized_slope_stderr"] = normalized_stderr if math.isfinite(normalized_stderr) else None
            if item["normalized_slope"] is not None:
                values.append(float(item["normalized_slope"]))
        self._refresh_calibration_cache_tree(select_id=self.selected_calibration_id or None)
        return values

    def _show_normalized_slope_distribution(self, values: list[float], coefficient: float) -> None:
        axis = self.normalized_axis
        axis.clear()
        bins = max(1, min(20, round(math.sqrt(len(values)))))
        axis.hist(values, bins=bins, color="#2563eb", edgecolor="white", alpha=0.82)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        axis.axvline(mean, color="#dc2626", linestyle="--", linewidth=1.8, label=f"均值={mean:.7g}")
        axis.set_title(f"标准化斜率分布（N={len(values)}，标准差={std:.6g}）")
        axis.set_xlabel("标准化斜率")
        axis.set_ylabel("频数")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        self.normalized_canvas.draw_idle()
        self.set_status(
            f"已统计 {len(values)} 条标准化斜率；当前 k={coefficient:.10g}，均值={mean:.10g}。"
        )

    def _selected_cache_pd_values(self, item: dict[str, Any]) -> np.ndarray:
        source = Path(str(item.get("source", "")))
        if source.is_file():
            return read_physical_pd(source)
        # A moved/renamed source can still be inspected from the saved cache.
        curves = [
            np.asarray(curve.get("pd", []), dtype=float)
            for curve in item.get("curves", [])
        ]
        curves = [values[np.isfinite(values)] for values in curves if len(values)]
        if not curves:
            raise FileNotFoundError(f"源文件不存在且缓存中没有 PD 数据：{source}")
        return max(curves, key=len)

    @staticmethod
    def pd_background_statistics(
        pd_values: np.ndarray, baseline: float, baseline_sigma: float = 0.0,
    ) -> dict[str, float | int]:
        values = np.asarray(pd_values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError("没有有限 PD 数据。")
        if not math.isfinite(baseline) or not math.isfinite(baseline_sigma) or baseline_sigma < 0:
            raise ValueError("baseline 必须有限，baseline 标准不确定度必须为非负数。")
        pd_mean = float(np.mean(values))
        sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        mean_sem = sample_std / math.sqrt(len(values))
        background_sigma = math.sqrt(mean_sem**2 + baseline_sigma**2)
        return {
            "count": int(len(values)), "pd_mean": pd_mean,
            "background": pd_mean - baseline, "sample_std": sample_std,
            "mean_sem": mean_sem, "baseline_sigma": baseline_sigma,
            "background_sigma": background_sigma,
        }

    def calculate_pd_background(self) -> None:
        item = self._selected_cache_item()
        if item is None:
            messagebox.showinfo("未选择缓存", "请先在上方缓存区选择一个 GMD–PD 文件。", parent=self)
            return
        try:
            baseline = float(self.pd_baseline_var.get().strip())
            baseline_sigma = float(self.pd_baseline_sigma_var.get().strip() or "0")
            pd_values = self._selected_cache_pd_values(item)
            statistics = self.pd_background_statistics(pd_values, baseline, baseline_sigma)
        except Exception as exc:
            messagebox.showerror("无法计算 PD 背底", str(exc), parent=self)
            return
        pd_mean = float(statistics["pd_mean"])
        background = float(statistics["background"])
        background_sigma = float(statistics["background_sigma"])
        corrected = pd_values - baseline
        self.last_pd_background = {
            "cache_id": str(item.get("cache_id", "")),
            "source": str(item.get("source", "")),
            "baseline": baseline,
            **statistics,
        }
        self.pd_background_result_var.set(
            f"N={len(pd_values)}；PD 平均值={pd_mean:.12g} V·s；"
            f"样本标准差={float(statistics['sample_std']):.6g} V·s；均值标准误={float(statistics['mean_sem']):.6g} V·s。\n"
            f"PD 背底={background:.12g} ± {background_sigma:.4g} V·s (1σ)。"
        )
        axis = self.background_axis
        axis.clear()
        axis.plot(
            np.arange(1, len(corrected) + 1), corrected,
            color="#2563eb", linewidth=1.0, alpha=0.8, label="PD − baseline",
        )
        axis.axhline(
            background, color="#dc2626", linestyle="--", linewidth=1.8,
            label=f"背底={background:.7g} ± {background_sigma:.3g} V·s",
        )
        if background_sigma > 0:
            axis.axhspan(
                background - background_sigma, background + background_sigma,
                color="#dc2626", alpha=0.12, label="背底 ±1σ",
            )
        axis.set_title(Path(str(item.get("source", ""))).name)
        axis.set_xlabel("采样序号")
        axis.set_ylabel("PD − baseline (V·s)")
        axis.grid(alpha=0.25)
        axis.legend()
        self.background_canvas.draw_idle()
        self.set_status("PD 背底及其标准不确定度已计算；确认后可一并应用到第 3 步。")

    def apply_pd_background(self) -> None:
        if not self.last_pd_background:
            messagebox.showinfo("尚未计算", "请先用当前选中的缓存计算 PD 背底。", parent=self)
            return
        background = float(self.last_pd_background["background"])
        background_sigma = float(self.last_pd_background.get("background_sigma", 0.0))
        self.state.constants["intercept"] = background
        self.state.constants["sigPDBackground"] = background_sigma
        self.state.notify("pd_background_applied")
        self.set_status(
            f"PD 背底 {background:.12g} V·s 及 sigPDBackground={background_sigma:.6g} V·s "
            "已应用到第 3 步命名常数。"
        )

    def _empty_plot(self, text: str) -> None:
        self.calibration_axis.clear()
        self.calibration_axis.set_xlabel("GMD (A)")
        self.calibration_axis.set_ylabel("PD = -mso:Area (V·s)")
        self.calibration_axis.text(0.5, 0.5, text, ha="center", va="center", transform=self.calibration_axis.transAxes)

    def export_config(self) -> dict[str, Any]:
        self._store_gas_attenuation_sheet()
        return {
            "calibration_folder": self.calibration_folder_var.get(),
            "single_calibration_file": self.single_calibration_file_var.get(),
            "offsets": self.offsets_var.get(),
            "calibration_cache": self.calibration_cache,
            "selected_calibration_id": self.selected_calibration_id,
            "pd_baseline": self.pd_baseline_var.get(),
            "pd_baseline_sigma": self.pd_baseline_sigma_var.get(),
            "last_pd_background": self.last_pd_background,
            "gas_attenuation_data": self.gas_attenuation_data,
            "gas_attenuation_fit": self.gas_attenuation_fit,
        }

    def import_config(self, data: dict[str, Any]) -> None:
        self._close_analysis_windows()
        folder = str(data.get("calibration_folder", ""))
        if not folder and data.get("calibration_file"):
            folder = str(Path(str(data["calibration_file"])).parent)
        self.calibration_folder_var.set(folder)
        self.single_calibration_file_var.set(str(data.get("single_calibration_file", "")))
        self.offsets_var.set(data.get("offsets", "0, ±1, ±2"))
        self.pd_baseline_var.set(str(data.get("pd_baseline", "0")))
        self.pd_baseline_sigma_var.set(str(data.get("pd_baseline_sigma", "0")))
        self.last_pd_background = dict(data.get("last_pd_background", {}))
        self.pd_background_file_var.set("请先在上方缓存区选择一个文件")
        self.pd_background_result_var.set("尚未计算 PD 背底。")
        self.gas_attenuation_data = [list(row[:2]) for row in data.get("gas_attenuation_data", [["", ""] for _ in range(12)])]
        self.gas_attenuation_fit = dict(data.get("gas_attenuation_fit", {}))
        self.gas_attenuation_result_var.set("尚未拟合气体衰减常数。")
        if self.gas_attenuation_fit:
            try:
                self.gas_attenuation_result_var.set(
                    f"上次拟合：k={float(self.gas_attenuation_fit['k']):.10g} ± "
                    f"{float(self.gas_attenuation_fit['sigma_k']):.4g}；A={float(self.gas_attenuation_fit['A']):.10g}"
                )
            except (KeyError, TypeError, ValueError):
                self.gas_attenuation_fit = {}
        self.calibration_cache = [dict(item) for item in data.get("calibration_cache", [])]
        active_cache = str(self.state.gmd_pd_fit.get("cache_id", ""))
        legacy_active_assigned = False
        seen_cache_ids: set[str] = set()
        import_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for item_index, item in enumerate(self.calibration_cache, 1):
            cache_id = str(item.get("cache_id", ""))
            if not cache_id or cache_id in seen_cache_ids:
                cache_id = f"cal_import_{import_stamp}_{item_index}"
                item["cache_id"] = cache_id
            seen_cache_ids.add(cache_id)
            item.setdefault("selected_offset", int(item.get("best_offset", 0)))
            if not item.get("file_mtime"):
                try:
                    item["file_mtime"] = datetime.fromtimestamp(
                        Path(str(item.get("source", ""))).stat().st_mtime
                    ).isoformat(timespec="seconds")
                except OSError:
                    item["file_mtime"] = ""
            item.setdefault("pd_minimum", None)
            item.setdefault("pd_min_count", 0)
            item.setdefault("pprime", None)
            item.setdefault("normalized_slope", None)
            item.setdefault("normalized_slope_stderr", None)
            item.setdefault("fit_ranges", {})
            is_active = bool(active_cache) and str(item.get("cache_id", "")) == active_cache
            # Very old projects stored the applied fit without its cache_id.
            # Match its numerical fit once so the newly introduced Pprime cell
            # is still populated from the project's formula constants.
            if not active_cache and not legacy_active_assigned and self.state.gmd_pd_fit:
                fit_offset = int(self.state.gmd_pd_fit.get("offset", item["selected_offset"]))
                result = self._calibration_result(item, fit_offset)
                if result is not None:
                    try:
                        is_active = math.isclose(float(result["slope"]), float(self.state.gmd_pd_fit["slope"]), rel_tol=1e-10, abs_tol=1e-18)
                        is_active = is_active and math.isclose(float(result["intercept"]), float(self.state.gmd_pd_fit["intercept"]), rel_tol=1e-10, abs_tol=1e-18)
                    except (KeyError, TypeError, ValueError):
                        is_active = False
            if is_active and item.get("pprime") is None:
                item["pprime"] = self.state.gmd_pd_fit.get("pprime", self.state.constants.get("Pprime"))
                legacy_active_assigned = True
            if is_active and item.get("normalized_slope") is None:
                item["normalized_slope"] = self.state.gmd_pd_fit.get(
                    "normalized_slope", self.state.constants.get("normalized_slope"),
                )
            if item.get("normalized_slope") is not None and item.get("normalized_slope_stderr") is None and item.get("pprime") is not None:
                result = self._calibration_result(
                    item, int(item.get("selected_offset", item.get("best_offset", 0))),
                )
                if result is not None:
                    try:
                        scale = math.exp(float(self.state.constants["k"]) * float(item["pprime"]))
                        stderr = float(result["slope_stderr"]) / abs(scale)
                        item["normalized_slope_stderr"] = stderr if math.isfinite(stderr) else None
                    except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
                        pass
        self._cache_sort_column = "file"
        self._cache_sort_reverse = False
        self._sort_calibration_cache_in_place()
        self._update_cache_sort_headings()
        self.selected_calibration_id = str(data.get("selected_calibration_id", ""))
        self.calibration_offsets_tree.delete(*self.calibration_offsets_tree.get_children())
        self._empty_plot("拟合后在左侧缓存区选择记录")
        self.calibration_canvas.draw_idle()
        fit = self.state.gmd_pd_fit
        if fit:
            self.fit_result_var.set(
                f"当前公式：偏移 {int(fit['offset']):+d}；PD={fit['slope']:.8g}×GMD{fit['intercept']:+.8g}；R²={fit['r_squared']:.6f}。"
            )
        else:
            self.fit_result_var.set("尚未拟合 GMD–PD。")
        self._refresh_calibration_cache_tree(select_id=self.selected_calibration_id or None)
        if self.last_pd_background:
            try:
                self.pd_background_result_var.set(
                    f"上次结果：N={int(self.last_pd_background.get('count', 0))}；"
                    f"PD 背底={float(self.last_pd_background['background']):.12g} ± "
                    f"{float(self.last_pd_background.get('background_sigma', 0.0)):.4g} V·s。"
                )
            except (KeyError, TypeError, ValueError):
                self.last_pd_background = {}
