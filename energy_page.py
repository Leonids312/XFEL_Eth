"""Step 3: damage GMD extraction and formula-driven energy tables."""

from __future__ import annotations

import math
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tksheet import Sheet

from remembering_toolbar import RememberingNavigationToolbar

from xfel_core import (
    COLUMN_NAMES,
    NAME_TO_COLUMN,
    ProjectState,
    RecentPathStore,
    align_damage_gmd,
    apply_formula,
    attenuation_sort_key,
    fit_gmd_pd,
    parse_offsets,
)


HEADINGS = {
    "serial": "序号",
    "gmd": "GMD 值 (A)",
    "pd": "PD 值 (V·s)",
    "energy": "单脉冲能量 (nJ)",
    "area": "损伤面积",
    "energy_rel_err": "能量相对误差",
    "area_rel_err": "面积相对误差",
}

DEFAULT_FORMULAS = {
    "pd": "=normalized_slope*exp(k*P)*gmd",
    "energy": "=pd/load_resistance/responsivity*(12398.419843/wavelength_A)*1e9",
    "energy_rel_err": "=sqrt(gmd_rel_err^2+(sig_normalized_slope/normalized_slope)^2+((P-Pprime)*sigk)^2+(sigeta/eta)^2)",
    "area_rel_err": "=area_rel_default",
}


def format_number(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.10g}" if math.isfinite(number) else ""


class EnergyExtractionPage(ttk.Frame):
    def __init__(self, master: tk.Misc, state: ProjectState, status_callback=None, path_store: RecentPathStore | None = None) -> None:
        super().__init__(master)
        self.state = state
        self.status_callback = status_callback or (lambda text: None)
        self.path_store = path_store
        self.tables: dict[str, Sheet] = {}
        self.table_frames: dict[str, ttk.Frame] = {}
        self.group_pressures: dict[str, float] = {}
        self.calibration_cache: list[dict[str, Any]] = []
        self.selected_calibration_id = ""
        self._refresh_pending = False
        self._sheet_sync_pending: set[str] = set()
        self.calibration_file_var = tk.StringVar()
        self.offsets_var = tk.StringVar(value="0, ±1, ±2")
        self.fit_result_var = tk.StringVar(value="尚未拟合 GMD–PD。")
        self.group_var = tk.StringVar()
        self.range_start_var = tk.IntVar(value=1)
        self.range_end_var = tk.IntVar(value=30)
        self.damage_h5_var = tk.StringVar()
        self.shift_min_var = tk.IntVar(value=-10)
        self.shift_max_var = tk.IntVar(value=10)
        self.alignment_result_var = tk.StringVar(value="选择衰减组、序号范围和损伤 HDF5 文件。")
        self.pd_slope_source_var = tk.StringVar(value="closest_calibration")
        self.pd_slope_source_result_var = tk.StringVar(value="自动模式将按 HDF5 修改时间选择最近的第 2 步标准化斜率。")
        self.calibration_cache_provider = lambda: []
        self.formula_column_var = tk.StringVar(value="pd")
        self.formula_var = tk.StringVar(value=DEFAULT_FORMULAS["pd"])
        self.formula_help_var = tk.StringVar()
        self.pressure_var = tk.DoubleVar(value=0.0)
        self._build_ui()
        self.state.subscribe(self._state_changed)
        self.refresh_groups()

    def _build_ui(self) -> None:
        outer = ttk.Panedwindow(self, orient=tk.VERTICAL)
        outer.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(outer, padding=7)
        data_area = ttk.Frame(outer, padding=(7, 2, 7, 7))
        outer.add(controls, weight=0)
        outer.add(data_area, weight=1)

        alignment = ttk.LabelFrame(controls, text="损伤 HDF5 时间戳匹配", padding=7)
        alignment.grid(row=0, column=0, sticky=tk.NSEW)
        ttk.Label(alignment, text="衰减倍数").grid(row=0, column=0, sticky=tk.W)
        self.group_combo = ttk.Combobox(alignment, textvariable=self.group_var, state="readonly", width=11)
        self.group_combo.grid(row=0, column=1, sticky=tk.W, padx=(4, 8))
        self.group_combo.bind("<<ComboboxSelected>>", self._group_selected)
        ttk.Label(alignment, text="序号").grid(row=0, column=2)
        ttk.Spinbox(alignment, textvariable=self.range_start_var, from_=1, to=100000, width=7).grid(row=0, column=3, padx=3)
        ttk.Label(alignment, text="—").grid(row=0, column=4)
        ttk.Spinbox(alignment, textvariable=self.range_end_var, from_=1, to=100000, width=7).grid(row=0, column=5, padx=3)
        ttk.Label(alignment, text="偏移").grid(row=0, column=6, padx=(8, 2))
        ttk.Spinbox(alignment, textvariable=self.shift_min_var, from_=-100, to=100, width=5).grid(row=0, column=7)
        ttk.Label(alignment, text="—").grid(row=0, column=8)
        ttk.Spinbox(alignment, textvariable=self.shift_max_var, from_=-100, to=100, width=5).grid(row=0, column=9)
        ttk.Entry(alignment, textvariable=self.damage_h5_var, width=55).grid(row=1, column=0, columnspan=8, sticky=tk.EW, pady=(5, 0))
        ttk.Button(alignment, text="选择损伤 HDF5", command=self.choose_damage_file).grid(row=1, column=8, columnspan=2, padx=(4, 0), pady=(5, 0))
        ttk.Button(alignment, text="自动匹配并填入 GMD", command=self.run_alignment).grid(row=2, column=8, columnspan=2, sticky=tk.EW, padx=(4, 0), pady=(5, 0))
        ttk.Label(alignment, textvariable=self.alignment_result_var, wraplength=720, foreground="#334155").grid(row=2, column=0, columnspan=8, sticky=tk.W, pady=(5, 0))

        slope_source = ttk.LabelFrame(alignment, text="自动计算 PD 时的标准化斜率来源", padding=(7, 5))
        slope_source.grid(row=3, column=0, columnspan=10, sticky=tk.EW, pady=(7, 0))
        ttk.Radiobutton(
            slope_source, text="自动选择：采用与损伤 HDF5 修改时间最接近的第 2 步缓存",
            variable=self.pd_slope_source_var, value="closest_calibration",
            command=self._update_slope_source_preview,
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(
            slope_source, text="命名常数：采用 normalized_slope",
            variable=self.pd_slope_source_var, value="named_constant",
            command=self._update_slope_source_preview,
        ).grid(row=0, column=1, sticky=tk.W, padx=(18, 0))
        ttk.Label(
            slope_source, textvariable=self.pd_slope_source_result_var,
            foreground="#475569", wraplength=1350,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

        formula = ttk.LabelFrame(controls, text="列公式（Excel / Origin 风格）", padding=7)
        formula.grid(row=1, column=0, sticky=tk.EW, pady=(6, 0))
        ttk.Label(formula, text="当前组气体压力 P").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(formula, textvariable=self.pressure_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=(4, 10))
        ttk.Label(formula, text="目标列").grid(row=0, column=2)
        target = ttk.Combobox(formula, textvariable=self.formula_column_var, state="readonly", width=17,
                              values=("gmd", "pd", "energy", "area", "energy_rel_err", "area_rel_err"))
        target.grid(row=0, column=3, padx=4)
        target.bind("<<ComboboxSelected>>", self._formula_column_selected)
        ttk.Entry(formula, textvariable=self.formula_var, width=95).grid(row=0, column=4, sticky=tk.EW, padx=4)
        ttk.Button(formula, text="应用到当前组整列", command=self.apply_current_formula).grid(row=0, column=5, padx=3)
        ttk.Button(formula, text="重算当前组全部公式", command=self.recalculate_group).grid(row=0, column=6, padx=3)
        ttk.Button(formula, text="命名常数…", command=self.open_constants_dialog).grid(row=0, column=7, padx=3)
        ttk.Label(
            formula,
            text="示例：=normalized_slope*exp(k*P)*col(B)    或 =C2/load_resistance/responsivity*(12398.419843/wavelength_A)*1e9。PD/GMD 背底均不参与计算。列字母：A序号 B:GMD C:PD D:能量 E:面积 F/G:相对误差。",
            wraplength=1450, foreground="#475569",
        ).grid(row=1, column=0, columnspan=8, sticky=tk.W, pady=(5, 0))
        formula.columnconfigure(4, weight=1)
        controls.columnconfigure(0, weight=1)

        table_toolbar = ttk.Frame(data_area)
        table_toolbar.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(table_toolbar, text="新增衰减组/序号", command=self.add_group_dialog).pack(side=tk.LEFT)
        ttk.Button(table_toolbar, text="删除当前衰减组", command=self.delete_current_group).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(table_toolbar, text="导出全部数据到 Excel", command=self.export_excel).pack(side=tk.LEFT, padx=5)
        ttk.Label(table_toolbar, text="鼠标拖选矩阵后 Ctrl+C；从 Excel/Origin 复制矩阵后 Ctrl+V。序号列可复制但不可修改。", foreground="#475569").pack(side=tk.LEFT, padx=10)
        self.group_notebook = ttk.Notebook(data_area)
        self.group_notebook.pack(fill=tk.BOTH, expand=True)
        self.group_notebook.bind("<<NotebookTabChanged>>", self._tab_changed)

    def _build_calibration_page(self, page: ttk.Frame) -> None:
        toolbar = ttk.LabelFrame(page, text="读取标定文件并扫描 GMD–PD 序号偏移", padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Entry(toolbar, textvariable=self.calibration_file_var, width=80).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(toolbar, text="选择标定 HDF5", command=self.choose_calibration_file).grid(row=0, column=1, padx=5)
        ttk.Label(toolbar, text="尝试偏移").grid(row=0, column=2, padx=(12, 3))
        ttk.Entry(toolbar, textvariable=self.offsets_var, width=18).grid(row=0, column=3)
        ttk.Button(toolbar, text="拟合并加入缓存", command=self.run_calibration).grid(row=0, column=4, padx=(6, 3))
        ttk.Button(toolbar, text="将选中缓存应用于公式", command=self.apply_selected_calibration).grid(row=0, column=5, padx=3)
        ttk.Button(toolbar, text="删除选中缓存", command=self.delete_selected_calibration).grid(row=0, column=6, padx=3)
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(toolbar, textvariable=self.fit_result_var, foreground="#334155", wraplength=1450).grid(row=1, column=0, columnspan=7, sticky=tk.W, pady=(7, 0))

        body = ttk.Panedwindow(page, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(7, 0))
        cache_panel = ttk.Frame(body, padding=(0, 0, 5, 0))
        detail_panel = ttk.Panedwindow(body, orient=tk.VERTICAL)
        body.add(cache_panel, weight=0)
        body.add(detail_panel, weight=1)

        ttk.Label(cache_panel, text="拟合缓存区", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W)
        self.calibration_cache_tree = ttk.Treeview(
            cache_panel,
            columns=("time", "file", "offset", "r2", "slope", "intercept", "n", "active"),
            show="headings",
            height=22,
            selectmode="browse",
        )
        cache_columns = (
            ("time", "时间", 105), ("file", "文件目录", 190), ("offset", "偏移", 48),
            ("r2", "R²", 65), ("slope", "斜率", 85), ("intercept", "截距", 88),
            ("n", "N", 42), ("active", "公式", 48),
        )
        for column, title, width in cache_columns:
            self.calibration_cache_tree.heading(column, text=title)
            self.calibration_cache_tree.column(column, width=width, anchor=tk.CENTER, stretch=column == "file")
        cache_y = ttk.Scrollbar(cache_panel, orient=tk.VERTICAL, command=self.calibration_cache_tree.yview)
        cache_x = ttk.Scrollbar(cache_panel, orient=tk.HORIZONTAL, command=self.calibration_cache_tree.xview)
        self.calibration_cache_tree.configure(yscrollcommand=cache_y.set, xscrollcommand=cache_x.set)
        cache_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.calibration_cache_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(5, 0))
        cache_y.pack(side=tk.RIGHT, fill=tk.Y, pady=(5, 0))
        self.calibration_cache_tree.bind("<<TreeviewSelect>>", self._on_calibration_cache_select)
        self.calibration_cache_tree.bind("<Delete>", lambda _event: self.delete_selected_calibration())

        # The placed Matplotlib canvas does not contribute a geometry request of
        # its own, so reserve enough height to keep labels clear without letting
        # the preview dominate the page.
        plot_frame = ttk.Frame(detail_panel, padding=8, height=340)
        plot_frame.pack_propagate(False)
        summary_frame = ttk.Frame(detail_panel, padding=(0, 6, 0, 0))
        detail_panel.add(plot_frame, weight=2)
        detail_panel.add(summary_frame, weight=1)
        self.calibration_figure = Figure(figsize=(6.4, 4.0), dpi=100)
        # Keep subplot parameters manually adjustable through Matplotlib's
        # "Configure subplots" tool.  Constrained layout would override them.
        self.calibration_figure.subplots_adjust(left=0.15, right=0.97, bottom=0.17, top=0.91)
        self.calibration_axis = self.calibration_figure.add_subplot(111)
        self.calibration_axis.set_xlabel("GMD (A)")
        self.calibration_axis.set_ylabel("PD = -mso:Area (V·s)")
        self.calibration_axis.text(0.5, 0.5, "拟合后在左侧缓存区选择记录", ha="center", va="center", transform=self.calibration_axis.transAxes)
        calibration_canvas_host = ttk.Frame(plot_frame)
        calibration_canvas_host.pack(fill=tk.BOTH, expand=True)
        self.calibration_canvas = FigureCanvasTkAgg(self.calibration_figure, master=calibration_canvas_host)
        calibration_toolbar = RememberingNavigationToolbar(
            self.calibration_canvas, plot_frame, path_store=self.path_store,
            path_key="energy_calibration_plot_save", pack_toolbar=False,
        )
        calibration_toolbar.update()
        calibration_toolbar.pack(fill=tk.X, before=calibration_canvas_host)
        calibration_widget = self.calibration_canvas.get_tk_widget()
        calibration_widget.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        def resize_calibration_preview(event) -> None:
            width = max(160, min(event.width - 12, 860))
            height = max(140, min(event.height - 12, 500))
            aspect = 1.6
            if width / height > aspect:
                width = round(height * aspect)
            else:
                height = round(width / aspect)
            calibration_widget.configure(width=width, height=height)

        calibration_canvas_host.bind("<Configure>", resize_calibration_preview)
        self.calibration_canvas.draw()

        ttk.Label(summary_frame, text="本次拟合的全部偏移结果", font=("Microsoft YaHei UI", 10, "bold")).pack(anchor=tk.W)
        self.calibration_offsets_tree = ttk.Treeview(
            summary_frame,
            columns=("offset", "n", "slope", "intercept", "r", "r2", "stderr", "state"),
            show="headings",
            height=6,
        )
        for column, title, width in (
            ("offset", "偏移", 60), ("n", "N", 60), ("slope", "斜率", 125),
            ("intercept", "截距", 125), ("r", "r", 90), ("r2", "R²", 90), ("stderr", "斜率标准误", 125),
            ("state", "状态", 95),
        ):
            self.calibration_offsets_tree.heading(column, text=title)
            self.calibration_offsets_tree.column(column, width=width, anchor=tk.CENTER)
        self.calibration_offsets_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def set_status(self, text: str) -> None:
        self.status_callback(text)

    def set_calibration_cache_provider(self, provider) -> None:
        """Connect Step 3 to the live Step 2 cache without duplicating it."""
        self.calibration_cache_provider = provider
        self._update_slope_source_preview()

    def _live_calibration_cache(self) -> list[dict[str, Any]]:
        try:
            return list(self.calibration_cache_provider())
        except Exception:
            return []

    @staticmethod
    def _cache_file_mtime(item: dict[str, Any]) -> float | None:
        stored = str(item.get("file_mtime", "")).strip()
        if stored:
            try:
                return datetime.fromisoformat(stored.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        source = Path(str(item.get("source", "")))
        try:
            return float(source.stat().st_mtime)
        except OSError:
            return None

    @classmethod
    def nearest_normalized_slope(
        cls, damage_path: str | Path, cache: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return the eligible Step 2 cache closest in HDF5 modification time."""
        damage = Path(damage_path)
        damage_mtime = float(damage.stat().st_mtime)
        candidates: list[tuple[float, str, dict[str, Any], float, float, float, float]] = []
        for item in cache:
            try:
                slope = float(item.get("normalized_slope"))
                slope_stderr = float(item.get("normalized_slope_stderr"))
                pprime = float(item.get("pprime"))
            except (TypeError, ValueError):
                continue
            calibration_mtime = cls._cache_file_mtime(item)
            if (
                calibration_mtime is None
                or not all(math.isfinite(value) for value in (slope, slope_stderr, pprime))
                or slope_stderr < 0
            ):
                continue
            delta = abs(calibration_mtime - damage_mtime)
            candidates.append((
                delta, str(item.get("source", "")).casefold(), item,
                slope, slope_stderr, pprime, calibration_mtime,
            ))
        if not candidates:
            raise ValueError(
                "第 2 步缓存中没有同时具备标准化斜率、标准误差、Pprime 和文件修改时间的记录。"
            )
        delta, _source_key, item, slope, slope_stderr, pprime, calibration_mtime = min(
            candidates, key=lambda entry: (entry[0], entry[1]),
        )
        return {
            "item": item,
            "slope": slope,
            "slope_stderr": slope_stderr,
            "pprime": pprime,
            "damage_mtime": damage_mtime,
            "calibration_mtime": calibration_mtime,
            "time_difference": delta,
        }

    @staticmethod
    def _short_source(source: str) -> str:
        path = Path(source)
        return str(Path(*path.parts[-3:])) if len(path.parts) >= 3 else str(path)

    def _resolve_pd_slope(self, damage_path: str | Path) -> tuple[float, str, dict[str, Any]]:
        if self.pd_slope_source_var.get() == "named_constant":
            try:
                slope = float(self.state.constants["normalized_slope"])
                slope_stderr = float(self.state.constants["sig_normalized_slope"])
                pprime = float(self.state.constants["Pprime"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("命名常数 normalized_slope、sig_normalized_slope 或 Pprime 尚未填写有效数值。") from None
            if not all(math.isfinite(value) for value in (slope, slope_stderr, pprime)) or slope_stderr < 0:
                raise ValueError("命名常数中的标准化斜率、标准误差和 Pprime 必须有效。")
            return slope, (
                f"命名常数：normalized_slope={slope:.10g}；"
                f"sig_normalized_slope={slope_stderr:.6g}；Pprime={pprime:.7g} Torr"
            ), {"mode": "named_constant"}

        match = self.nearest_normalized_slope(damage_path, self._live_calibration_cache())
        item = match["item"]
        difference = float(match["time_difference"])
        description = (
            f"第 2 步最近缓存：{self._short_source(str(item.get('source', '')))}；"
            f"标准化斜率={float(match['slope']):.10g}；"
            f"标准误差={float(match['slope_stderr']):.6g}；"
            f"Pprime={float(match['pprime']):.7g} Torr；修改时间差={difference:.3f} s"
        )
        return float(match["slope"]), description, {
            "mode": "closest_calibration",
            "cache_id": str(item.get("cache_id", "")),
            "calibration_source": str(item.get("source", "")),
            "calibration_mtime": float(match["calibration_mtime"]),
            "damage_mtime": float(match["damage_mtime"]),
            "time_difference": difference,
            "sigma_normalized_slope": match.get("slope_stderr"),
            "pprime": match.get("pprime"),
        }

    def _update_slope_source_preview(self) -> None:
        path = self.damage_h5_var.get().strip()
        if self.pd_slope_source_var.get() == "named_constant":
            try:
                _slope, description, _metadata = self._resolve_pd_slope(path or ".")
                self.pd_slope_source_result_var.set(description)
            except Exception as exc:
                self.pd_slope_source_result_var.set(str(exc))
            return
        if not path or not Path(path).is_file():
            self.pd_slope_source_result_var.set("选择损伤 HDF5 后，将预览修改时间最接近的第 2 步缓存。")
            return
        try:
            _slope, description, _metadata = self._resolve_pd_slope(path)
            self.pd_slope_source_result_var.set(description)
        except Exception as exc:
            self.pd_slope_source_result_var.set(str(exc))

    def _state_changed(self, reason: str) -> None:
        if reason == "table_matrix_edit":
            return
        if self._refresh_pending:
            return
        self._refresh_pending = True
        self.after_idle(self._refresh_from_state)

    def _refresh_from_state(self) -> None:
        self._refresh_pending = False
        self.refresh_groups()
        self._update_slope_source_preview()

    def choose_calibration_file(self) -> None:
        fallback = Path(self.calibration_file_var.get()).parent if self.calibration_file_var.get() else Path.cwd()
        initial_dir = self.path_store.initial_dir("energy_calibration_hdf5_open", fallback) if self.path_store else str(fallback)
        filename = filedialog.askopenfilename(parent=self, initialdir=initial_dir, title="选择含 GMD 与 PD 的 HDF5", filetypes=[("HDF5", "*.hdf5 *.h5"), ("所有文件", "*.*")])
        if filename:
            self.calibration_file_var.set(filename)
            if self.path_store:
                self.path_store.remember_file("energy_calibration_hdf5_open", filename)

    def choose_damage_file(self) -> None:
        fallback = Path(self.damage_h5_var.get()).parent if self.damage_h5_var.get() else Path.cwd()
        initial_dir = self.path_store.initial_dir("energy_damage_hdf5_open", fallback) if self.path_store else str(fallback)
        filename = filedialog.askopenfilename(parent=self, initialdir=initial_dir, title="选择打损伤时的 HDF5", filetypes=[("HDF5", "*.hdf5 *.h5"), ("所有文件", "*.*")])
        if filename:
            self.damage_h5_var.set(filename)
            if self.path_store:
                self.path_store.remember_file("energy_damage_hdf5_open", filename)
            self._update_slope_source_preview()

    def run_calibration(self) -> None:
        path = self.calibration_file_var.get().strip()
        if not path:
            self.choose_calibration_file()
            path = self.calibration_file_var.get().strip()
        if not path:
            return
        try:
            requested_offsets = parse_offsets(self.offsets_var.get())
            result = fit_gmd_pd(path, requested_offsets)
        except Exception as exc:
            messagebox.showerror("GMD–PD 拟合失败", str(exc), parent=self)
            return
        if self.path_store:
            self.path_store.remember_file("energy_calibration_hdf5_open", path)
        created_at = datetime.now().isoformat(timespec="seconds")
        cache_id = f"cal_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{len(self.calibration_cache) + 1}"
        cache_item = {
            "cache_id": cache_id,
            "created_at": created_at,
            "source": path,
            "requested_offsets": requested_offsets,
            "best_offset": result.offset,
            "all_results": result.all_results,
            "curves": [
                {**{key: value for key, value in curve.items() if key not in {"gmd", "pd"}},
                 "gmd": np.asarray(curve["gmd"], dtype=float).tolist(),
                 "pd": np.asarray(curve["pd"], dtype=float).tolist()}
                for curve in result.curves
            ],
        }
        self.calibration_cache.append(cache_item)
        self._refresh_calibration_cache_tree(select_id=cache_id)
        self.state.notify("calibration_cache")
        self.fit_result_var.set(
            f"已加入缓存：{Path(path).name}；最优偏移 {result.offset:+d}，R²={result.r_squared:.6f}。"
            "如需用于 PD/能量公式，请点击“将选中缓存应用于公式”。"
        )
        self.set_status("GMD–PD 拟合完成并已加入缓存；尚未改变当前公式常数。")

    def _refresh_calibration_cache_tree(self, select_id: str | None = None) -> None:
        if not hasattr(self, "calibration_cache_tree"):
            return
        active_id = str(self.state.gmd_pd_fit.get("cache_id", ""))
        self.calibration_cache_tree.delete(*self.calibration_cache_tree.get_children())
        for item in self.calibration_cache:
            best = self._calibration_result(item, int(item.get("best_offset", 0)))
            if best is None:
                continue
            cache_id = str(item["cache_id"])
            created = str(item.get("created_at", "")).replace("T", " ")
            self.calibration_cache_tree.insert(
                "", tk.END, iid=cache_id,
                values=(created, self._cache_source_label(str(item.get("source", ""))), f"{int(best['offset']):+d}",
                        f"{float(best['r_squared']):.6g}", f"{float(best['slope']):.7g}",
                        f"{float(best['intercept']):.7g}", int(best["n"]), "当前" if cache_id == active_id else ""),
            )
        target = select_id or self.selected_calibration_id or active_id
        if target and self.calibration_cache_tree.exists(target):
            self.calibration_cache_tree.selection_set(target)
            self.calibration_cache_tree.focus(target)
            self.calibration_cache_tree.see(target)
            self._display_calibration_cache(target)
            if target == active_id and "offset" in self.state.gmd_pd_fit:
                active_offset = int(self.state.gmd_pd_fit["offset"])
                active_offset_iid = f"{target}::offset::{active_offset}"
                item = self._selected_cache_item()
                if item is not None and self.calibration_offsets_tree.exists(active_offset_iid):
                    self.calibration_offsets_tree.selection_set(active_offset_iid)
                    self._display_calibration_offset(item, active_offset)

    @staticmethod
    def _cache_source_label(source: str) -> str:
        path = Path(source)
        parts = path.parts
        if len(parts) >= 3:
            return str(Path(*parts[-3:]))
        return str(path)

    def delete_selected_calibration(self) -> None:
        item = self._selected_cache_item()
        if item is None:
            messagebox.showinfo("未选择缓存", "请先在缓存区选择要删除的拟合记录。", parent=self)
            return
        source_label = self._cache_source_label(str(item.get("source", "")))
        if not messagebox.askyesno("删除拟合缓存", f"确定删除这次拟合缓存吗？\n\n{source_label}\n{item.get('created_at', '')}", parent=self):
            return
        cache_id = str(item.get("cache_id", ""))
        self.calibration_cache = [record for record in self.calibration_cache if str(record.get("cache_id", "")) != cache_id]
        if str(self.state.gmd_pd_fit.get("cache_id", "")) == cache_id:
            self.state.gmd_pd_fit["cache_id"] = ""
        self.selected_calibration_id = ""
        self.calibration_offsets_tree.delete(*self.calibration_offsets_tree.get_children())
        self.calibration_axis.clear()
        self.calibration_axis.set_xlabel("GMD (A)")
        self.calibration_axis.set_ylabel("PD = -mso:Area (V·s)")
        self.calibration_axis.text(0.5, 0.5, "选择其他缓存记录", ha="center", va="center", transform=self.calibration_axis.transAxes)
        self.calibration_canvas.draw_idle()
        self._refresh_calibration_cache_tree()
        self.state.notify("calibration_cache_deleted")
        self.fit_result_var.set("选中的拟合缓存已删除；已经写入公式的数值常数保持不变。")
        self.set_status("GMD–PD 拟合缓存已删除。")

    @staticmethod
    def _calibration_result(item: dict[str, Any], offset: int) -> dict[str, Any] | None:
        return next((result for result in item.get("all_results", []) if int(result.get("offset", 999999)) == int(offset)), None)

    def _selected_cache_item(self) -> dict[str, Any] | None:
        cache_id = self.selected_calibration_id
        return next((item for item in self.calibration_cache if str(item.get("cache_id")) == cache_id), None)

    def _on_calibration_cache_select(self, _event=None) -> None:
        selection = self.calibration_cache_tree.selection()
        if selection:
            self._display_calibration_cache(selection[0])

    def _display_calibration_cache(self, cache_id: str) -> None:
        item = next((record for record in self.calibration_cache if str(record.get("cache_id")) == str(cache_id)), None)
        if item is None:
            return
        self.selected_calibration_id = str(cache_id)
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
        self.calibration_offsets_tree.bind("<<TreeviewSelect>>", self._on_calibration_offset_select)
        default_offset = int(active_offset) if str(cache_id) == active_cache and active_offset is not None else best_offset
        default_iid = f"{cache_id}::offset::{default_offset}"
        if self.calibration_offsets_tree.exists(default_iid):
            self.calibration_offsets_tree.selection_set(default_iid)
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
        self.calibration_axis.clear()
        self.calibration_axis.scatter(gmd, pd, s=22, alpha=0.72, label=f"{len(gmd)} points")
        if len(gmd):
            xline = np.linspace(float(np.min(gmd)), float(np.max(gmd)), 300)
            self.calibration_axis.plot(xline, float(result["slope"]) * xline + float(result["intercept"]), color="red", linewidth=1.8,
                                       label=f"offset {offset:+d}, R²={float(result['r_squared']):.6g}")
        self.calibration_axis.set_xlabel("GMD (A)")
        self.calibration_axis.set_ylabel("PD = -mso:Area (V·s)")
        self.calibration_axis.set_title(f"{Path(str(item.get('source', ''))).name} — offset {offset:+d}")
        self.calibration_axis.grid(alpha=0.25)
        self.calibration_axis.legend()
        self.calibration_canvas.draw_idle()
        is_best = offset == int(item.get("best_offset", 0))
        self.fit_result_var.set(
            f"缓存 {Path(str(item.get('source', ''))).name}；当前查看偏移 {offset:+d}{'（本次最优）' if is_best else ''}；"
            f"PD={float(result['slope']):.9g}×GMD{float(result['intercept']):+.9g}；R²={float(result['r_squared']):.7g}。"
        )

    def apply_selected_calibration(self) -> None:
        item = self._selected_cache_item()
        if item is None:
            messagebox.showinfo("未选择缓存", "请先在左侧缓存区选择一次拟合记录。", parent=self)
            return
        selection = self.calibration_offsets_tree.selection()
        offset = int(item.get("best_offset", 0))
        if selection:
            try:
                offset = int(selection[0].rsplit("::", 1)[1])
            except (ValueError, IndexError):
                pass
        result = self._calibration_result(item, offset)
        if result is None:
            return
        self.state.constants["Bfit"] = float(result["slope"])
        self.state.constants["sigBfit"] = float(result["slope_stderr"])
        self.state.gmd_pd_fit = {
            "cache_id": item["cache_id"], "source": item.get("source", ""), "offset": offset,
            "slope": float(result["slope"]), "intercept": float(result["intercept"]),
            "r": float(result["r"]), "r_squared": float(result["r_squared"]),
            "slope_stderr": float(result["slope_stderr"]), "n": int(result["n"]),
            "all_results": item.get("all_results", []),
        }
        self._refresh_calibration_cache_tree(select_id=str(item["cache_id"]))
        offset_iid = f"{item['cache_id']}::offset::{offset}"
        if self.calibration_offsets_tree.exists(offset_iid):
            self.calibration_offsets_tree.selection_set(offset_iid)
            self._display_calibration_offset(item, offset)
        self.state.notify("calibration_applied")
        self.fit_result_var.set(
            f"已应用：{Path(str(item.get('source', ''))).name}，偏移 {offset:+d}；"
            f"Bfit={float(result['slope']):.9g}，sigBfit={float(result['slope_stderr']):.6g}；拟合截距仅供检查，不参与 PD 计算。"
        )
        self.set_status("选中的 GMD–PD 缓存拟合已应用到 Bfit 和 sigBfit；拟合截距不参与 PD 计算。")

    def run_alignment(self) -> None:
        group = self.group_var.get().strip()
        path = self.damage_h5_var.get().strip()
        if not group:
            messagebox.showinfo("未选择衰减组", "请先新增或选择衰减组。", parent=self)
            return
        if not path:
            self.choose_damage_file()
            path = self.damage_h5_var.get().strip()
        if not path:
            return
        try:
            start, end = int(self.range_start_var.get()), int(self.range_end_var.get())
            if end < start:
                raise ValueError("结束序号不能小于起始序号。")
            normalized_slope, slope_description, slope_metadata = self._resolve_pd_slope(path)
            rows = self.state.ensure_rows(group, start, end)
            result = align_damage_gmd(path, [record.area for record in rows], len(rows), self.shift_min_var.get(), self.shift_max_var.get())
        except Exception as exc:
            messagebox.showerror("GMD 提取或公式准备失败", str(exc), parent=self)
            return
        if self.path_store:
            self.path_store.remember_file("energy_damage_hdf5_open", path)
        for record, value in zip(rows, result.values):
            record.gmd = float(value)
            record.source_hdf5 = path
            record.gmd_shift = result.shift
        formulas = self.state.formulas.setdefault(group, {})
        for target, default in DEFAULT_FORMULAS.items():
            formulas.setdefault(target, default)
        constants = self._constants_for_group(group)
        constants["normalized_slope"] = normalized_slope
        if slope_metadata.get("mode") == "closest_calibration":
            try:
                local_sigma = float(slope_metadata.get("sigma_normalized_slope"))
                local_pprime = float(slope_metadata.get("pprime"))
                if not math.isfinite(local_sigma) or local_sigma < 0 or not math.isfinite(local_pprime):
                    raise ValueError("自动缓存的标准误差或 Pprime 无效。")
                constants["sig_normalized_slope"] = local_sigma
                constants["Pprime"] = local_pprime
            except (TypeError, ValueError):
                messagebox.showerror("自动缓存参数无效", "无法读取该缓存的 sig_normalized_slope 或 Pprime。", parent=self)
                return
        formula_errors: list[tuple[str, str]] = []
        for target in ("pd", "energy", "energy_rel_err", "area_rel_err"):
            formula_errors.extend(apply_formula(rows, target, formulas[target], constants))
        log = {
            "atten": group, "start": start, "end": end, "source": path, "shift": result.shift,
            "pearson": result.pearson, "spearman": result.spearman, "score": result.score,
            "level": result.level, "shot_number": result.shot_number, "mean_time_difference": result.mean_time_difference,
            "normalized_slope": normalized_slope,
            "pd_slope_source": slope_metadata,
        }
        self.state.alignment_log.append(log)
        self.alignment_result_var.set(
            f"已填入 {group} 倍 {start}–{end} 号：最佳偏移 {result.shift:+d}；"
            f"Pearson[ln(GMD),面积]={result.pearson:.5f}，Spearman={result.spearman:.5f}，综合={result.score:.5f}；"
            f"粒度={result.level}，shot_number={result.shot_number}。\n"
            f"已自动计算 PD、单脉冲能量和相对误差；{slope_description}。"
        )
        self.pd_slope_source_result_var.set(slope_description)
        self.state.notify("gmd_and_formulas")
        self.refresh_table(group)
        if formula_errors:
            examples = "\n".join(f"{uid}: {error}" for uid, error in formula_errors[:8])
            messagebox.showwarning(
                "GMD 已填入，但部分公式未能计算",
                f"共有 {len(formula_errors)} 个单元格留空。\n\n{examples}", parent=self,
            )
        self.set_status(
            f"GMD 匹配及其余列自动计算完成，偏移量 {result.shift:+d}；{slope_description}。"
        )

    def refresh_groups(self) -> None:
        groups = self.state.groups()
        for group in groups:
            if group not in self.tables:
                self._create_group_tab(group)
        for group in list(self.tables):
            if group not in groups:
                frame = self.table_frames.pop(group)
                self.group_notebook.forget(frame)
                self.tables.pop(group, None)
        for group in groups:
            self.refresh_table(group)
        self.group_combo["values"] = groups
        if groups and self.group_var.get() not in groups:
            self.group_var.set(groups[0])
            self._sync_group_controls(groups[0])

    def _create_group_tab(self, group: str) -> None:
        frame = ttk.Frame(self.group_notebook)
        sheet = Sheet(
            frame,
            headers=[HEADINGS[column] for column in HEADINGS],
            show_row_index=False,
            show_vertical_grid=True,
            show_horizontal_grid=True,
            table_grid_fg="#cbd5e1",
            header_grid_fg="#94a3b8",
            alternate_color="#f8fafc",
            paste_can_expand_x=False,
            paste_can_expand_y=False,
            font=("Microsoft YaHei UI", 10, "normal"),
            header_font=("Microsoft YaHei UI", 10, "bold"),
        )
        sheet.pack(fill=tk.BOTH, expand=True)
        sheet.enable_bindings(
            "single_select", "drag_select", "select_all", "column_select", "row_select",
            "ctrl_select", "column_width_resize", "double_click_column_resize",
            "row_height_resize", "arrowkeys", "copy", "paste", "undo", "redo",
            "edit_cell", "delete", "right_click_popup_menu", "rc_select",
        )
        sheet.readonly_columns(0, readonly=True)
        sheet.set_column_widths([78, 150, 150, 165, 140, 160, 160])
        sheet.bind("<<SheetModified>>", lambda _event, g=group: self._schedule_sheet_sync(g), add="+")
        self.group_notebook.add(frame, text=f"{group} 倍衰减")
        self.tables[group] = sheet
        self.table_frames[group] = frame

    def refresh_table(self, group: str) -> None:
        sheet = self.tables.get(group)
        if sheet is None:
            return
        data = []
        deleted_rows = []
        for record in self.state.group_records(group):
            if record.deleted:
                deleted_rows.append(len(data))
            data.append([record.serial, format_number(record.gmd), format_number(record.pd), format_number(record.energy),
                         format_number(record.area), format_number(record.energy_rel_err), format_number(record.area_rel_err)])
        sheet.set_sheet_data(data, reset_col_positions=False, reset_row_positions=True, redraw=False)
        sheet.readonly_columns(0, readonly=True)
        sheet.dehighlight_all(redraw=False)
        if deleted_rows:
            sheet.highlight_rows(deleted_rows, bg="#f1f5f9", fg="#94a3b8", redraw=False)
        sheet.refresh()

    def _schedule_sheet_sync(self, group: str) -> None:
        if group in self._sheet_sync_pending:
            return
        self._sheet_sync_pending.add(group)
        self.after_idle(lambda g=group: self._sync_sheet_to_state(g))

    def _sync_sheet_to_state(self, group: str) -> None:
        self._sheet_sync_pending.discard(group)
        sheet = self.tables.get(group)
        records = self.state.group_records(group)
        if sheet is None:
            return
        data = sheet.get_sheet_data()
        invalid: list[str] = []
        for row_index, record in enumerate(records):
            if row_index >= len(data):
                break
            row = data[row_index]
            for column_index, column_name in enumerate(list(HEADINGS)[1:], start=1):
                text = row[column_index] if column_index < len(row) else ""
                try:
                    value = None if str(text).strip() == "" else float(text)
                except (TypeError, ValueError):
                    invalid.append(f"第 {row_index + 1} 行 {HEADINGS[column_name]}")
                    continue
                setattr(record, column_name, value)
        if invalid:
            self.refresh_table(group)
            messagebox.showerror("粘贴数据中有非数值", "以下单元格未写入：\n" + "\n".join(invalid[:12]), parent=self)
            return
        self.state.notify("table_matrix_edit")
        self.set_status(f"第 3 步 {group} 倍衰减表格数据已更新。")

    def _edit_cell(self, event: tk.Event, group: str) -> None:
        tree = self.tables[group]
        region = tree.identify_region(event.x, event.y)
        row_id, column_id = tree.identify_row(event.y), tree.identify_column(event.x)
        if region != "cell" or not row_id or not column_id:
            return
        column_index = int(column_id[1:]) - 1
        column_name = list(HEADINGS)[column_index]
        if column_name == "serial":
            return
        bbox = tree.bbox(row_id, column_id)
        if not bbox:
            return
        x, y, width, height = bbox
        entry = ttk.Entry(tree)
        entry.place(x=x, y=y, width=width, height=height)
        entry.insert(0, tree.set(row_id, column_name))
        entry.select_range(0, tk.END)
        entry.focus_set()

        def commit(_event=None) -> None:
            text = entry.get().strip()
            try:
                value = None if text == "" else float(text)
            except ValueError:
                messagebox.showerror("数值无效", f"无法把“{text}”写入 {HEADINGS[column_name]}。", parent=self)
                entry.destroy()
                return
            setattr(self.state.records[row_id], column_name, value)
            entry.destroy()
            self.state.notify("cell")

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", lambda _event: entry.destroy())

    def add_group_dialog(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("新增衰减组/序号")
        dialog.transient(self.winfo_toplevel())
        group_var = tk.StringVar(value=self.group_var.get())
        start_var = tk.IntVar(value=1)
        end_var = tk.IntVar(value=30)
        ttk.Label(dialog, text="衰减倍数").grid(row=0, column=0, padx=10, pady=8, sticky=tk.W)
        ttk.Entry(dialog, textvariable=group_var, width=16).grid(row=0, column=1, padx=10, pady=8)
        ttk.Label(dialog, text="序号范围").grid(row=1, column=0, padx=10, pady=8, sticky=tk.W)
        box = ttk.Frame(dialog)
        box.grid(row=1, column=1, padx=10, pady=8)
        ttk.Entry(box, textvariable=start_var, width=7).pack(side=tk.LEFT)
        ttk.Label(box, text=" — ").pack(side=tk.LEFT)
        ttk.Entry(box, textvariable=end_var, width=7).pack(side=tk.LEFT)

        def confirm() -> None:
            try:
                self.state.ensure_rows(group_var.get(), start_var.get(), end_var.get())
            except Exception as exc:
                messagebox.showerror("无法新增", str(exc), parent=dialog)
                return
            self.group_var.set(group_var.get().strip())
            self.range_start_var.set(start_var.get())
            self.range_end_var.set(end_var.get())
            dialog.destroy()

        ttk.Button(dialog, text="确定", command=confirm).grid(row=2, column=0, columnspan=2, pady=(2, 10))

    def delete_current_group(self) -> None:
        group = self.group_var.get().strip()
        if not group or not self.state.group_records(group):
            messagebox.showinfo("未选择衰减组", "请先选择要删除的衰减组。", parent=self)
            return
        count = len(self.state.group_records(group))
        if not messagebox.askyesno(
            "删除衰减组",
            f"确定删除 {group} 倍衰减组及其 {count} 行共享数据吗？\n\n"
            "该操作会同时使第 4 步不再使用这些点；保存项目前可重新从第 1 步发布面积或重新新增该组。",
            parent=self,
        ):
            return
        for record in list(self.state.group_records(group)):
            self.state.records.pop(record.uid, None)
        self.state.formulas.pop(group, None)
        self.group_pressures.pop(group, None)
        self.group_var.set("")
        self.state.notify("group_deleted")
        self.set_status(f"第 3 步已删除 {group} 倍衰减组（{count} 行）。")

    def _formula_column_selected(self, _event=None) -> None:
        column = self.formula_column_var.get()
        group = self.group_var.get()
        formula = self.state.formulas.get(group, {}).get(column, DEFAULT_FORMULAS.get(column, ""))
        self.formula_var.set(formula)

    def _group_selected(self, _event=None) -> None:
        group = self.group_var.get()
        self._select_group_tab(group)
        self._sync_group_controls(group)

    def _sync_group_controls(self, group: str) -> None:
        rows = self.state.group_records(group)
        if rows:
            self.range_start_var.set(rows[0].serial)
            self.range_end_var.set(rows[-1].serial)
        self.pressure_var.set(self.group_pressures.get(group, self.state.constants.get("P", 0.0)))
        self._formula_column_selected()

    def _tab_changed(self, _event=None) -> None:
        selected = self.group_notebook.select()
        for group, frame in self.table_frames.items():
            if str(frame) == selected:
                self.group_var.set(group)
                self._sync_group_controls(group)
                break

    def _select_group_tab(self, group: str) -> None:
        frame = self.table_frames.get(group)
        if frame:
            self.group_notebook.select(frame)

    def _constants_for_group(self, group: str) -> dict[str, float]:
        constants = dict(self.state.constants)
        try:
            pressure = float(self.pressure_var.get()) if group == self.group_var.get() else self.group_pressures.get(group, constants["P"])
        except (tk.TclError, ValueError):
            pressure = constants["P"]
        constants["P"] = pressure
        self.group_pressures[group] = pressure
        return constants

    def apply_current_formula(self) -> None:
        group, target, formula = self.group_var.get(), self.formula_column_var.get(), self.formula_var.get().strip()
        if not group or not formula:
            return
        records = self.state.group_records(group)
        if not records:
            return
        errors = apply_formula(records, target, formula, self._constants_for_group(group))
        self.state.formulas.setdefault(group, {})[target] = formula
        self.state.notify("formula")
        self.refresh_table(group)
        if errors:
            examples = "\n".join(f"{uid}: {error}" for uid, error in errors[:8])
            messagebox.showwarning("部分行未能计算", f"{len(errors)} 行结果留空。常见原因是前置列为空或分母为 0。\n\n{examples}", parent=self)
        self.set_status(f"已将公式应用到 {group} 倍衰减的 {HEADINGS[target]} 列。")

    def recalculate_group(self) -> None:
        group = self.group_var.get()
        if not group:
            return
        formulas = self.state.formulas.setdefault(group, {})
        for target, default in DEFAULT_FORMULAS.items():
            formulas.setdefault(target, default)
        all_errors = []
        for target in ("pd", "energy", "energy_rel_err", "area_rel_err"):
            all_errors.extend(apply_formula(self.state.group_records(group), target, formulas[target], self._constants_for_group(group)))
        self.state.notify("formula")
        self.refresh_table(group)
        if all_errors:
            messagebox.showwarning("重算完成但有空值", f"共有 {len(all_errors)} 个单元格因缺少前置数据或数学错误而留空。", parent=self)
        self.set_status(f"已按依赖顺序重算 {group} 倍衰减的 PD、能量及误差列。")

    def open_constants_dialog(self) -> None:
        window = tk.Toplevel(self)
        window.title("命名常数")
        window.geometry("620x680")
        ttk.Label(window, text="公式可直接使用下列名称。修改后点击保存；项目存档会保留这些数值。", padding=10).pack(anchor=tk.W)
        canvas = tk.Canvas(window, highlightthickness=0)
        scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=canvas.yview)
        body = ttk.Frame(canvas, padding=10)
        body.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        variables: dict[str, tk.StringVar] = {}
        descriptions = {
            "normalized_slope": "标准化斜率 Bfit/exp(k·Pprime)",
            "sig_normalized_slope": "标准化斜率标准误差",
            "k": "气体衰减系数", "sigk": "气体衰减系数误差", "P": "默认损伤气压",
            "Pprime": "标准化斜率对应的 GMD–PD 标定气压 (Torr)",
            "load_resistance": "示波器负载电阻 Ω",
            "responsivity": "PD 响应/增益项", "eta": "PD 量子效率/响应度项", "sigeta": "eta 标准误差",
            "wavelength_A": "波长 Å", "q": "元电荷 C", "gmd_rel_err": "GMD 相对误差",
            "area_rel_default": "默认面积相对误差",
        }
        hidden_legacy = {"Bfit", "sigBfit", "intercept", "sigPDBackground", "IBG", "sigBG"}
        visible_keys = [key for key in self.state.constants if key not in hidden_legacy]
        for row, key in enumerate(visible_keys):
            ttk.Label(body, text=key, width=18).grid(row=row, column=0, sticky=tk.W, pady=3)
            variable = tk.StringVar(value=format_number(self.state.constants[key]))
            variables[key] = variable
            ttk.Entry(body, textvariable=variable, width=22).grid(row=row, column=1, pady=3)
            ttk.Label(body, text=descriptions.get(key, ""), foreground="#475569").grid(row=row, column=2, sticky=tk.W, padx=8)

        def save() -> None:
            try:
                values = {key: float(variable.get()) for key, variable in variables.items()}
            except ValueError as exc:
                messagebox.showerror("常数无效", str(exc), parent=window)
                return
            self.state.constants.update(values)
            self.state.notify("constants")
            window.destroy()

        ttk.Button(body, text="保存", command=save).grid(row=len(variables), column=0, columnspan=3, pady=12)

    def export_excel(self) -> None:
        if not self.state.groups():
            messagebox.showinfo("没有数据", "请先完成面积统计或新增衰减组。", parent=self)
            return
        initial_dir = self.path_store.initial_dir("energy_excel_save", Path.cwd()) if self.path_store else str(Path.cwd())
        filename = filedialog.asksaveasfilename(parent=self, initialdir=initial_dir, title="导出第 3 步数据", defaultextension=".xlsx", filetypes=[("Excel 工作簿", "*.xlsx")], initialfile="XFEL_energy_area_data.xlsx")
        if not filename:
            return
        if self.path_store:
            self.path_store.remember_file("energy_excel_save", filename)
        from openpyxl import Workbook
        from openpyxl.styles import Font
        workbook = Workbook()
        workbook.remove(workbook.active)
        for group in self.state.groups():
            sheet = workbook.create_sheet((f"atten_{group}")[:31])
            headers = list(HEADINGS.values()) + ["面积单位", "HDF5", "GMD偏移", "已删除"]
            sheet.append(headers)
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for record in self.state.group_records(group):
                sheet.append([record.serial, record.gmd, record.pd, record.energy, record.area, record.energy_rel_err,
                              record.area_rel_err, record.area_unit, record.source_hdf5, record.gmd_shift, record.deleted])
            sheet.freeze_panes = "A2"
            for column in range(1, len(headers) + 1):
                sheet.column_dimensions[chr(64 + column)].width = 18 if column < 9 else 35
        formula_sheet = workbook.create_sheet("formulas_constants")
        formula_sheet.append(["衰减组", "目标列", "公式"])
        for group, formulas in self.state.formulas.items():
            for target, formula in formulas.items():
                formula_sheet.append([group, target, formula])
        formula_sheet.append([])
        formula_sheet.append(["常数", "值"])
        for key, value in self.state.constants.items():
            formula_sheet.append([key, value])
        workbook.save(filename)
        self.set_status(f"第 3 步全部表格已导出：{filename}")

    def export_config(self) -> dict[str, Any]:
        return {
            "damage_file": self.damage_h5_var.get(), "shift_min": self.shift_min_var.get(), "shift_max": self.shift_max_var.get(),
            "group_pressures": self.group_pressures,
            "pd_slope_source": self.pd_slope_source_var.get(),
        }

    def import_config(self, data: dict[str, Any]) -> None:
        self.damage_h5_var.set(data.get("damage_file", ""))
        self.shift_min_var.set(data.get("shift_min", -10))
        self.shift_max_var.set(data.get("shift_max", 10))
        slope_source = str(data.get("pd_slope_source", "closest_calibration"))
        self.pd_slope_source_var.set(
            slope_source if slope_source in {"closest_calibration", "named_constant"} else "closest_calibration"
        )
        self.group_pressures = {str(key): float(value) for key, value in data.get("group_pressures", {}).items()}
        self.alignment_result_var.set("选择衰减组、序号范围和损伤 HDF5 文件。")
        self.refresh_groups()
        self._update_slope_source_preview()
