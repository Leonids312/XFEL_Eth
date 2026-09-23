"""Integrated three-step XFEL single-pulse damage threshold application."""

from __future__ import annotations

import ctypes
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from area_page import AreaAnalysisPage
from calibration_page import GMDPDCalibrationPage
from energy_page import EnergyExtractionPage
from threshold_page import ThresholdAnalysisPage
from xfel_core import ProjectState, RecentPathStore, load_json, save_json


APP_DIR = Path(__file__).resolve().parent
USER_CONFIG = APP_DIR / ".xfel_integrated_config.json"
DIALOG_PATHS_CONFIG = APP_DIR / ".xfel_dialog_paths.json"
PROJECT_VERSION = 21


class XFELDamageThresholdApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("XFEL 单脉冲辐照材料损伤阈值统计")
        self.geometry("1580x940")
        self.minsize(1220, 760)
        self.project_path: Path | None = None
        self.path_store = RecentPathStore(DIALOG_PATHS_CONFIG)
        self.dirty = False
        self.state = ProjectState()
        self.status_var = tk.StringVar(value="按顺序完成：1. 统计面积 → 2. GMD–PD 标定 → 3. 提取能量 → 4. 拟合阈值。")
        self._configure_style()
        self._build_ui()
        self.state.subscribe(self._mark_dirty)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._load_user_config()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 9))
        style.configure("TButton", padding=(9, 5))
        style.configure("Treeview", rowheight=25)
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Step.TNotebook.Tab", padding=(24, 8), font=("Microsoft YaHei UI", 10, "bold"))

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=(8, 6))
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="新建项目", command=self.new_project).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="打开项目", command=self.open_project).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(toolbar, text="保存项目", command=self.save_project).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(toolbar, text="项目另存为", command=self.save_project_as).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Label(toolbar, text="项目：").pack(side=tk.LEFT)
        self.project_label = ttk.Label(toolbar, text="未命名项目", foreground="#334155")
        self.project_label.pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(toolbar, text="四个步骤数据共享；保存项目会同时保存图像轮廓、标定缓存、公式、绘图设置和回收站。", foreground="#475569").pack(side=tk.LEFT)

        self.notebook = ttk.Notebook(self, style="Step.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.area_page = AreaAnalysisPage(self.notebook, self.state, self.set_status, self.path_store)
        self.calibration_page = GMDPDCalibrationPage(self.notebook, self.state, self.set_status, self.path_store)
        self.energy_page = EnergyExtractionPage(self.notebook, self.state, self.set_status, self.path_store)
        self.energy_page.set_calibration_cache_provider(lambda: self.calibration_page.calibration_cache)
        self.threshold_page = ThresholdAnalysisPage(self.notebook, self.state, self.set_status, self.path_store)
        self.notebook.add(self.area_page, text="1  统计损伤面积")
        self.notebook.add(self.calibration_page, text="2  GMD–PD 标定与缓存")
        self.notebook.add(self.energy_page, text="3  提取并对应单脉冲能量")
        self.notebook.add(self.threshold_page, text="4  绘图与拟合损伤阈值")
        ttk.Label(self, textvariable=self.status_var, anchor=tk.W, padding=(9, 5), background="#e8edf3").pack(fill=tk.X)

        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="新建项目", command=self.new_project, accelerator="Ctrl+N")
        file_menu.add_command(label="打开项目…", command=self.open_project, accelerator="Ctrl+O")
        file_menu.add_command(label="保存项目", command=self.save_project, accelerator="Ctrl+S")
        file_menu.add_command(label="项目另存为…", command=self.save_project_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_close)
        menu.add_cascade(label="文件", menu=file_menu)
        step_menu = tk.Menu(menu, tearoff=False)
        step_menu.add_command(label="1. 统计损伤面积", command=lambda: self.notebook.select(self.area_page))
        step_menu.add_command(label="2. GMD–PD 标定与缓存", command=lambda: self.notebook.select(self.calibration_page))
        step_menu.add_command(label="3. 提取并对应单脉冲能量", command=lambda: self.notebook.select(self.energy_page))
        step_menu.add_command(label="4. 绘图与拟合损伤阈值", command=lambda: self.notebook.select(self.threshold_page))
        menu.add_cascade(label="步骤", menu=step_menu)
        self.configure(menu=menu)
        self.bind_all("<Control-n>", lambda _event: self.new_project())
        self.bind_all("<Control-o>", lambda _event: self.open_project())
        self.bind_all("<Control-s>", lambda _event: self.save_project())
        self.bind_all("<Control-Shift-S>", lambda _event: self.save_project_as())

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.update_idletasks()

    def _mark_dirty(self, _reason: str) -> None:
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = self.project_path.name if self.project_path else "未命名项目"
        self.project_label.configure(text=name + (" *" if self.dirty else ""))
        self.title(f"XFEL 单脉冲损伤阈值统计 — {name}{' *' if self.dirty else ''}")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("项目尚未保存", "当前项目有未保存修改。是否先保存？", parent=self)
        if answer is None:
            return False
        if answer:
            return self.save_project()
        return True

    def new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.project_path = None
        self.state.load_dict({}, notify=False)
        self.area_page.root_dir = None
        self.area_page.entries.clear()
        self.area_page.current_entry = None
        self.area_page.results.clear()
        self.area_page.folder_var.set("")
        self.area_page.file_list.delete(0, tk.END)
        self.area_page.render_canvas(True)
        self.calibration_page.import_config({})
        self.energy_page.import_config({})
        self.threshold_page.import_config({})
        self.state.notify("new")
        self.dirty = False
        self._update_title()
        self.notebook.select(self.area_page)
        self.set_status("已新建空项目，请从第 1 步选择徕卡图像文件夹。")

    def project_data(self) -> dict[str, Any]:
        return {
            "application": "XFEL single-pulse damage threshold statistics",
            "version": PROJECT_VERSION,
            "state": self.state.to_dict(),
            "area_page": self.area_page.export_config(),
            "calibration_page": self.calibration_page.export_config(),
            "energy_page": self.energy_page.export_config(),
            "threshold_page": self.threshold_page.export_config(),
            "selected_step": self.notebook.index(self.notebook.select()),
        }

    def save_project(self) -> bool:
        if self.project_path is None:
            return self.save_project_as()
        try:
            save_json(self.project_path, self.project_data())
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)
            return False
        self.dirty = False
        self._update_title()
        self.set_status(f"项目已保存：{self.project_path}")
        self._save_user_config()
        return True

    def save_project_as(self) -> bool:
        fallback = self.project_path.parent if self.project_path else (self.area_page.root_dir or Path.cwd())
        initial_dir = self.path_store.initial_dir("project_save", fallback)
        filename = filedialog.asksaveasfilename(parent=self, initialdir=initial_dir, title="保存 XFEL 统计项目",
                                                defaultextension=".xfel.json", filetypes=[("XFEL 项目", "*.xfel.json"), ("JSON", "*.json")],
                                                initialfile=self.project_path.name if self.project_path else "XFEL_damage_threshold.xfel.json")
        if not filename:
            return False
        self.path_store.remember_file("project_save", filename)
        self.project_path = Path(filename)
        return self.save_project()

    def open_project(self, path: str | Path | None = None) -> None:
        if not self._confirm_discard():
            return
        if path is None:
            initial_dir = self.path_store.initial_dir("project_open", self.project_path.parent if self.project_path else Path.cwd())
            filename = filedialog.askopenfilename(parent=self, initialdir=initial_dir, title="打开 XFEL 统计项目", filetypes=[("XFEL 项目", "*.xfel.json *.json"), ("所有文件", "*.*")])
            if not filename:
                return
            self.path_store.remember_file("project_open", filename)
            path = filename
        project_path = Path(path)
        try:
            data = load_json(project_path)
            if "state" not in data:
                if "results" in data and (data.get("folder") or data.get("root_dir") or data.get("image_folder")):
                    self.state.load_dict({}, notify=False)
                    self.area_page.import_config(data)
                    self.project_path = None
                    self.state.notify("legacy_import")
                    self.dirty = True
                    self._update_title()
                    self.notebook.select(self.area_page)
                    self.set_status(f"已导入旧面积存档：{project_path}。请用“项目另存为”保存为集成项目。")
                    return
                raise ValueError("这既不是集成项目，也不是可识别的旧面积项目。")
            self.state.load_dict(data["state"], notify=False)
            self.area_page.import_config(data.get("area_page", {}))
            energy_config = dict(data.get("energy_page", {}))
            calibration_config = data.get("calibration_page")
            if calibration_config is None:
                # Version <= 6 stored the calibration page inside energy_page.
                calibration_config = {
                    key: energy_config.get(key)
                    for key in ("calibration_file", "offsets", "calibration_cache", "selected_calibration_id")
                    if key in energy_config
                }
                calibration_config["selected_section"] = 0
            self.calibration_page.import_config(calibration_config)
            self.energy_page.import_config(energy_config)
            self.threshold_page.import_config(data.get("threshold_page", {}))
            selected_step = int(data.get("selected_step", 0))
            if int(data.get("version", 0)) <= 6:
                if selected_step == 1:
                    selected_step = 1 if int(energy_config.get("selected_section", 1)) == 0 else 2
                elif selected_step >= 2:
                    selected_step += 1
            self.notebook.select(min(selected_step, 3))
        except Exception as exc:
            messagebox.showerror("打开项目失败", f"{project_path}\n\n{exc}", parent=self)
            return
        self.project_path = project_path
        self.state.notify("load")
        self.dirty = False
        self._update_title()
        self.set_status(f"项目已打开：{project_path}")
        self._save_user_config()

    def _load_user_config(self) -> None:
        if not USER_CONFIG.exists():
            return
        try:
            data = json.loads(USER_CONFIG.read_text(encoding="utf-8"))
            geometry = data.get("geometry")
            if geometry:
                self.geometry(geometry)
        except Exception:
            return

    def _save_user_config(self) -> None:
        data = {"geometry": self.geometry(), "last_project": str(self.project_path) if self.project_path else ""}
        try:
            USER_CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def on_close(self) -> None:
        if not self._confirm_discard():
            return
        self._save_user_config()
        self.destroy()


def enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def main() -> int:
    enable_windows_dpi_awareness()
    app = XFELDamageThresholdApp()
    if "--smoke-test" in sys.argv:
        app.withdraw()
        app.update_idletasks()
        app.dirty = False
        app.destroy()
        return 0
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.is_file():
            app.after(50, lambda: app.open_project(candidate))
        elif candidate.is_dir():
            app.after(50, lambda: app.area_page.load_folder(candidate))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
