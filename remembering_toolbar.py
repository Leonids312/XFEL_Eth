"""Matplotlib Tk toolbar whose save dialog remembers a purpose-specific directory."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from xfel_core import RecentPathStore


class RememberingNavigationToolbar(NavigationToolbar2Tk):
    def __init__(
        self,
        canvas,
        window: tk.Misc,
        *,
        path_store: RecentPathStore | None,
        path_key: str,
        pack_toolbar: bool = False,
        dpi_getter: Callable[[], int] | None = None,
    ) -> None:
        self.path_store = path_store
        self.path_key = path_key
        self.dpi_getter = dpi_getter
        super().__init__(canvas, window, pack_toolbar=pack_toolbar)

    def save_figure(self, *args) -> None:
        grouped = self.canvas.get_supported_filetypes_grouped()
        filetypes = []
        for description, extensions in grouped.items():
            patterns = " ".join(f"*.{extension}" for extension in extensions)
            filetypes.append((description, patterns))
        default_extension = self.canvas.get_default_filetype()
        initial_dir = self.path_store.initial_dir(self.path_key, Path.cwd()) if self.path_store else str(Path.cwd())
        filename = filedialog.asksaveasfilename(
            parent=self,
            title="保存图像",
            initialdir=initial_dir,
            initialfile=self.canvas.get_default_filename(),
            defaultextension=f".{default_extension}",
            filetypes=filetypes,
        )
        if not filename:
            return
        if self.path_store:
            self.path_store.remember_file(self.path_key, filename)
        try:
            kwargs = {"dpi": self.dpi_getter()} if self.dpi_getter else {}
            self.canvas.figure.savefig(filename, **kwargs)
        except Exception as exc:
            messagebox.showerror("保存图像失败", str(exc), parent=self)
