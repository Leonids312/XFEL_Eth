"""On-demand native RGB section viewer; never modifies measurements or writes files."""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from area_detection import cross_section, crossings


def show_profile(parent, title, roi, origin, axis, contour, paths=(), initial=0.5):
    window = tk.Toplevel(parent)
    window.title(f"原图 RGB 截面 — {title}")
    window.geometry("740x480")
    controls = ttk.Frame(window, padding=8)
    controls.pack(fill=tk.X)
    ttk.Label(controls, text="沿条纹移动截面：").pack(side=tk.LEFT)
    position = tk.DoubleVar(window, value=initial)
    slider = ttk.Scale(controls, variable=position, from_=0, to=1)
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    caption = ttk.Label(window, padding=(10, 0))
    caption.pack(anchor=tk.W)
    fig = Figure(figsize=(6, 3.5), dpi=100, layout="constrained")
    ax = fig.add_subplot()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    vertical = axis == "vertical"
    offset, along_offset = origin if vertical else origin[::-1]
    pts = np.asarray(contour, dtype=float).reshape(-1, 2) - origin if contour else None
    partial = [np.asarray(p, dtype=float).reshape(-1, 2) - origin for p in paths if len(p) >= 2]
    lines = [ax.plot([], [], color=color, lw=lw, label=label)[0] for color, lw, label in
             (("#d54b45", 1, "R"), ("#17814d", 1, "G"), ("#245dba", 1, "B"), ("#444444", 1, "Gray"))]
    ax.set_xlabel(("X" if vertical else "Y") + " (original px)")
    ax.set_ylabel("Digital counts")
    ax.legend(loc="upper right", ncol=4)
    markers = []

    def draw(_=None):
        at, profile = cross_section(roi, axis, position.get())
        x = offset + np.arange(len(profile))
        for line, values in zip(lines, (*profile.T, profile @ [.299, .587, .114])):
            line.set_data(x, values)
        for marker in markers:
            marker.remove()
        markers.clear()
        if pts is not None:
            for pos in crossings(pts, axis, at):
                markers.append(ax.axvline(offset + pos, ls="--", lw=1, color="#78350f"))
        for p in partial:
            for pos in crossings(p, axis, at, closed=False):
                markers.append(ax.axvline(offset + pos, ls=":", lw=1.4, color="#996515"))
        ax.relim()
        ax.autoscale_view()
        caption.configure(text=f"{'Y' if vertical else 'X'} = {at + along_offset} px；最多 5 条扫描线平均。虚线：轮廓；点线：未闭合候选边。")
        canvas.draw_idle()
    slider.configure(command=draw)
    draw()
