"""Editable, uncommitted pit outline in original image coordinates."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


def _nodes(points, closed):
    pts = np.asarray(points, np.int32).reshape(-1, 2)
    if len(pts) < (3 if closed else 2):
        raise ValueError("草稿中没有足够的不同边界点。")
    eps = 0.5
    while True:
        reduced = cv2.approxPolyDP(pts, eps, closed).reshape(-1, 2)
        if len(reduced) <= 128:
            break
        eps *= 1.5
    if len(reduced) < (3 if closed else 2):
        raise ValueError("草稿退化为线/点，请改用手动画。")
    return [tuple(map(int, p)) for p in reduced]


def _orientation(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])


def _intersect(a, b, c, d):
    if any(max(a[i], b[i]) < min(c[i], d[i]) or max(c[i], d[i]) < min(a[i], b[i]) for i in (0, 1)):
        return False
    return _orientation(a, b, c) * _orientation(a, b, d) <= 0 and _orientation(c, d, a) * _orientation(c, d, b) <= 0


class OutlineDraft:
    """The caller's result is not changed. Open DIC paths have no closing edges until marked."""
    def __init__(self, result, bounds):
        if result.get("status") == "图像截断":
            raise ValueError("已判为图像截断，不能把画面外的部分用草稿封口。")
        # A DIC review may carry a speculative polygon as well as two observed
        # ridges. Never let that polygon bypass physical confirmation of both tips.
        paired_draft = (result.get("status") == "待复核" and result.get("method") == "dic"
                        and len(result.get("paths") or []) == 2)
        contour = None if paired_draft else result.get("contour") or result.get("review_contour")
        self.bounds = bounds
        self.open_ends = not bool(contour)
        self.tips = [None, None]
        self.edited = False
        self.original = None
        if contour:
            self.original = [tuple(map(int, p)) for p in contour]
            self.chains = [_nodes(contour, True)]
        else:
            paths = result.get("paths") or []
            if len(paths) != 2 or result.get("method") != "dic":
                raise ValueError("没有可修的闭合轮廓或完整双边草稿；请使用手动画轮廓。")
            a, b = (_nodes(path, False) for path in paths)
            if math.dist(a[0], b[-1]) + math.dist(a[-1], b[0]) < math.dist(a[0], b[0]) + math.dist(a[-1], b[-1]):
                b.reverse()
            self.chains = [a, b]

    def within(self, p):
        x0, y0, x1, y1 = self.bounds
        return x0 <= p[0] < x1 and y0 <= p[1] < y1

    def move(self, selected, point):
        if self.within(point):
            i, j = selected
            self.chains[i][j] = point
            self.edited = True

    def insert(self, point, radius):
        if not self.within(point):
            return None
        best = (float("inf"), 0, 0)
        x, y = point
        for i, nodes in enumerate(self.chains):
            for j in range(len(nodes) - int(self.open_ends)):
                a, b = nodes[j], nodes[(j+1) % len(nodes)]
                dx, dy = b[0]-a[0], b[1]-a[1]
                t = min(1., max(0., ((x-a[0])*dx+(y-a[1])*dy) / max(1., dx*dx+dy*dy)))
                best = min(best, (math.hypot(x-a[0]-t*dx, y-a[1]-t*dy), i, j+1))
        distance, i, j = best
        if distance > radius:
            return None
        self.chains[i].insert(j, point)
        self.edited = True
        return (i, j)

    def remove(self, selected):
        i, j = selected
        if len(self.chains[i]) <= (2 if self.open_ends else 3):
            raise ValueError("该边段已达到最少节点数。")
        del self.chains[i][j]
        self.edited = True

    def polygon(self):
        if self.open_ends:
            if any(p is None for p in self.tips):
                raise ValueError("请先核实两个端部：点击“标首端 / 标末端”后，在原图上分别点出可见端点。")
            pts = self.chains[0] + [self.tips[1]] + self.chains[1][::-1] + [self.tips[0]]
        else:
            pts = self.chains[0]
        points = [p for i, p in enumerate(pts) if p != pts[i-1]]
        if len(points) < 3 or not all(self.within(p) for p in points):
            raise ValueError("轮廓无效或越出当前坑位。")
        edges = list(zip(points, points[1:] + points[:1]))
        for i, (a, b) in enumerate(edges):
            if any(_intersect(a,b,c,d) for j,(c,d) in enumerate(edges[i+2:], i+2)
                   if not (i == 0 and j == len(edges)-1)):
                raise ValueError("边界发生自交，请调整控制点或重新标记端点。")
        # No editing of a closed candidate must preserve its native area rather than
        # replacing it with the lower-resolution handles used for interactive display.
        if self.original is not None and not self.edited:
            points = self.original
        if not all(self.within(p) for p in points) or cv2.contourArea(np.asarray(points, np.int32)) <= 0:
            raise ValueError("有效面积为零或越出当前坑位。")
        return [list(p) for p in points]


def edit_outline(parent, rgb, draft, label, commit):
    """Display-only RGB crop. All handles remain in full-image, unscaled coordinates."""
    win = tk.Toplevel(parent)
    win.title(f"人工修边 · {label}")
    win.transient(parent.winfo_toplevel())
    win.geometry(f"{min(900, int(win.winfo_screenwidth()*.9))}x{min(720, int(win.winfo_screenheight()*.85))}")
    x0, y0, x1, y1 = draft.bounds
    source = Image.fromarray(rgb[y0:y1, x0:x1])
    toolbar = ttk.Frame(win, padding=5)
    toolbar.pack(fill=tk.X)
    info = tk.StringVar(win, value="拖动圆点修边；Shift+单击边线添加点；Delete 删除选中点；滚轮缩放，右键平移。")
    ttk.Label(win, textvariable=info, padding=4, wraplength=730).pack(fill=tk.X)
    canvas = tk.Canvas(win, bg="#202329", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    selected = pending_tip = drag_origin = None
    zoom, pan_x, pan_y = 1., 0., 0.

    def transform():
        w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        s = min(w / source.width, h / source.height) * zoom
        return s, (w-source.width*s)/2+pan_x, (h-source.height*s)/2+pan_y

    def coords(pt):
        s, ox, oy = transform()
        return ox+(pt[0]-x0)*s, oy+(pt[1]-y0)*s

    def native(e):
        s, ox, oy = transform()
        return round((e.x-ox)/s)+x0, round((e.y-oy)/s)+y0

    def line(nodes, **kw):
        if len(nodes) >= 2:
            canvas.create_line(*[v for p in nodes for v in coords(p)], width=2, tags="handles", **kw)

    def redraw_handles():
        canvas.delete("handles")
        for i, nodes in enumerate(draft.chains):
            color = ("#37f28d", "#52dcff")[i]
            line(nodes if draft.open_ends else nodes + nodes[:1], fill=color)
            for j,p in enumerate(nodes):
                x,y = coords(p)
                canvas.create_oval(x-3,y-3,x+3,y+3, fill=("orange" if selected == (i,j) else color),
                                   outline="#202329", tags="handles")
        if draft.open_ends:
            for end,idx in enumerate((0,-1)):
                a,b = draft.chains[0][idx], draft.chains[1][idx]
                tip = draft.tips[end]
                if tip is not None:
                    line([a,tip,b], fill="orange", dash=(4,3))
                x,y = coords(tip if tip is not None else a)
                canvas.create_text(x+4, y, anchor="sw", fill="orange", text=("首端","末端")[end], tags="handles")

    def render(event=None):
        s, ox, oy = transform()
        w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        display = source.transform((w,h), Image.Transform.AFFINE, (1/s,0,-ox/s,0,1/s,-oy/s),
                                   Image.Resampling.BILINEAR, fillcolor="#202329")
        canvas.img = ImageTk.PhotoImage(display, master=canvas)
        canvas.delete("raster")
        canvas.create_image(0,0, anchor="nw", image=canvas.img, tags="raster")
        redraw_handles()

    def press(e):
        nonlocal selected,pending_tip
        canvas.focus_set()
        pt = native(e)
        if not draft.within(pt):
            return
        s = transform()[0]
        if pending_tip is not None:
            draft.tips[pending_tip] = pt
            pending_tip = selected = None
        elif e.state & 1:
            selected = draft.insert(pt, 12/s)
        else:
            d,i,j = min((math.dist(coords(p),(e.x,e.y)), i, j)
                        for i,nodes in enumerate(draft.chains) for j,p in enumerate(nodes))
            selected = (i,j) if d <= 10 else None
        redraw_handles()

    def move(e):
        if selected is not None:
            draft.move(selected,native(e))
            redraw_handles()

    def remove(e=None):
        nonlocal selected
        if selected is not None:
            try:
                draft.remove(selected)
            except ValueError as exc:
                info.set(str(exc)); return
            selected = None
            redraw_handles()

    def mark_tip(n):
        nonlocal pending_tip
        pending_tip = n
        info.set("请在原图点选" + ("首端" if n == 0 else "末端") + "的可见闭合位置；端部不可见请取消，不计面积。")

    def wheel(e):
        nonlocal zoom,pan_x,pan_y
        old_s,ox,oy = transform()
        zoom = max(.5,min(32,zoom*(1.25 if e.delta>0 else .8)))
        s,nx,ny = transform()
        pan_x += e.x-nx-(e.x-ox)*s/old_s
        pan_y += e.y-ny-(e.y-oy)*s/old_s
        render()

    def pan_start(e):
        nonlocal drag_origin
        drag_origin = (e.x,e.y,pan_x,pan_y)

    def pan(e):
        nonlocal pan_x,pan_y
        if drag_origin:
            px,py,bx,by = drag_origin
            pan_x,pan_y = bx+e.x-px, by+e.y-py
            render()

    def accept():
        try:
            commit(draft.polygon())
        except ValueError as exc:
            messagebox.showwarning("草稿未采用", str(exc), parent=win)
            return
        win.destroy()

    ttk.Button(toolbar,text="删除选中节点",command=remove).pack(side="left")
    if draft.open_ends:
        for n,title in enumerate(("标首端","标末端")):
            ttk.Button(toolbar,text=title,command=lambda n=n:mark_tip(n)).pack(side="left",padx=2)
    ttk.Button(toolbar,text="取消",command=win.destroy).pack(side="right")
    ttk.Button(toolbar,text="确认轮廓并计算面积",command=accept).pack(side="right",padx=4)
    canvas.bind("<Button-1>", press)
    canvas.bind("<B1-Motion>", move)
    canvas.bind("<Delete>", remove)
    canvas.bind("<Configure>", render)
    canvas.bind("<MouseWheel>", wheel)
    canvas.bind("<Button-3>", pan_start)
    canvas.bind("<B3-Motion>", pan)
    win.bind("<Escape>",lambda e:win.destroy())
    win.grab_set()
