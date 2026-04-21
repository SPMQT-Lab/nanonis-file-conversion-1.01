"""ProbeFlow -- graphical interface for Createc-to-Nanonis file conversion."""

import json
import logging
import queue
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog
from typing import Callable, List, Optional
import webbrowser

from PIL import Image, ImageTk

CONFIG_PATH     = Path.home() / ".probeflow_config.json"
REPO_ROOT       = Path(__file__).resolve().parent.parent
DEFAULT_CUSHION = REPO_ROOT / "src" / "file_cushions"
LOGO_PATH       = REPO_ROOT / "assets" / "logo.png"
GITHUB_URL      = "https://github.com/SPMQT-Lab/Createc-to-Nanonis-file-conversion"

NAVBAR_BG = "#3273dc"
NAVBAR_FG = "#ffffff"
NAVBAR_H  = 58

THEMES = {
    "dark": {
        "bg":         "#1e1e2e",
        "fg":         "#cdd6f4",
        "entry_bg":   "#313244",
        "btn_bg":     "#45475a",
        "btn_fg":     "#cdd6f4",
        "log_bg":     "#181825",
        "log_fg":     "#cdd6f4",
        "ok_fg":      "#a6e3a1",
        "err_fg":     "#f38ba8",
        "warn_fg":    "#fab387",
        "accent_bg":  "#89b4fa",
        "accent_fg":  "#1e1e2e",
        "sep":        "#45475a",
        "sub_fg":     "#6c7086",
        "sidebar_bg": "#181825",
        "main_bg":    "#1e1e2e",
        "status_bg":  "#313244",
        "status_fg":  "#6c7086",
        "card_bg":    "#313244",
        "card_sel":   "#89b4fa",
        "card_fg":    "#cdd6f4",
        "tab_act":    "#45475a",
        "tab_inact":  "#313244",
    },
    "light": {
        "bg":         "#f8f9fa",
        "fg":         "#1e1e2e",
        "entry_bg":   "#ffffff",
        "btn_bg":     "#e0e0e0",
        "btn_fg":     "#1e1e2e",
        "log_bg":     "#ffffff",
        "log_fg":     "#1e1e2e",
        "ok_fg":      "#1a7a1a",
        "err_fg":     "#c0392b",
        "warn_fg":    "#b07800",
        "accent_bg":  "#3273dc",
        "accent_fg":  "#ffffff",
        "sep":        "#dee2e6",
        "sub_fg":     "#6c757d",
        "sidebar_bg": "#eff5fb",
        "main_bg":    "#eef6fc",
        "status_bg":  "#f5f5f5",
        "status_fg":  "#6c757d",
        "card_bg":    "#d0e8f8",
        "card_sel":   "#3273dc",
        "card_fg":    "#1e1e2e",
        "tab_act":    "#ffffff",
        "tab_inact":  "#d8eaf8",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatGroup:
    stem:    str
    pngs:    List[Path] = field(default_factory=list)
    sxm:     Optional[Path] = None
    preview: Optional[Path] = None   # preferred Z_forward PNG


def parse_sxm_header(sxm_path: Path) -> dict:
    """Read the ASCII header section of an .sxm file into a key->value dict."""
    params: dict = {}
    current_key: Optional[str] = None
    lines_buf: List[str] = []

    def _flush():
        if current_key is not None:
            params[current_key] = " ".join(lines_buf).strip()

    try:
        with open(sxm_path, "rb") as fh:
            for raw in fh:
                if raw.strip() == b":SCANIT_END:":
                    break
                line = raw.decode("latin-1", errors="replace").rstrip("\r\n")
                if line.startswith(":") and line.endswith(":") and len(line) > 2:
                    _flush()
                    current_key = line[1:-1]
                    lines_buf = []
                elif current_key is not None:
                    s = line.strip()
                    if s:
                        lines_buf.append(s)
        _flush()
    except Exception:
        pass
    return params


def scan_output_folder(root: Path) -> List[DatGroup]:
    """Find DatGroups under a ProbeFlow output folder."""
    groups: dict = {}
    root = Path(root)

    for png in sorted(root.rglob("*.png")):
        stem = png.parents[1].name if png.parent.name == "pngs" else png.parent.name
        if stem not in groups:
            groups[stem] = {"pngs": [], "sxm": None}
        groups[stem]["pngs"].append(png)

    for sxm in sorted(root.rglob("*.sxm")):
        s = sxm.stem
        if s in groups:
            groups[s]["sxm"] = sxm

    result = []
    for stem, data in sorted(groups.items()):
        pngs = sorted(data["pngs"])
        preview = (
            next((p for p in pngs if "Z" in p.name and "forward" in p.name), None)
            or (pngs[0] if pngs else None)
        )
        result.append(DatGroup(stem=stem, pngs=pngs, sxm=data["sxm"], preview=preview))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    defaults = {
        "dark_mode": False, "input_dir": "", "output_dir": "",
        "do_png": True, "do_sxm": True, "clip_low": 1.0, "clip_high": 99.0,
    }
    try:
        if CONFIG_PATH.exists():
            defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return defaults


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

class QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(record)


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail grid widget
# ─────────────────────────────────────────────────────────────────────────────

class ThumbnailGrid(tk.Frame):
    CARD_W = 164
    CARD_H = 148
    IMG_W  = 148
    IMG_H  = 116
    GAP    = 10

    def __init__(self, parent, on_select: Callable, theme: dict, **kw):
        super().__init__(parent, **kw)
        self._on_select = on_select
        self._t = theme
        self._groups: List[DatGroup] = []
        self._photos: dict = {}
        self._selected: Optional[str] = None

        self._canvas = tk.Canvas(self, highlightthickness=0, bd=0)
        self._vsb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._vsb.pack(side="right", fill="y")

        self._canvas.bind("<Configure>", lambda e: self._layout())
        self._canvas.bind("<MouseWheel>", self._scroll)
        self._canvas.bind("<Button-4>",   self._scroll)
        self._canvas.bind("<Button-5>",   self._scroll)

    def load(self, groups: List[DatGroup]) -> None:
        self._groups = groups
        self._photos.clear()
        self._selected = None
        self._canvas.delete("all")
        for g in groups:
            self._photos[g.stem] = self._make_photo(g.preview)
        self._layout()

    def _make_photo(self, path: Optional[Path]) -> Optional[ImageTk.PhotoImage]:
        if not path or not path.exists():
            return None
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.IMG_W, self.IMG_H), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _layout(self) -> None:
        cw = self._canvas.winfo_width()
        if cw < 10:
            self.after(50, self._layout)
            return
        cols = max(1, (cw + self.GAP) // (self.CARD_W + self.GAP))
        t = self._t
        self._canvas.delete("all")

        for i, g in enumerate(self._groups):
            row, col = divmod(i, cols)
            x0 = col * (self.CARD_W + self.GAP) + self.GAP
            y0 = row * (self.CARD_H + self.GAP) + self.GAP
            x1, y1 = x0 + self.CARD_W, y0 + self.CARD_H
            sel = (g.stem == self._selected)

            self._canvas.create_rectangle(
                x0, y0, x1, y1,
                fill=t["card_sel"] if sel else t["card_bg"],
                outline=t["accent_bg"] if sel else t["sep"],
                width=2 if sel else 1,
                tags=("card", f"s:{g.stem}"),
            )
            photo = self._photos.get(g.stem)
            if photo:
                self._canvas.create_image(
                    x0 + self.CARD_W // 2, y0 + self.IMG_H // 2 + 4,
                    image=photo, tags=("card", f"s:{g.stem}"),
                )
            label = g.stem if len(g.stem) <= 20 else g.stem[:18] + ".."
            self._canvas.create_text(
                x0 + self.CARD_W // 2, y1 - 11,
                text=label, font=("Helvetica", 7),
                fill="#ffffff" if sel else t["card_fg"],
                tags=("card", f"s:{g.stem}"),
            )

        total_rows = max(1, (len(self._groups) + cols - 1) // cols)
        total_h = total_rows * (self.CARD_H + self.GAP) + self.GAP
        self._canvas.configure(scrollregion=(0, 0, cw, max(total_h, 1)))
        self._canvas.tag_bind("card", "<Button-1>", self._on_click)

    def _on_click(self, event: tk.Event) -> None:
        items = self._canvas.find_closest(event.x, event.y)
        if not items:
            return
        for tag in self._canvas.gettags(items[0]):
            if tag.startswith("s:"):
                stem = tag[2:]
                for g in self._groups:
                    if g.stem == stem:
                        self._selected = stem
                        self._layout()
                        self._on_select(g)
                        return

    def _scroll(self, event: tk.Event) -> None:
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def apply_theme(self, t: dict) -> None:
        self._t = t
        self._canvas.configure(bg=t["main_bg"])
        self._layout()


# ─────────────────────────────────────────────────────────────────────────────
# About popup
# ─────────────────────────────────────────────────────────────────────────────

def _show_about(parent: tk.Tk, dark: bool) -> None:
    t = THEMES["dark" if dark else "light"]
    win = tk.Toplevel(parent)
    win.title("About ProbeFlow")
    win.resizable(False, False)
    win.configure(bg=t["bg"])
    win.grab_set()

    try:
        img = Image.open(LOGO_PATH).convert("RGBA")
        img.thumbnail((300, 100), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo, bg=t["bg"])
        lbl.image = photo
        lbl.pack(pady=(18, 6))
    except Exception:
        pass

    def row(text, size=10, bold=False, color=None):
        tk.Label(win, text=text,
                 font=("Helvetica", size, "bold" if bold else "normal"),
                 bg=t["bg"], fg=color or t["fg"],
                 wraplength=360, justify="center").pack(pady=2, padx=24)

    row("ProbeFlow", 15, bold=True)
    row("Createc -> Nanonis File Conversion", 10, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    row("Developed at SPMQT-Lab", 10, bold=True)
    row("Under the supervision of Dr. Peter Jacobson\nThe University of Queensland", 9,
        color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    row("Original code by Rohan Platts", 10, bold=True)
    row("The core conversion algorithms were built by Rohan Platts.\n"
        "This software is a refactored and extended version of his work,\n"
        "developed within SPMQT-Lab.", 9, color=t["sub_fg"])
    tk.Frame(win, height=1, bg=t["sep"]).pack(fill="x", padx=24, pady=10)
    tk.Button(win, text="View on GitHub", bg=NAVBAR_BG, fg=NAVBAR_FG,
              relief="flat", cursor="hand2", font=("Helvetica", 9),
              command=lambda: webbrowser.open(GITHUB_URL)
              ).pack(pady=(0, 18), ipadx=14, ipady=5)


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class ProbeFlowGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ProbeFlow")
        self.root.minsize(900, 620)
        self.root.resizable(True, True)

        self.cfg = load_config()
        self.log_queue: queue.Queue = queue.Queue()
        self._running   = False
        self._adv_vis   = False
        self._mode      = "convert"   # "convert" | "browse"

        self.dark_mode  = tk.BooleanVar(value=self.cfg["dark_mode"])
        self.input_dir  = tk.StringVar(value=self.cfg["input_dir"])
        self.output_dir = tk.StringVar(value=self.cfg["output_dir"])
        self.do_png     = tk.BooleanVar(value=self.cfg["do_png"])
        self.do_sxm     = tk.BooleanVar(value=self.cfg["do_sxm"])
        self.clip_low   = tk.DoubleVar(value=self.cfg["clip_low"])
        self.clip_high  = tk.DoubleVar(value=self.cfg["clip_high"])
        self._status    = tk.StringVar(value="Ready")

        self._build_ui()
        self._apply_theme()
        self._poll_log()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────────────────────
    # Build
    # ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_navbar()
        self._build_body()
        self._build_statusbar()

    def _build_navbar(self) -> None:
        nav = tk.Frame(self.root, bg=NAVBAR_BG, height=NAVBAR_H)
        nav.pack(fill="x")
        nav.pack_propagate(False)
        self._nav = nav

        try:
            img = Image.open(LOGO_PATH).convert("RGBA")
            img.thumbnail((120, 44), Image.LANCZOS)
            data = img.getdata()
            img.putdata([(r, g, b, 0) if r > 220 and g > 220 and b > 220 else (r, g, b, a)
                         for r, g, b, a in data])
            self._nav_logo = ImageTk.PhotoImage(img)
            logo_lbl = tk.Label(nav, image=self._nav_logo, bg=NAVBAR_BG, cursor="hand2")
            logo_lbl.pack(side="left", padx=(8, 0), pady=6)
            logo_lbl.bind("<Button-1>", lambda e: webbrowser.open(GITHUB_URL))
        except Exception:
            pass

        tf = tk.Frame(nav, bg=NAVBAR_BG)
        tf.pack(side="left", padx=(6, 0))
        tk.Label(tf, text="ProbeFlow", font=("Helvetica", 14, "bold"),
                 bg=NAVBAR_BG, fg=NAVBAR_FG).pack(anchor="w")
        tk.Label(tf, text="Createc -> Nanonis", font=("Helvetica", 8),
                 bg=NAVBAR_BG, fg="#a8c8f0").pack(anchor="w")

        def _navbtn(text, cmd):
            return tk.Button(nav, text=text, bg=NAVBAR_BG, fg=NAVBAR_FG,
                             relief="flat", cursor="hand2", font=("Helvetica", 9),
                             bd=0, padx=10, activebackground="#2563c0",
                             activeforeground=NAVBAR_FG, command=cmd)

        _navbtn("About",  lambda: _show_about(self.root, self.dark_mode.get())).pack(side="right", pady=14)
        _navbtn("GitHub", lambda: webbrowser.open(GITHUB_URL)).pack(side="right", pady=14)

        self._open_btn = _navbtn("Open folder", self._open_browse_folder)
        self._open_btn.pack(side="right", pady=14)

        self._theme_btn = _navbtn(
            "Light mode" if self.dark_mode.get() else "Dark mode",
            self._toggle_theme,
        )
        self._theme_btn.pack(side="right", pady=14)

    def _build_body(self) -> None:
        body = tk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self._body = body

        # Left: content area with mode tabs
        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        self._left = left

        # Tab bar
        tabbar = tk.Frame(left)
        tabbar.pack(fill="x")
        self._tabbar = tabbar
        self._tab_convert = tk.Button(tabbar, text="Convert",
                                      font=("Helvetica", 9, "bold"),
                                      relief="flat", cursor="hand2", bd=0, padx=16, pady=6,
                                      command=lambda: self._switch_mode("convert"))
        self._tab_convert.pack(side="left")
        self._tab_browse = tk.Button(tabbar, text="Browse",
                                     font=("Helvetica", 9, "bold"),
                                     relief="flat", cursor="hand2", bd=0, padx=16, pady=6,
                                     command=lambda: self._switch_mode("browse"))
        self._tab_browse.pack(side="left")

        # Convert panel
        self._conv_frame = tk.Frame(left)
        self._build_convert_panel(self._conv_frame)

        # Browse panel
        self._browse_frame = tk.Frame(left)
        self._grid = ThumbnailGrid(
            self._browse_frame, self._on_group_select,
            THEMES["dark" if self.dark_mode.get() else "light"],
        )
        self._grid.pack(fill="both", expand=True)

        # Show convert by default
        self._conv_frame.pack(fill="both", expand=True)

        # Right sidebar (290px)
        self._sidebar = tk.Frame(body, width=290)
        self._sidebar.pack(side="right", fill="y")
        self._sidebar.pack_propagate(False)

        self._conv_sidebar = tk.Frame(self._sidebar)
        self._build_convert_sidebar(self._conv_sidebar)

        self._browse_sidebar = tk.Frame(self._sidebar)
        self._build_browse_sidebar(self._browse_sidebar)

        self._conv_sidebar.pack(fill="both", expand=True)

    def _build_convert_panel(self, p: tk.Frame) -> None:
        self._folder_row(p, "Input folder:",  self.input_dir,  self._browse_input)
        self._folder_row(p, "Output folder:", self.output_dir, self._browse_output)
        self._hsep(p)

        log_hdr = tk.Frame(p)
        log_hdr.pack(fill="x", padx=16, pady=(2, 0))
        tk.Label(log_hdr, text="Conversion log", font=("Helvetica", 9, "bold"),
                 anchor="w").pack(side="left")
        tk.Button(log_hdr, text="Clear", relief="flat", cursor="hand2",
                  font=("Helvetica", 8), command=self._clear_log).pack(side="right")

        self.log_text = tk.Text(p, height=14, wrap="word", relief="flat", bd=0,
                                font=("Courier", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(2, 8))

    def _build_convert_sidebar(self, s: tk.Frame) -> None:
        self._sec(s, "Convert to")
        cbf = tk.Frame(s)
        cbf.pack(fill="x", padx=16, pady=4)
        self.png_cb = tk.Checkbutton(cbf, text="PNG preview",   variable=self.do_png)
        self.sxm_cb = tk.Checkbutton(cbf, text="SXM (Nanonis)", variable=self.do_sxm)
        self.png_cb.pack(anchor="w", pady=2)
        self.sxm_cb.pack(anchor="w", pady=2)
        self._hsep(s)

        adv_hdr = tk.Frame(s)
        adv_hdr.pack(fill="x", padx=12, pady=(2, 0))
        self._adv_btn = tk.Button(adv_hdr, text="[+] Advanced",
                                  relief="flat", bd=0, cursor="hand2",
                                  font=("Helvetica", 9), anchor="w",
                                  command=self._toggle_adv)
        self._adv_btn.pack(side="left")

        self._adv_frame = tk.Frame(s)
        self._slider_row(self._adv_frame, "Clip low (%):",  self.clip_low,  0.0,  10.0)
        self._slider_row(self._adv_frame, "Clip high (%):", self.clip_high, 90.0, 100.0)
        self._hsep(s)

        rf = tk.Frame(s)
        rf.pack(fill="x", padx=16, pady=10)
        self.run_btn = tk.Button(rf, text="  RUN  ",
                                 font=("Helvetica", 12, "bold"),
                                 relief="flat", cursor="hand2", command=self._run)
        self.run_btn.pack(fill="x", ipady=8)
        self._hsep(s)

        # File count label
        self._fcount_var = tk.StringVar(value="No folder selected")
        self._fcount_lbl = tk.Label(s, textvariable=self._fcount_var,
                                    font=("Helvetica", 8), anchor="w",
                                    wraplength=250, justify="left")
        self._fcount_lbl.pack(fill="x", padx=14, pady=4)

        self.input_dir.trace_add("write", lambda *_: self._update_count())

        self._conv_footer = tk.Label(
            s,
            text="SPMQT-Lab  |  Dr. Peter Jacobson\nThe University of Queensland\nOriginal code by Rohan Platts",
            font=("Helvetica", 7), anchor="center", justify="center",
        )
        self._conv_footer.pack(side="bottom", pady=8)

    def _build_browse_sidebar(self, s: tk.Frame) -> None:
        self._sec(s, "Selected scan")

        # Channel thumbnails (2x2 grid of small images)
        self._ch_frame = tk.Frame(s)
        self._ch_frame.pack(fill="x", padx=8, pady=4)
        self._ch_labels: List[tk.Label] = []
        self._ch_photos: List[Optional[ImageTk.PhotoImage]] = []
        for row in range(2):
            for col in range(2):
                lbl = tk.Label(self._ch_frame, relief="flat", bd=1)
                lbl.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
                self._ch_labels.append(lbl)
                self._ch_photos.append(None)
        self._ch_frame.grid_columnconfigure(0, weight=1)
        self._ch_frame.grid_columnconfigure(1, weight=1)

        self._ch_name_lbl = tk.Label(s, text="", font=("Helvetica", 8, "bold"),
                                     anchor="w", wraplength=250)
        self._ch_name_lbl.pack(fill="x", padx=14, pady=(0, 4))

        self._hsep(s)
        self._sec(s, "Metadata")

        # Scrollable metadata text
        meta_frame = tk.Frame(s)
        meta_frame.pack(fill="both", expand=True, padx=8, pady=4)
        meta_vsb = tk.Scrollbar(meta_frame, orient="vertical")
        self._meta_text = tk.Text(meta_frame, wrap="word", relief="flat", bd=0,
                                  font=("Courier", 8), state="disabled",
                                  yscrollcommand=meta_vsb.set)
        meta_vsb.configure(command=self._meta_text.yview)
        self._meta_text.pack(side="left", fill="both", expand=True)
        meta_vsb.pack(side="right", fill="y")

        # Key metadata fields to show (in order)
        self._META_KEYS = [
            ("Date",       ["REC_DATE", "REC_TIME"]),
            ("Pixels",     ["SCAN_PIXELS"]),
            ("Size (m)",   ["SCAN_RANGE"]),
            ("Offset (m)", ["SCAN_OFFSET"]),
            ("Bias (V)",   ["BIAS"]),
            ("Scan dir",   ["SCAN_DIR"]),
            ("Angle",      ["SCAN_ANGLE"]),
            ("Temp (K)",   ["REC_TEMP"]),
            ("Comment",    ["COMMENT"]),
            ("Clip low",   ["Clip_percentile_Lower"]),
            ("Clip high",  ["Clip_percentile_Higher"]),
        ]

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._statusbar = bar
        self._status_lbl = tk.Label(bar, textvariable=self._status,
                                    font=("Helvetica", 8), anchor="w")
        self._status_lbl.pack(side="left", padx=12)

    # ──────────────────────────────────────────────────────────────────────
    # Mode switching
    # ──────────────────────────────────────────────────────────────────────

    def _switch_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        t = THEMES["dark" if self.dark_mode.get() else "light"]

        if mode == "convert":
            self._browse_frame.pack_forget()
            self._conv_frame.pack(fill="both", expand=True)
            self._browse_sidebar.pack_forget()
            self._conv_sidebar.pack(fill="both", expand=True)
            self._status.set("Ready")
        else:
            self._conv_frame.pack_forget()
            self._browse_frame.pack(fill="both", expand=True)
            self._conv_sidebar.pack_forget()
            self._browse_sidebar.pack(fill="both", expand=True)
            self._repaint(self._browse_sidebar, t, "sidebar")
            n = len(self._grid._groups)
            self._status.set(f"{n} scan(s) loaded" if n else "Open a folder to browse")

        self._update_tabs()
        self._apply_theme()

    def _update_tabs(self) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        for btn, name in ((self._tab_convert, "convert"), (self._tab_browse, "browse")):
            active = (self._mode == name)
            btn.configure(
                bg=t["tab_act"] if active else t["tab_inact"],
                fg=t["fg"],
                relief="flat",
            )

    # ──────────────────────────────────────────────────────────────────────
    # Browse callbacks
    # ──────────────────────────────────────────────────────────────────────

    def _open_browse_folder(self) -> None:
        d = filedialog.askdirectory(title="Open output folder to browse")
        if not d:
            return
        self._switch_mode("browse")
        groups = scan_output_folder(Path(d))
        self._grid.load(groups)
        n = len(groups)
        self._status.set(f"{n} scan(s) loaded from {d}")
        self._clear_browse_sidebar()

    def _on_group_select(self, g: DatGroup) -> None:
        self._ch_name_lbl.configure(text=g.stem)
        self._load_channel_thumbnails(g.pngs)
        self._load_metadata(g.sxm)
        self._status.set(f"Selected: {g.stem}  |  {len(g.pngs)} channel(s)"
                         + ("  |  SXM found" if g.sxm else "  |  no SXM"))

    def _load_channel_thumbnails(self, pngs: List[Path]) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        # Show up to 4 channels in 2x2 grid
        ordered = sorted(pngs)
        # Z forward first, then Z backward, Current forward, Current backward
        def _key(p):
            n = p.name
            return (0 if "Z" in n and "forward" in n else
                    1 if "Z" in n and "backward" in n else
                    2 if "Current" in n and "forward" in n else 3)
        ordered = sorted(pngs, key=_key)[:4]

        for i in range(4):
            if i < len(ordered) and ordered[i].exists():
                try:
                    img = Image.open(ordered[i]).convert("RGB")
                    img.thumbnail((110, 90), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self._ch_photos[i] = photo
                    self._ch_labels[i].configure(image=photo, bg=t["sidebar_bg"],
                                                 text="", compound="none")
                except Exception:
                    self._ch_labels[i].configure(image="", text="err",
                                                 bg=t["sidebar_bg"], fg=t["sub_fg"])
            else:
                self._ch_photos[i] = None
                self._ch_labels[i].configure(image="", text="",
                                             bg=t["sidebar_bg"])

    def _load_metadata(self, sxm: Optional[Path]) -> None:
        self._meta_text.configure(state="normal")
        self._meta_text.delete("1.0", "end")

        if sxm and sxm.exists():
            hdr = parse_sxm_header(sxm)
            lines = []
            for label, keys in self._META_KEYS:
                vals = [hdr.get(k, "").strip() for k in keys if hdr.get(k, "").strip()]
                if vals:
                    v = "  ".join(vals)
                    lines.append(f"{label:<12}  {v}")
            self._meta_text.insert("end", "\n".join(lines))
        else:
            self._meta_text.insert("end", "(no SXM file found)")

        self._meta_text.configure(state="disabled")

    def _clear_browse_sidebar(self) -> None:
        self._ch_name_lbl.configure(text="")
        t = THEMES["dark" if self.dark_mode.get() else "light"]
        for i, lbl in enumerate(self._ch_labels):
            self._ch_photos[i] = None
            lbl.configure(image="", text="", bg=t["sidebar_bg"])
        self._meta_text.configure(state="normal")
        self._meta_text.delete("1.0", "end")
        self._meta_text.configure(state="disabled")

    # ──────────────────────────────────────────────────────────────────────
    # Widget helpers
    # ──────────────────────────────────────────────────────────────────────

    def _sec(self, parent, text: str) -> None:
        tk.Label(parent, text=text, font=("Helvetica", 9, "bold"), anchor="w"
                 ).pack(fill="x", padx=14, pady=(10, 2))

    def _hsep(self, parent) -> None:
        tk.Frame(parent, height=1).pack(fill="x", padx=10, pady=5)

    def _folder_row(self, parent, label: str, var: tk.StringVar, cmd) -> None:
        f = tk.Frame(parent)
        f.pack(fill="x", padx=16, pady=4)
        tk.Label(f, text=label, width=13, anchor="w").pack(side="left")
        tk.Entry(f, textvariable=var, relief="flat", bd=2).pack(
            side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(f, text="Browse", relief="flat", cursor="hand2",
                  font=("Helvetica", 8), command=cmd).pack(side="right")

    def _slider_row(self, parent, label: str, var: tk.DoubleVar,
                    from_: float, to: float) -> None:
        f = tk.Frame(parent)
        f.pack(fill="x", padx=16, pady=4)
        tk.Label(f, text=label, width=13, anchor="w").pack(side="left")
        tk.Scale(f, variable=var, from_=from_, to=to, resolution=0.5,
                 orient="horizontal", length=170, sliderlength=14,
                 relief="flat", bd=0, highlightthickness=0).pack(side="left")

    # ──────────────────────────────────────────────────────────────────────
    # Theme
    # ──────────────────────────────────────────────────────────────────────

    def _apply_theme(self) -> None:
        t = THEMES["dark" if self.dark_mode.get() else "light"]

        for w in (self.root, self._body, self._left, self._tabbar,
                  self._conv_frame, self._browse_frame):
            try:
                w.configure(bg=t["main_bg"])
            except Exception:
                pass

        for w in (self._sidebar, self._conv_sidebar, self._browse_sidebar,
                  self._adv_frame, self._ch_frame):
            try:
                w.configure(bg=t["sidebar_bg"])
            except Exception:
                pass

        self._repaint(self._conv_frame,    t, "main")
        self._repaint(self._conv_sidebar,  t, "sidebar")
        self._repaint(self._browse_sidebar, t, "sidebar")
        self._repaint(self._tabbar,         t, "main")

        self.log_text.configure(bg=t["log_bg"], fg=t["log_fg"], insertbackground=t["fg"])
        for tag, col in (("ok", t["ok_fg"]), ("err", t["err_fg"]),
                         ("warn", t["warn_fg"]), ("info", t["log_fg"])):
            self.log_text.tag_config(tag, foreground=col)

        self._meta_text.configure(bg=t["sidebar_bg"], fg=t["fg"],
                                  insertbackground=t["fg"])

        self.run_btn.configure(bg=t["accent_bg"], fg=t["accent_fg"],
                               activebackground=t["accent_bg"],
                               activeforeground=t["accent_fg"])

        self._statusbar.configure(bg=t["status_bg"])
        self._status_lbl.configure(bg=t["status_bg"], fg=t["status_fg"])

        self._conv_footer.configure(bg=t["sidebar_bg"], fg=t["sub_fg"])
        self._fcount_lbl.configure(bg=t["sidebar_bg"], fg=t["sub_fg"])
        self._ch_name_lbl.configure(bg=t["sidebar_bg"], fg=t["fg"])

        self._grid.apply_theme(t)
        self._update_tabs()

    def _repaint(self, widget, t: dict, zone: str = "main") -> None:
        bg = t["main_bg"] if zone == "main" else t["sidebar_bg"]
        cls = widget.winfo_class()
        try:
            if cls == "Frame":
                widget.configure(bg=bg)
            elif cls == "Label":
                widget.configure(bg=bg, fg=t["fg"])
            elif cls == "Button":
                widget.configure(bg=t["btn_bg"], fg=t["btn_fg"],
                                 activebackground=bg, activeforeground=t["fg"],
                                 relief="flat")
            elif cls == "Checkbutton":
                widget.configure(bg=bg, fg=t["fg"], selectcolor=t["entry_bg"],
                                 activebackground=bg, activeforeground=t["fg"])
            elif cls == "Entry":
                widget.configure(bg=t["entry_bg"], fg=t["fg"],
                                 insertbackground=t["fg"], relief="flat")
            elif cls == "Scale":
                widget.configure(bg=bg, fg=t["fg"],
                                 troughcolor=t["entry_bg"],
                                 activebackground=t["accent_bg"])
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._repaint(child, t, zone)

    def _toggle_theme(self) -> None:
        self.dark_mode.set(not self.dark_mode.get())
        self._theme_btn.configure(
            text="Light mode" if self.dark_mode.get() else "Dark mode")
        self._apply_theme()

    # ──────────────────────────────────────────────────────────────────────
    # Advanced toggle
    # ──────────────────────────────────────────────────────────────────────

    def _toggle_adv(self) -> None:
        if self._adv_vis:
            self._adv_frame.pack_forget()
            self._adv_btn.configure(text="[+] Advanced")
        else:
            self._adv_frame.pack(fill="x")
            t = THEMES["dark" if self.dark_mode.get() else "light"]
            self._repaint(self._adv_frame, t, "sidebar")
            self._adv_btn.configure(text="[-] Advanced")
        self._adv_vis = not self._adv_vis

    # ──────────────────────────────────────────────────────────────────────
    # File count
    # ──────────────────────────────────────────────────────────────────────

    def _update_count(self) -> None:
        d = self.input_dir.get().strip()
        if d and Path(d).is_dir():
            n = len(list(Path(d).glob("*.dat")))
            self._fcount_var.set(f"{n} .dat file(s) in input folder")
            self._status.set(f"{n} .dat file(s) found")
        else:
            self._fcount_var.set("No folder selected")
            self._status.set("Ready")

    # ──────────────────────────────────────────────────────────────────────
    # Folder pickers
    # ──────────────────────────────────────────────────────────────────────

    def _browse_input(self) -> None:
        d = filedialog.askdirectory(title="Select input folder containing .dat files")
        if d:
            self.input_dir.set(d)

    def _browse_output(self) -> None:
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir.set(d)

    # ──────────────────────────────────────────────────────────────────────
    # Log
    # ──────────────────────────────────────────────────────────────────────

    def _log(self, msg: str, tag: str = "info") -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n", tag)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _poll_log(self) -> None:
        try:
            while True:
                rec = self.log_queue.get_nowait()
                msg = rec.getMessage()
                tag = ("err"  if rec.levelno >= logging.ERROR  else
                       "warn" if rec.levelno == logging.WARNING else
                       "ok"   if "[OK]" in msg else "info")
                self._log(msg, tag)
        except queue.Empty:
            pass
        self.root.after(80, self._poll_log)

    # ──────────────────────────────────────────────────────────────────────
    # Conversion
    # ──────────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        if self._running:
            return
        in_dir  = self.input_dir.get().strip()
        out_dir = self.output_dir.get().strip()
        if not in_dir:
            self._log("ERROR: Please select an input folder.", "err"); return
        if not out_dir:
            self._log("ERROR: Please select an output folder.", "err"); return
        if not self.do_png.get() and not self.do_sxm.get():
            self._log("ERROR: Select at least one output format.", "err"); return
        if not Path(in_dir).is_dir():
            self._log(f"ERROR: Input folder not found: {in_dir}", "err"); return

        self._running = True
        self.run_btn.configure(text="  Running...  ", state="disabled")
        self._status.set("Converting...")
        self._clear_log()
        handler = QueueHandler(self.log_queue)
        handler.setLevel(logging.DEBUG)
        threading.Thread(
            target=self._worker, args=(in_dir, out_dir, handler), daemon=True
        ).start()

    def _worker(self, in_dir: str, out_dir: str, handler: QueueHandler) -> None:
        logger = logging.getLogger("nanonis_tools")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        clip_low  = self.clip_low.get()
        clip_high = self.clip_high.get()
        in_path   = Path(in_dir)
        out_path  = Path(out_dir)
        try:
            if self.do_png.get():
                from nanonis_tools.dats_to_pngs import main as png_main
                logger.info("-- PNG conversion --")
                png_main(src=in_path, out_root=out_path / "png",
                         clip_low=clip_low, clip_high=clip_high, verbose=True)

            if self.do_sxm.get():
                from nanonis_tools.dat_sxm_cli import convert_dat_to_sxm
                logger.info("-- SXM conversion --")
                files = sorted(in_path.glob("*.dat"))
                if not files:
                    logger.warning("No .dat files found in %s", in_path)
                else:
                    sxm_out = out_path / "sxm"
                    sxm_out.mkdir(parents=True, exist_ok=True)
                    errors = {}
                    logger.info("Found %d .dat file(s)", len(files))
                    for i, dat in enumerate(files, 1):
                        logger.info("[%d/%d] %s ...", i, len(files), dat.name)
                        try:
                            convert_dat_to_sxm(dat, sxm_out, DEFAULT_CUSHION,
                                               clip_low, clip_high)
                        except Exception as exc:
                            logger.error("FAILED %s: %s", dat.name, exc)
                            errors[dat.name] = str(exc)
                    if errors:
                        import json as _j
                        (sxm_out / "errors.json").write_text(_j.dumps(errors, indent=2))
                        logger.warning("%d file(s) failed -- see errors.json", len(errors))
                    else:
                        logger.info("All SXM files processed successfully.")
                    logger.info("Output: %s", sxm_out)
        except Exception as exc:
            logger.error("Unexpected error: %s", exc)
        finally:
            logger.removeHandler(handler)
            self.root.after(0, lambda: self._done(out_dir))

    def _done(self, out_dir: str) -> None:
        self._running = False
        self.run_btn.configure(text="  RUN  ", state="normal")
        self._status.set("Done")
        # Auto-load browse view with the output folder
        groups = scan_output_folder(Path(out_dir))
        if groups:
            self._grid.load(groups)
            self._switch_mode("browse")
            self._status.set(f"Done -- {len(groups)} scan(s) ready to browse")

    # ──────────────────────────────────────────────────────────────────────
    # Close
    # ──────────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        save_config({
            "dark_mode":  self.dark_mode.get(),
            "input_dir":  self.input_dir.get(),
            "output_dir": self.output_dir.get(),
            "do_png":     self.do_png.get(),
            "do_sxm":     self.do_sxm.get(),
            "clip_low":   self.clip_low.get(),
            "clip_high":  self.clip_high.get(),
        })
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ProbeFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
