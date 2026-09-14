"""
src/console.py
--------------
Small helpers that make the command-line output pleasant to read: framed
sections, rules, aligned columns, score bars and a little colour.

Design rules
    * Pure presentation. Nothing in here knows about RAG; main.py, ingest.py
      and the evaluation scripts call these helpers to print what they
      already computed.
    * Degrades gracefully. Colour is used only when stdout is a real
      terminal (not a pipe or a file) and the user has not set NO_COLOR;
      box-drawing characters fall back to plain ASCII (+, -, |) when the
      console cannot print UTF-8 or when RAG_ASCII=1 is set. Piped output
      therefore stays clean and grep-able.
    * Width-aware. Frames follow the terminal width, re-measured whenever a
      section opens (capped at 140 columns; RAG_WIDTH=<n> forces a value;
      piped output uses 100) and long lines are wrapped inside the frame.

Typical use
    from src.console import ui
    ui.banner("Multi-Document RAG", "Aurora Dynamics knowledge base")
    ui.section("Answer")
    ui.line("Employees receive 20 paid annual leave days [2][3].")
    ui.end()
"""

import os
import re
import shutil
import sys


class Console:
    """Stateful printer: remembers whether colour/Unicode are available."""

    # ANSI escape codes (only emitted when colour is enabled)
    _CODES = {
        "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m", "italic": "\033[3m",
        "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m", "blue": "\033[34m",
        "magenta": "\033[35m", "cyan": "\033[36m", "white": "\033[97m", "grey": "\033[90m",
    }

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.color = self._detect_color()
        self.unicode = self._detect_unicode()
        self.width = self._measure_width()
        # Glyph set: Unicode box drawing, or ASCII when the console cannot show it.
        if self.unicode:
            self.g = dict(h="─", v="│", tl="┌", tr="┐", bl="└", br="┘", dh="═", dv="║",
                          dtl="╔", dtr="╗", dbl="╚", dbr="╝", bullet="•", star="★",
                          dot="·", ok="✔", warn="!", arrow="❯", block="█", light="░")
        else:
            self.g = dict(h="-", v="|", tl="+", tr="+", bl="+", br="+", dh="=", dv="|",
                          dtl="+", dtr="+", dbl="+", dbr="+", bullet="*", star="*",
                          dot="-", ok="OK", warn="!", arrow=">", block="#", light=".")
        self._in_box = False

    # ------------------------------------------------------------------ setup
    def _detect_color(self) -> bool:
        if os.getenv("NO_COLOR") or os.getenv("RAG_NO_COLOR"):
            return False
        if os.getenv("RAG_COLOR"):          # force colour even through a pipe / pager
            return True
        if not hasattr(self.stream, "isatty") or not self.stream.isatty():
            return False
        if os.name == "nt":
            # Ask the Windows console to honour ANSI escape sequences.
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                mode = ctypes.c_uint32()
                if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            except Exception:  # pragma: no cover - best effort only
                return False
        return True

    def _detect_unicode(self) -> bool:
        if os.getenv("RAG_ASCII"):
            return False
        encoding = (getattr(self.stream, "encoding", "") or "").lower()
        return "utf" in encoding

    # ------------------------------------------------------------------ styling
    def style(self, text: str, *codes: str) -> str:
        """Wrap text in ANSI codes (no-op when colour is disabled)."""
        if not self.color or not codes:
            return text
        return "".join(self._CODES[c] for c in codes) + str(text) + self._CODES["reset"]

    def bold(self, text): return self.style(text, "bold")
    def dim(self, text): return self.style(text, "grey")
    def accent(self, text): return self.style(text, "cyan", "bold")
    def good(self, text): return self.style(text, "green")
    def bad(self, text): return self.style(text, "red")
    def warn(self, text): return self.style(text, "yellow")

    @staticmethod
    def _visible_len(text: str) -> int:
        """Length of a string ignoring ANSI escape sequences."""
        out, skip = 0, False
        for ch in text:
            if ch == "\033":
                skip = True
            elif skip:
                if ch == "m":
                    skip = False
            else:
                out += 1
        return out

    # ------------------------------------------------------------------ primitives
    def print(self, text: str = "") -> None:
        print(text, file=self.stream)

    def blank(self) -> None:
        self.print("")

    # ------------------------------------------------------------------ width
    MAX_WIDTH = 140   # prose wider than this gets hard to read; RAG_WIDTH overrides

    def _measure_width(self) -> int:
        """Current terminal width, capped; RAG_WIDTH=<cols> forces a value."""
        forced = os.getenv("RAG_WIDTH")
        if forced and forced.isdigit():
            return max(60, int(forced))
        cols = shutil.get_terminal_size((100, 24)).columns
        # A pipe or a file has no width: use a fixed one so logs are stable.
        if not self.stream.isatty():
            cols = 100
        return max(60, min(cols, self.MAX_WIDTH))

    def rule(self, title: str = "", double: bool = False) -> None:
        """A horizontal rule, optionally with a title:  ── Title ─────"""
        self.width = self._measure_width()   # re-measure: the window may have been resized
        h = self.g["dh"] if double else self.g["h"]
        if title:
            head = f"{h * 2} {self.bold(title)} "
            pad = max(0, self.width - self._visible_len(head))
            self.print(head + h * pad)
        else:
            self.print(h * self.width)

    def banner(self, title: str, subtitle: str = "") -> None:
        """Double-lined box with a centred title (used once at start-up)."""
        self.width = self._measure_width()
        g = self.g
        inner = self.width - 2
        self.print(g["dtl"] + g["dh"] * inner + g["dtr"])
        for text, styler in ((title, self.accent), (subtitle, self.dim)):
            if not text:
                continue
            pad_total = inner - len(text)
            left = pad_total // 2
            self.print(g["dv"] + " " * left + styler(text) + " " * (pad_total - left) + g["dv"])
        self.print(g["dbl"] + g["dh"] * inner + g["dbr"])

    # ------------------------------------------------------------------ framed sections
    def section(self, title: str, hint: str = "") -> None:
        """Open a framed section:  ┌─ Title  (hint) ───────────"""
        self.width = self._measure_width()   # frames follow the window as it is resized
        g = self.g
        head = f"{g['tl']}{g['h']} {self.bold(title)}"
        if hint:
            head += f"  {self.dim(hint)}"
        head += " "
        pad = max(0, self.width - self._visible_len(head) - 1)
        self.print(head + g["h"] * pad + g["tr"])
        self._in_box = True

    def line(self, text: str = "", indent: int = 2, hang: int = 0) -> None:
        """One line inside the open section, wrapped to the frame width.

        `hang` indents continuation lines further, so a wrapped value stays
        aligned under its first line instead of under its label.
        """
        g = self.g
        avail = self.width - 2 - indent - 1
        if text == "":
            self.print(f"{g['v']}{' ' * (self.width - 2)}{g['v']}")
            return
        for piece in self._wrap_visible(text, avail, hang):
            pad = max(0, self.width - 2 - indent - self._visible_len(piece))
            self.print(f"{g['v']}{' ' * indent}{piece}{' ' * pad}{g['v']}")

    def _wrap_visible(self, text: str, width: int, hang: int = 0) -> list[str]:
        """Greedy word wrap that measures VISIBLE characters only.

        textwrap.wrap() counts ANSI colour codes as text, so a coloured table
        row that fits on screen was being broken onto two lines. Runs of
        spaces (column padding) are preserved, and an open colour is closed at
        the end of a wrapped piece and re-opened on the next one.
        """
        if self._visible_len(text) <= width:
            return [text]
        lines: list[str] = []
        current: list[str] = []
        current_len = 0
        limit = width
        for word in text.split(" "):
            extra = self._visible_len(word) + (1 if current else 0)
            if current and current_len + extra > limit:
                lines.append(" ".join(current))
                current, current_len, limit = [word], self._visible_len(word), width - hang
            else:
                current.append(word)
                current_len += extra
        lines.append(" ".join(current))
        # Keep colours per line: close an open style at a break, reopen it after.
        out, active = [], ""
        for index, piece in enumerate(lines):
            piece = (" " * hang if index else "") + active + piece
            codes = re.findall("\033\\[[0-9;]*m", piece)
            active = "" if not codes or codes[-1] == "\033[0m" else codes[-1]
            if active:
                piece += "\033[0m"
            out.append(piece)
        return out

    def end(self) -> None:
        """Close the framed section."""
        g = self.g
        self.print(g["bl"] + g["h"] * (self.width - 2) + g["br"])
        self._in_box = False

    # ------------------------------------------------------------------ composites
    def columns(self, rows: list[list[str]], widths: list[int], indent: int = 2,
                inside: bool = False, header: bool = False) -> None:
        """Print aligned columns. Cells longer than their width are truncated."""
        for r_index, row in enumerate(rows):
            cells = []
            for value, width in zip(row, widths):
                value = str(value)
                if width and self._visible_len(value) > width:
                    value = self._truncate_visible(value, width)
                pad = width - self._visible_len(value) if width else 0
                cells.append(value + " " * max(0, pad))
            text = "  ".join(cells).rstrip()
            if header and r_index == 0:
                text = self.bold(text)
            if inside:
                self.line(text, indent)
            else:
                self.print(" " * indent + text)

    def _truncate_visible(self, text: str, width: int) -> str:
        """Cut to `width` visible characters (+ ellipsis), passing ANSI codes through
        untouched and closing any open style so it cannot leak into the next cell."""
        ellipsis = "…" if self.unicode else ".."
        keep = width - len(ellipsis)
        out, shown, in_code, styled = [], 0, False, False
        for ch in text:
            if ch == "":
                in_code, styled = True, True
            if in_code:
                out.append(ch)
                if ch == "m":
                    in_code = False
                continue
            if shown == keep:
                break          # later codes (usually the closing reset) are re-added below
            out.append(ch)
            shown += 1
        return "".join(out) + ellipsis + ("[0m" if styled else "")

    def bar(self, value, width: int = 20) -> str:
        """A textual progress/score bar for a 0-1 value:  ████████░░░░"""
        if value is None:
            return self.dim(self.g["light"] * width)
        filled = int(round(float(value) * width))
        colour = "green" if value >= 0.8 else "yellow" if value >= 0.5 else "red"
        return self.style(self.g["block"] * filled, colour) + self.dim(self.g["light"] * (width - filled))

    def status(self, ok: bool, text: str) -> None:
        """A one-line confirmation:  ✔ Strategy set to llm_hyde"""
        mark = self.good(self.g["ok"]) if ok else self.bad(self.g["warn"])
        self.print(f"  {mark} {text}")

    def kv(self, label: str, value: str, label_width: int = 18, indent: int = 2, inside: bool = False) -> None:
        """Label/value pair with an aligned column:  Faithfulness      1.00"""
        text = f"{self.dim(label.ljust(label_width))}{value}"
        if inside:
            self.line(text, indent, hang=label_width)
        else:
            self.print(" " * indent + text)


# One shared instance for the whole process.
ui = Console()
