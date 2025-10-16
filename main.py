import sys
import termios
import tty
import os
import argparse
from dataclasses import dataclass
from typing import Dict
from io import StringIO

import polars as pl
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from rich.live import Live
from rich.layout import Layout

console = Console()


@dataclass
class Keypress:
    char: str
    code: int


def read_key() -> Keypress:
    """Read a single key (non-blocking for modifiers) supporting arrow keys, page, ctrl combos."""
    # If stdin is not a TTY (e.g., piped input), open /dev/tty directly
    if not sys.stdin.isatty():
        tty_fd = os.open("/dev/tty", os.O_RDONLY)
    else:
        tty_fd = sys.stdin.fileno()

    old_settings = termios.tcgetattr(tty_fd)
    try:
        tty.setraw(tty_fd)
        ch = os.read(tty_fd, 1).decode("utf-8")
        # Escape sequences for arrows / page keys
        if ch == "\x1b":
            seq = os.read(tty_fd, 2).decode("utf-8")  # [A or [1 etc
            full = ch + seq
            # For sequences like ESC[1~ or ESC[4~, read the trailing character
            if seq and len(seq) > 1 and seq[1] in "1456":
                extra = os.read(tty_fd, 1).decode("utf-8")
                full += extra
            return Keypress(full, ord(ch))
        return Keypress(ch, ord(ch))
    finally:
        termios.tcsetattr(tty_fd, termios.TCSADRAIN, old_settings)
        if not sys.stdin.isatty():
            os.close(tty_fd)


def handle_keypress(start: int, page: int, total: int) -> int:
    """Process keyboard input and return new start position.

    Returns:
        New start position, or -1 if user wants to quit
    """
    key = read_key()

    # Quit
    if key.char == "q":
        return -1

    # Enter key (PageDown) ord=13 or '\r'
    elif key.char == "\r" or key.code == 13:
        new_start = start + page
        # Don't go past the end; last page may show fewer rows
        if new_start < total:
            return new_start
        # If at the end, exit
        elif new_start >= total:
            return -1

    # Ctrl+F (forward page) ord=6
    elif key.code == 6:
        return min(start + page, total - 1)

    # Ctrl+B (back page) ord=2
    elif key.code == 2:
        return max(start - page, 0)

    # Home key (go to first page) - ESC[H or ESC[1~
    elif key.char in ("\x1b[H", "\x1b[1~"):
        return 0

    # End key (go to last page) - ESC[F or ESC[4~
    elif key.char in ("\x1b[F", "\x1b[4~"):
        # Go to last page showing remaining rows
        return max(0, total - page)

    # Arrow up/down and other escape sequences
    elif key.char.startswith("\x1b["):
        if len(key.char) >= 3:
            code = key.char[2]
            if code == "A":  # Up arrow
                return max(start - 1, 0)
            elif code == "B":  # Down arrow
                return min(start + 1, total - 1)
        # Check full sequence for PageUp/PageDown
        if key.char == "\x1b[5~":  # PageUp
            return max(start - page, 0)
        elif key.char == "\x1b[6~":  # PageDown
            return min(start + page, total - 1)

    # No change
    return start


def dtype_style_map() -> Dict[str, Dict[str, str]]:
    return {
        "Int64": {"style": "cyan", "justify": "right"},
        "Int32": {"style": "cyan", "justify": "right"},
        "UInt32": {"style": "cyan", "justify": "right"},
        "Float64": {"style": "magenta", "justify": "right"},
        "Float32": {"style": "magenta", "justify": "right"},
        "Utf8": {"style": "green", "justify": "left"},
        "Boolean": {"style": "yellow", "justify": "center"},
        "Date": {"style": "blue", "justify": "center"},
        "Datetime": {"style": "blue", "justify": "center"},
    }


def build_table(df: pl.DataFrame, start: int, end: int, box_style=box.SIMPLE) -> Table:
    styles = dtype_style_map()

    # Explicitly enable headers; some environments (piped / non-TTY via uv run | head) may
    # suppress them if Rich detects non-interactive output. Setting show_header=True
    # and a header_style makes headers reliably visible.
    # padding=(0,1,0,1) = (top, right, bottom, left) removes vertical padding
    table = Table(
        box=box_style,
        expand=True,
        padding=(0, 1, 0, 1),
    )

    # Add columns with styles based on dtype
    for col, dtype in zip(df.columns, df.dtypes):
        dtype_name = str(dtype)
        meta = styles.get(dtype_name, {"style": "green", "justify": "left"})
        table.add_column(
            col, style=meta["style"], justify=meta["justify"], overflow="fold"
        )

    # Add rows for the current slice
    for row in df.slice(start, end - start).rows():
        rendered = []
        for val, dtype in zip(row, df.dtypes):
            if val is None:
                rendered.append("-")
            else:
                if str(dtype).startswith("Float"):
                    rendered.append(f"{val:.4g}")
                else:
                    rendered.append(str(val))
        table.add_row(*rendered)
    return table


def build_status_bar(filename: str, start: int, end: int, total: int) -> Text:
    term_width = console.size.width
    left = filename if filename else "stdout"
    right = f"rows {start + 1}-{end} / {total}"
    padding = term_width - len(left) - len(right)
    if padding < 0:
        padding = 0
    text = Text()
    text.append(left, style="bold white on blue")
    text.append(" " * padding, style="on blue")
    text.append(right, style="bold white on blue")
    return text


def build_display(
    df: pl.DataFrame,
    filename: str,
    start: int,
    end: int,
    total: int,
    box_style=box.SIMPLE,
):
    """Build the complete display with table and status bar."""
    layout = Layout()
    layout.split_column(
        Layout(build_table(df, start, end, box_style), name="main"),
        Layout(build_status_bar(filename, start, end, total), name="footer", size=1),
    )
    return layout


def display_dataframe(df: pl.DataFrame, filename: str, box_style=box.SIMPLE):
    """Interactive dataframe viewer with keyboard navigation.

    Navigation:
      Up/Down: move 1 row
      PageUp/PageDown, Enter, or Ctrl+B/Ctrl+F: move one page
      Home/End: go to start/end
      q: quit
    """
    height = console.size.height

    # Account for:
    # - Table header takes 3 lines (top border + header text + separator with SIMPLE)
    # - Status bar takes 1 line
    # So data rows = total height - 4
    # This should give us ~50 rows in a standard 54-line terminal
    page = max(height - 4, 1)
    total = df.height
    start = 0

    # Use Rich Live for smooth, flicker-free updates
    # screen=True enables full screen mode with proper clearing
    # auto_refresh=False to manually control refresh timing
    with Live(
        build_display(df, filename, start, min(start + page, total), total, box_style),
        console=console,
        screen=True,
        auto_refresh=False,
    ) as live:
        while True:
            new_start = handle_keypress(start, page, total)

            # User wants to quit
            if new_start == -1:
                break

            # Only update display if position changed
            if new_start != start:
                start = new_start
                end = min(start + page, total)
                live.update(build_display(df, filename, start, end, total, box_style))
                live.refresh()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CSV viewer for the terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python main.py data.csv\n"
        "  python main.py data.csv --box rounded\n"
        "  cat data.csv | python main.py --box heavy\n"
        "  python main.py data.csv --box minimal_double_head\n",
    )
    parser.add_argument("file", nargs="?", help="CSV file to view (or read from stdin)")
    parser.add_argument(
        "--box",
        default="simple",
        help="Box style name (e.g., simple, rounded, minimal, heavy, double, ascii, minimal_double_head, etc.)",
    )

    args = parser.parse_args()

    # Convert user input to uppercase and get box style from rich.box module
    box_style_name = args.box.upper()
    try:
        box_style = getattr(box, box_style_name)
    except AttributeError:
        console.print(f"Error: Unknown box style '{args.box}'")
        console.print(
            f"Available styles: {', '.join([name.lower() for name in dir(box) if name.isupper()])}"
        )
        sys.exit(1)

    # Check if reading from stdin (pipe or redirect)
    if not sys.stdin.isatty():
        # Read CSV from stdin into memory first (stdin is not seekable)
        stdin_data = sys.stdin.read()
        df = pl.read_csv(StringIO(stdin_data))
        display_dataframe(df, "stdin", box_style)
    elif args.file:
        # Read from file
        filename = args.file
        if not os.path.exists(filename):
            console.print(f"File not found: {filename}")
            sys.exit(1)
        df = pl.read_csv(filename)
        display_dataframe(df, filename, box_style)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
