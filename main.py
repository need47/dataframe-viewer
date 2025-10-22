import argparse
import os
import sys
from io import StringIO

import polars as pl
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable

STYLES = {
    "Int64": {"style": "cyan", "justify": "right"},
    "Float64": {"style": "magenta", "justify": "right"},
    "String": {"style": "green", "justify": "left"},
    "Boolean": {"style": "yellow", "justify": "center"},
    "Date": {"style": "blue", "justify": "center"},
    "Datetime": {"style": "blue", "justify": "center"},
}

# Pagination settings
INITIAL_BATCH_SIZE = 100  # Load this many rows initially
BATCH_SIZE = 50  # Load this many rows when scrolling


class DataFrameViewer(App):
    """A Textual app to view dataframe interactively."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("t", "toggle_row_labels", "Toggle Row Labels"),
        ("c", "copy_cell", "Copy Cell"),
    ]

    def __init__(self, df: pl.DataFrame):
        super().__init__()
        self.df = df
        self.loaded_rows = 0  # Track how many rows are currently loaded
        self.total_rows = len(df)

        # Reopen stdin to /dev/tty for proper terminal interaction
        if not sys.stdin.isatty():
            tty = open("/dev/tty")
            os.dup2(tty.fileno(), sys.stdin.fileno())

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield DataTable(zebra_stripes=True)

    def action_toggle_row_labels(self) -> None:
        """Toggle row labels visibility using CSS property."""
        table = self.query_one(DataTable)
        table.show_row_labels = not table.show_row_labels

    def action_copy_cell(self) -> None:
        """Copy the current cell to clipboard."""
        import subprocess

        table = self.query_one(DataTable)
        row_idx = table.cursor_row
        col_idx = table.cursor_column

        # Get the cell value
        cell_str = str(self.df.item(row_idx, col_idx))

        # Copy to clipboard using xclip or pbcopy (macOS)
        try:
            subprocess.run(
                [
                    "pbcopy" if sys.platform == "darwin" else "xclip",
                    "-selection",
                    "clipboard",
                ],
                input=cell_str,
                text=True,
            )
            msg = f"Copied: {cell_str[:50]}"
            self.notify(msg)
        except FileNotFoundError:
            msg = f"Copied (clipboard tool not available): {cell_str[:50]}"
            self.notify(msg)

    def on_mount(self) -> None:
        """Set up the DataTable when app starts."""
        table = self.query_one(DataTable)
        self._setup_table_columns(table)
        self._load_rows(table, INITIAL_BATCH_SIZE)
        # Hide labels by default after initial load
        self.call_later(lambda: setattr(table, "show_row_labels", False))

    def on_key(self, event) -> None:
        """Handle key events."""
        table = self.query_one(DataTable)

        if event.key == "g":
            table.move_cursor(row=0)
        elif event.key == "G":
            # Load all remaining rows before jumping to end
            remaining = self.total_rows - self.loaded_rows
            if remaining > 0:
                self._load_rows(table, remaining)
            table.move_cursor(row=table.row_count - 1)
        elif event.key in ("pagedown", "down"):
            # Let the table handle the navigation first
            self._check_and_load_more()

    def on_mouse_scroll_down(self, event) -> None:
        """Load more rows when scrolling down with mouse."""
        self._check_and_load_more()

    def _setup_table_columns(self, table: DataTable) -> None:
        """Clear table and setup columns."""
        table.clear(columns=True)
        self.loaded_rows = 0

        # Add columns with justified headers
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            style_config = STYLES.get(str(dtype), {"style": "green", "justify": "left"})
            table.add_column(Text(col, justify=style_config["justify"]))

        table.cursor_type = "cell"
        table.focus()

    def _check_and_load_more(self) -> None:
        """Check if we need to load more rows and load them."""
        # If we've loaded everything, no need to check
        if self.loaded_rows >= self.total_rows:
            return

        table = self.query_one(DataTable)
        visible_row_count = table.size.height - table.header_height
        bottom_visible_row = table.scroll_y + visible_row_count

        # If visible area is close to the end of loaded rows, load more
        if bottom_visible_row >= self.loaded_rows - 10:
            self._load_rows(table, BATCH_SIZE)

    def _load_rows(self, table: DataTable, count: int) -> None:
        """Load a batch of rows into the table."""
        start_idx = self.loaded_rows
        if start_idx >= self.total_rows:
            return

        end_idx = min(start_idx + count, self.total_rows)
        df_slice = self.df.slice(start_idx, end_idx - start_idx)

        for offset, row in enumerate(df_slice.rows()):
            row_idx = start_idx + offset
            formatted_row = self._format_row(row)
            # Always add labels so they can be shown/hidden via CSS
            table.add_row(*formatted_row, label=str(row_idx + 1))

        self.loaded_rows = end_idx

    def _format_row(self, row) -> list[Text]:
        """Format a single row with proper styling and justification."""
        formatted_row = []

        for val, dtype in zip(row, self.df.dtypes):
            style_config = STYLES.get(str(dtype), {"style": "green", "justify": "left"})

            # Format the value
            if val is None:
                text_val = "-"
            elif str(dtype).startswith("Float"):
                text_val = f"{val:.4g}"
            else:
                text_val = str(val)

            formatted_row.append(
                Text(
                    text_val,
                    style=style_config["style"],
                    justify=style_config["justify"],
                )
            )

        return formatted_row


def main():
    parser = argparse.ArgumentParser(
        description="Interactive CSV viewer for the terminal (Textual version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python main.py data.csv\n"
        "  cat data.csv | python main.py\n",
    )
    parser.add_argument("file", nargs="?", help="CSV file to view (or read from stdin)")

    args = parser.parse_args()

    # Check if reading from stdin (pipe or redirect)
    if not sys.stdin.isatty():
        # Read CSV from stdin into memory first (stdin is not seekable)
        stdin_data = sys.stdin.read()
        df = pl.read_csv(StringIO(stdin_data))
    elif args.file:
        # Read from file
        filename = args.file
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            sys.exit(1)
        df = pl.read_csv(filename)
    else:
        parser.print_help()
        sys.exit(1)

    # Run the app
    app = DataFrameViewer(df)
    app.run()


if __name__ == "__main__":
    main()
