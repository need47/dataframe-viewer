import argparse
import os
import sys
from io import StringIO

import polars as pl
from rich.text import Text
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import DataTable

STYLES = {
    "Int64": {"style": "cyan", "justify": "right"},
    "Float64": {"style": "magenta", "justify": "right"},
    "String": {"style": "green", "justify": "left"},
    "Boolean": {"style": "yellow", "justify": "center"},
    "Date": {"style": "blue", "justify": "center"},
    "Datetime": {"style": "blue", "justify": "center"},
}


def _format_row(row, dtypes, apply_justify=True) -> list[Text]:
    """Format a single row with proper styling and justification."""
    formatted_row = []

    for val, dtype in zip(row, dtypes, strict=True):
        style_config = STYLES.get(str(dtype), {"style": "", "justify": ""})

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
                justify=style_config["justify"] if apply_justify else "",
            )
        )

    return formatted_row


class RowDetailScreen(ModalScreen):
    """Modal screen to display a single row's details."""

    BINDINGS = [
        ("q", "app.pop_screen", "Close"),
    ]

    CSS = """
    RowDetailScreen {
        align: center middle;
    }

    RowDetailScreen > DataTable {
        width: 80;
        border: solid $primary;
    }
    """

    def __init__(self, columns: list[str], row_values: list, dtypes: list):
        super().__init__()
        self.columns = columns
        self.row_values = row_values
        self.dtypes = dtypes

    def on_key(self, event) -> None:
        """Handle key events."""
        # Prevent Enter from propagating to parent screen
        if event.key == "enter":
            event.stop()

    def compose(self) -> ComposeResult:
        """Create the detail table."""
        detail_table = DataTable(zebra_stripes=True)

        # Add two columns: Column Name and Value
        detail_table.add_column("Column")
        detail_table.add_column("Value")

        # Add rows for each column with color based on dtype
        for col, val, dtype in zip(self.columns, self.row_values, self.dtypes):
            detail_table.add_row(
                *_format_row([col, val], [None, dtype], apply_justify=False)
            )

        yield detail_table


# Pagination settings
INITIAL_BATCH_SIZE = 100  # Load this many rows initially
BATCH_SIZE = 50  # Load this many rows when scrolling


class DataFrameViewer(App):
    """A Textual app to view dataframe interactively."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle Dark Mode"),
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
        self.table = DataTable(zebra_stripes=True)
        yield self.table

    def on_mount(self) -> None:
        """Set up the DataTable when app starts."""
        self._setup_table_columns()
        self._load_rows(INITIAL_BATCH_SIZE)
        # Hide labels by default after initial load
        self.call_later(lambda: setattr(self.table, "show_row_labels", False))

    def on_key(self, event) -> None:
        """Handle key events."""
        if event.key == "g":
            self.table.move_cursor(row=0)
        elif event.key == "G":
            # Load all remaining rows before jumping to end
            remaining = self.total_rows - self.loaded_rows
            if remaining > 0:
                self._load_rows(remaining)
            self.table.move_cursor(row=self.table.row_count - 1)
        elif event.key in ("pagedown", "down"):
            # Let the table handle the navigation first
            self._check_and_load_more()
        elif event.key == "enter":
            # Open row detail modal
            self._view_row_detail()

    def on_mouse_scroll_down(self, event) -> None:
        """Load more rows when scrolling down with mouse."""
        self._check_and_load_more()

    def action_toggle_row_labels(self) -> None:
        """Toggle row labels visibility using CSS property."""
        self.table.show_row_labels = not self.table.show_row_labels

    def action_copy_cell(self) -> None:
        """Copy the current cell to clipboard."""
        import subprocess

        row_idx = self.table.cursor_row
        col_idx = self.table.cursor_column

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
            self.notify(cell_str[:50], title="Clipboard")
        except FileNotFoundError:
            self.notify("clipboard tool not available", title="FileNotFound")

    def _setup_table_columns(self) -> None:
        """Clear table and setup columns."""
        self.table.clear(columns=True)
        self.loaded_rows = 0

        # Add columns with justified headers
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            style_config = STYLES.get(str(dtype), {"style": "green", "justify": "left"})
            self.table.add_column(Text(col, justify=style_config["justify"]))

        self.table.cursor_type = "cell"
        self.table.focus()

    def _check_and_load_more(self) -> None:
        """Check if we need to load more rows and load them."""
        # If we've loaded everything, no need to check
        if self.loaded_rows >= self.total_rows:
            return

        visible_row_count = self.table.size.height - self.table.header_height
        bottom_visible_row = self.table.scroll_y + visible_row_count

        # If visible area is close to the end of loaded rows, load more
        if bottom_visible_row >= self.loaded_rows - 10:
            self._load_rows(BATCH_SIZE)

    def _load_rows(self, count: int) -> None:
        """Load a batch of rows into the table."""
        start_idx = self.loaded_rows
        if start_idx >= self.total_rows:
            return

        end_idx = min(start_idx + count, self.total_rows)
        df_slice = self.df.slice(start_idx, end_idx - start_idx)

        for offset, row in enumerate(df_slice.rows()):
            row_idx = start_idx + offset
            formatted_row = _format_row(row, self.df.dtypes)
            # Always add labels so they can be shown/hidden via CSS
            self.table.add_row(*formatted_row, label=str(row_idx + 1))

        self.loaded_rows = end_idx

    def _view_row_detail(self) -> None:
        """Open a modal screen to view the selected row's details."""
        row_idx = self.table.cursor_row
        if row_idx < self.df.__len__():
            # Get the row data from the dataframe
            row_data = self.df.row(row_idx)
            # Get column names and dtypes
            columns = list(self.df.columns)
            dtypes = list(self.df.dtypes)
            # Push the modal screen
            self.push_screen(RowDetailScreen(columns, row_data, dtypes))


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
