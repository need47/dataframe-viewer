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


class DataFrameViewer(App):
    """A Textual app to view dataframe interactively."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("pageup", "page_up", "Page Up"),
        ("pagedown", "page_down", "Page Down"),
        ("t", "toggle_row_labels", "Toggle Row Labels"),
    ]

    def __init__(self, df: pl.DataFrame):
        super().__init__()
        self.df = df
        self.show_row_labels = False  # Row labels hidden by default

        # Reopen stdin to /dev/tty for proper terminal interaction
        if not sys.stdin.isatty():
            tty = open("/dev/tty")
            os.dup2(tty.fileno(), sys.stdin.fileno())

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield DataTable()

    def on_mount(self) -> None:
        """Set up the DataTable when app starts."""
        self._populate_table()

    def on_key(self, event) -> None:
        """Handle key events."""
        table = self.query_one(DataTable)

        if event.key in ("home", "g"):
            table.move_cursor(row=0)
        elif event.key in ("end", "G"):
            table.move_cursor(row=table.row_count - 1)

    def action_toggle_row_labels(self) -> None:
        """Toggle row labels visibility."""
        self.show_row_labels = not self.show_row_labels
        self._populate_table()

    def _populate_table(self) -> None:
        """Populate or repopulate the table with current settings."""
        table = self.query_one(DataTable)

        # Clear existing table
        table.clear(columns=True)

        # Add columns with justified headers based on dtype
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            style_config = STYLES.get(str(dtype), {"style": "green", "justify": "left"})
            justify = style_config["justify"]
            # Create column header with justification
            table.add_column(Text(col, justify=justify))

        # Add rows with colored cells based on dtype
        for row_idx, row in enumerate(self.df.rows()):
            # Convert values to strings, handling None
            formatted_row = []

            for val, dtype in zip(row, self.df.dtypes):
                # Get the style config for this data type
                style_config = STYLES.get(
                    str(dtype), {"style": "green", "justify": "left"}
                )
                color = style_config["style"]
                justify = style_config["justify"]

                # Format the value
                if val is None:
                    text_val = "-"
                elif str(dtype).startswith("Float"):
                    text_val = f"{val:.4g}"
                else:
                    text_val = str(val)

                # Create a styled Text object with justification
                formatted_row.append(Text(text_val, style=color, justify=justify))

            # Add row with label (1-based index) if enabled
            label = str(row_idx + 1) if self.show_row_labels else None
            table.add_row(*formatted_row, label=label)

        # Enable cursor and focus
        table.cursor_type = "row"
        table.focus()


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
    DataFrameViewer(df).run()


if __name__ == "__main__":
    main()
