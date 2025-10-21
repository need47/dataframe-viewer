import argparse
import os
import sys
from io import StringIO

import polars as pl
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable


class DataFrameViewer(App):
    """A Textual app to view dataframe interactively."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("pageup", "page_up", "Page Up"),
        ("pagedown", "page_down", "Page Down"),
    ]

    def __init__(self, df: pl.DataFrame):
        super().__init__()
        self.df = df

        # Reopen stdin to /dev/tty for proper terminal interaction
        if not sys.stdin.isatty():
            tty = open("/dev/tty")
            os.dup2(tty.fileno(), sys.stdin.fileno())

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield DataTable()

    def on_mount(self) -> None:
        """Set up the DataTable when app starts."""
        table = self.query_one(DataTable)
        styles = {
            "Int64": "cyan",
            "Int32": "cyan",
            "UInt32": "cyan",
            "Float64": "magenta",
            "Float32": "magenta",
            "Utf8": "green",
            "Boolean": "yellow",
            "Date": "blue",
            "Datetime": "blue",
        }

        # Add columns
        columns = self.df.columns
        table.add_columns(*columns)

        # Add rows with colored cells based on dtype
        for row in self.df.rows():
            # Convert values to strings, handling None
            formatted_row = []

            for val, dtype in zip(row, self.df.dtypes):
                # Get the color for this data type
                color = styles.get(str(dtype), "green")

                # Format the value
                if val is None:
                    text_val = "-"
                elif str(dtype).startswith("Float"):
                    text_val = f"{val:.4g}"
                else:
                    text_val = str(val)

                # Create a styled Text object
                formatted_row.append(Text(text_val, style=color))

            table.add_row(*formatted_row)

        # Enable cursor and focus
        table.cursor_type = "row"
        table.focus()

    def on_key(self, event) -> None:
        """Handle key events."""
        table = self.query_one(DataTable)

        if event.key == "home":
            table.move_cursor(row=0)
        elif event.key == "end":
            table.move_cursor(row=table.row_count - 1)


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
