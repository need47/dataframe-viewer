import os
import sys

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


def _format_row(vals, dtypes, apply_justify=True) -> list[Text]:
    """Format a single row with proper styling and justification.

    Args:
        vals: The list of values in the row.
        dtypes: The list of data types corresponding to each value.
        apply_justify: Whether to apply justification styling. Defaults to True.
    """
    formatted_row = []

    for val, dtype in zip(vals, dtypes, strict=True):
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


# Pagination settings
INITIAL_BATCH_SIZE = 100  # Load this many rows initially
BATCH_SIZE = 50  # Load this many rows when scrolling


class DataFrameViewer(App):
    """A Textual app to view dataframe interactively."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("t", "toggle_row_labels", "Toggle Row Labels"),
    ]

    def __init__(self, df: pl.DataFrame):
        super().__init__()
        self.dataframe = df  # Original dataframe
        self.df = df  # Internal dataframe
        self.loaded_rows = 0  # Track how many rows are currently loaded

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        self.table = DataTable(zebra_stripes=True)
        yield self.table

    def on_mount(self) -> None:
        """Set up the DataTable when app starts."""
        self._setup_table()
        # Hide labels by default after initial load
        self.call_later(lambda: setattr(self.table, "show_row_labels", False))

    def on_key(self, event) -> None:
        """Handle key events."""
        # Restore original display
        if event.key == "r":
            self._setup_table()
            # Hide labels by default after initial load
            self.call_later(lambda: setattr(self.table, "show_row_labels", False))
        elif event.key in ("pagedown", "down"):
            # Let the table handle the navigation first
            self._check_and_load_more()

    def on_mouse_scroll_down(self, event) -> None:
        """Load more rows when scrolling down with mouse."""
        self._check_and_load_more()

    def action_toggle_row_labels(self) -> None:
        """Toggle row labels visibility using CSS property."""
        self.table.show_row_labels = not self.table.show_row_labels

    def _setup_table(self) -> None:
        """Setup the table for display."""
        # Reset to original dataframe
        self.df = self.dataframe
        self.loaded_rows = 0
        self.sorted_columns = {}

        self._setup_columns()
        self._load_rows(INITIAL_BATCH_SIZE)

    def _setup_columns(self) -> None:
        """Clear table and setup columns."""
        self.table.clear(columns=True)
        self.loaded_rows = 0

        # Add columns with justified headers
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            style_config = STYLES.get(str(dtype), {"style": "green", "justify": "left"})
            self.table.add_column(Text(col, justify=style_config["justify"]), key=col)

        self.table.cursor_type = "cell"
        self.table.focus()

    def _check_and_load_more(self) -> None:
        """Check if we need to load more rows and load them."""
        # If we've loaded everything, no need to check
        if self.loaded_rows >= len(self.df):
            return

        visible_row_count = self.table.size.height - self.table.header_height
        bottom_visible_row = self.table.scroll_y + visible_row_count

        # If visible area is close to the end of loaded rows, load more
        if bottom_visible_row >= self.loaded_rows - 10:
            self._load_rows(BATCH_SIZE)

    def _load_rows(self, count: int) -> None:
        """Load a batch of rows into the table."""
        start_idx = self.loaded_rows
        if start_idx >= len(self.df):
            return

        end_idx = min(start_idx + count, len(self.df))
        df_slice = self.df.slice(start_idx, end_idx - start_idx)

        for row_idx, row in enumerate(df_slice.rows(), start_idx):
            vals, dtypes = [], []
            for val, dtype in zip(row, self.df.dtypes):
                vals.append(val)
                dtypes.append(dtype)
            formatted_row = _format_row(vals, dtypes)
            # Always add labels so they can be shown/hidden via CSS
            rid = str(row_idx + 1)
            self.table.add_row(*formatted_row, key=rid, label=rid)

        self.loaded_rows = end_idx

        if count != INITIAL_BATCH_SIZE:
            self.notify(f"Loaded {self.loaded_rows}/{len(self.df)} rows", title="Load")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive CSV viewer for the terminal (Textual version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  python main.py data.csv\n"
        "  cat data.csv | python main.py\n",
    )
    parser.add_argument("file", nargs="?", help="CSV file to view (or read from stdin)")

    args = parser.parse_args()

    # Read from file
    if args.file:
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
