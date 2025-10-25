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


class DataFrameViewer(App):
    """A Textual app to view dataframe interactively."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("t", "toggle_row_labels", "Toggle Row Labels"),
        ("r", "setup_table", "Reset Table"),
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
        self.action_setup_table()

    def action_toggle_row_labels(self) -> None:
        """Toggle row labels visibility using CSS property."""
        self.table.show_row_labels = not self.table.show_row_labels

    def action_setup_table(self) -> None:
        """Setup the table for display."""
        # Reset to original dataframe
        self.df = self.dataframe

        self._setup_columns()
        self._load_rows()

        # Hide labels by default after initial load
        self.call_later(lambda: setattr(self.table, "show_row_labels", False))

    def _setup_columns(self) -> None:
        """Clear table and setup columns."""
        self.table.clear(columns=True)

        # Add columns with justified headers
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            style_config = STYLES.get(str(dtype), {"style": "green", "justify": "left"})
            self.table.add_column(Text(col, justify=style_config["justify"]), key=col)

        self.table.cursor_type = "cell"
        self.table.focus()

    def _load_rows(self) -> None:
        """Load all rows into the table."""

        for row_idx, row in enumerate(self.df.rows()):
            vals, dtypes = [], []
            for val, dtype in zip(row, self.df.dtypes):
                vals.append(val)
                dtypes.append(dtype)
            formatted_row = _format_row(vals, dtypes)
            # Always add labels so they can be shown/hidden via CSS
            rid = str(row_idx + 1)
            self.table.add_row(*formatted_row, key=rid, label=rid)


if __name__ == "__main__":
    # Read dataframe from CSV file
    df = pl.read_csv(sys.argv[1])

    # Run the app
    app = DataFrameViewer(df)
    app.run()
