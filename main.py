import os
import sys
from io import StringIO

import polars as pl
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Static

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


class SaveFileScreen(ModalScreen):
    """Modal screen to save the dataframe to a CSV file."""

    CSS = """
    SaveFileScreen {
        align: center middle;
    }

    SaveFileScreen > Vertical {
        width: 60;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    SaveFileScreen #title {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    SaveFileScreen Input {
        margin: 1 0;
    }

    SaveFileScreen #button-container {
        width: 100%;
        height: 3;
        align: center middle;
    }

    SaveFileScreen Button {
        margin: 0 1;
    }
    """

    def __init__(self, filename: str = "dataframe.csv"):
        super().__init__()
        self.filename = filename

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Save DataFrame", id="title")
            self.filename_input = Input(value=self.filename, id="input")
            self.filename_input.value = self.filename
            self.filename_input.select_all()
            yield self.filename_input

            with Horizontal(id="button-container"):
                yield Button("Confirm", id="confirm", variant="success")
                yield Button("Cancel", id="cancel", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            filename = self.filename_input.value.strip()
            if filename:
                self.dismiss(filename)
            else:
                self.app.notify("Filename cannot be empty", title="Error")
        elif event.button.id == "cancel":
            self.dismiss(None)

    def on_key(self, event):
        if event.key == "enter":
            filename = self.filename_input.value.strip()
            if filename:
                self.dismiss(filename)
            else:
                self.app.notify("Filename cannot be empty", title="Error")
            event.stop()
        elif event.key == "escape":
            self.dismiss(None)


class OverwriteFileScreen(ModalScreen):
    """Modal screen to confirm file overwrite."""

    CSS = """
    OverwriteFileScreen {
        align: center middle;
    }

    OverwriteFileScreen > Vertical {
        width: 60;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    OverwriteFileScreen #message {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    OverwriteFileScreen #button-container {
        width: 100%;
        height: 3;
        margin-top: 1;
        align: center middle;
    }

    OverwriteFileScreen Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("File already exists. Overwrite?", id="message")
            with Horizontal(id="button-container"):
                yield Button("Confirm", id="confirm", variant="success")
                yield Button("Cancel", id="cancel", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.dismiss(True)
        elif event.button.id == "cancel":
            self.dismiss(False)

    def on_key(self, event) -> None:
        if event.key == "enter":
            self.dismiss(True)
            event.stop()
        elif event.key == "escape":
            self.dismiss(False)


class RowDetailScreen(ModalScreen):
    """Modal screen to display a single row's details."""

    BINDINGS = [
        ("q,escape", "app.pop_screen", "Close"),
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

    def __init__(
        self,
        columns: list[str],
        row_values: list,
        dtypes: list,
    ):
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


class FrequencyScreen(ModalScreen):
    """Modal screen to display frequency of values in a column."""

    BINDINGS = [
        ("q,escape", "app.pop_screen", "Close"),
    ]

    CSS = """
    FrequencyScreen {
        align: center middle;
    }

    FrequencyScreen > DataTable {
        width: 60;
        height: auto;
        border: solid $primary;
    }
    """

    def __init__(self, column: str, dtype: pl.DataType, df: pl.DataFrame):
        super().__init__()
        self.column = column
        self.dtype = dtype
        self.df = df

    def compose(self) -> ComposeResult:
        """Create the frequency table."""
        column_style = STYLES.get(str(self.dtype), {"style": "", "justify": ""})
        # Create frequency table
        freq_table = DataTable(zebra_stripes=True)
        freq_table.add_column(Text(self.column, justify=column_style["justify"]))
        freq_table.add_column(Text("Count", justify="right"))
        freq_table.add_column(Text("%", justify="right"))

        # Calculate frequencies using Polars
        freq_df = (
            self.df[self.column].value_counts(sort=True).sort("count", descending=True)
        )

        total_count = len(self.df)

        # Get style config for Int64 and Float64
        int_style_config = STYLES.get("Int64")
        float_style_config = STYLES.get("Float64")

        # Add rows to the frequency table
        for row in freq_df.rows():
            value, count = row
            percentage = (count / total_count) * 100

            freq_table.add_row(
                Text(
                    str(value) if value is not None else "-",
                    style=column_style["style"],
                    justify=column_style["justify"],
                ),
                Text(
                    str(count),
                    style=int_style_config["style"],
                    justify=int_style_config["justify"],
                ),
                Text(
                    f"{percentage:.2f}",
                    style=float_style_config["style"],
                    justify=float_style_config["justify"],
                ),
            )

        yield freq_table


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

    def __init__(self, df: pl.DataFrame, filename: str = ""):
        super().__init__()
        self.dataframe = df  # Original dataframe
        self.df = df  # Internal dataframe
        self.filename = filename
        self.loaded_rows = 0  # Track how many rows are currently loaded
        self.total_rows = len(df)
        self.sorted_columns = {}  # Track sort keys as dict of col_name -> descending
        self.visible_columns = list(df.columns)  # Track which columns are visible

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
        self._setup_table()
        # Hide labels by default after initial load
        self.call_later(lambda: setattr(self.table, "show_row_labels", False))

    def on_key(self, event) -> None:
        """Handle key events."""
        if event.key == "g":
            # Jump to top
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
        elif event.key == "minus":
            # Remove the current column
            self._remove_current_column()
        elif event.key == "left_square_bracket":  # '['
            # Sort by current column in ascending order
            self._sort_by_column(descending=False)
        elif event.key == "right_square_bracket":  # ']'
            # Sort by current column in descending order
            self._sort_by_column(descending=True)
        elif event.key == "r":
            # Restore original display
            self._setup_table()
            # Hide labels by default after initial load
            # self.call_later(lambda: setattr(self.table, "show_row_labels", False))
            self.log(f"{self.table.show_row_labels = }")

            self.notify("Restored original display", title="Reset")
        elif event.key == "s":
            # Save dataframe to CSV
            self._save_to_file()
        elif event.key == "f":
            # Open frequency modal for current column
            self._show_frequency()

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

        # Get the cell value from the sorted dataframe
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

    def _setup_table(self) -> None:
        """Setup the table for display."""
        # Reset to original dataframe
        self.df = self.dataframe
        self.loaded_rows = 0
        self.total_rows = len(self.dataframe)
        self.sorted_columns = {}
        self.visible_columns = list(self.dataframe.columns)

        self._setup_columns()
        self._load_rows(INITIAL_BATCH_SIZE)

    def _setup_columns(self) -> None:
        """Clear table and setup columns.

        Args:
            visible_columns: List of column names to display. If None, all columns are visible.
        """

        self.table.clear(columns=True)
        self.loaded_rows = 0

        # Add columns with justified headers (only visible columns)
        for col, dtype in zip(self.df.columns, self.df.dtypes):
            if col not in self.visible_columns:
                continue
            style_config = STYLES.get(str(dtype), {"style": "green", "justify": "left"})
            self.table.add_column(Text(col, justify=style_config["justify"]), key=col)

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

        for row_idx, row in enumerate(df_slice.rows(), start_idx):
            vals, dtypes = [], []
            for val, col, dtype in zip(row, self.df.columns, self.df.dtypes):
                if col not in self.visible_columns:
                    continue
                vals.append(val)
                dtypes.append(dtype)
            formatted_row = _format_row(vals, dtypes)
            # Always add labels so they can be shown/hidden via CSS
            self.table.add_row(*formatted_row, label=str(row_idx + 1))

        self.loaded_rows = end_idx

        if count != INITIAL_BATCH_SIZE:
            self.notify(
                f"Loaded {self.loaded_rows}/{self.total_rows} rows", title="Load"
            )

    def _view_row_detail(self) -> None:
        """Open a modal screen to view the selected row's details."""
        row_idx = self.table.cursor_row
        if row_idx >= self.total_rows:
            return

        columns, vals, dtypes = [], [], []
        # Get the row data from the dataframe
        for val, col, dtype in zip(
            self.df.row(row_idx), self.df.columns, self.df.dtypes
        ):
            if col not in self.visible_columns:
                continue
            columns.append(col)
            vals.append(val)
            dtypes.append(dtype)

        # Push the modal screen
        self.push_screen(RowDetailScreen(columns, vals, dtypes))

    def _remove_current_column(self) -> None:
        """Remove the currently selected column from the table."""
        col_idx = self.table.cursor_column
        if col_idx >= len(self.visible_columns):
            return

        # Get the column name to remove
        col_to_remove = self.visible_columns[col_idx]

        # Remove the column from the table display using the column name as key
        self.table.remove_column(col_to_remove)

        # Remove from visible columns
        self.visible_columns.remove(col_to_remove)

        # Remove from sorted columns if present
        if col_to_remove in self.sorted_columns:
            del self.sorted_columns[col_to_remove]

        # Remove from dataframe view
        self.df = self.df.drop(col_to_remove)

        self.notify(
            f"Removed column [on $primary]{col_to_remove}[/] from display",
            title="Column",
        )

    def _sort_by_column(self, descending: bool = False) -> None:
        """Sort the dataframe by the currently selected column.

        Supports multi-column sorting:
        - First press on a column: sort by that column only
        - Subsequent presses on other columns: add to sort order

        Args:
            descending: If True, sort in descending order. If False, ascending order.
        """
        col_idx = self.table.cursor_column
        if col_idx >= len(self.visible_columns):
            return

        col_to_sort = self.visible_columns[col_idx]

        # Check if this column is already in the sort keys
        old_desc = self.sorted_columns.get(col_to_sort)
        if old_desc is not None:
            del self.sorted_columns[col_to_sort]

            if old_desc == descending:
                # Same direction - remove this column from sort
                self.notify(
                    f"Already sorted. Removed [on $primary]{col_to_sort}[/] from the sorted list",
                    title="Sort",
                )
                return
            else:
                # Toggle direction
                self.sorted_columns[col_to_sort] = descending
        else:
            # Add new column to sort
            self.sorted_columns[col_to_sort] = descending

        # If no sort keys, reset to original order
        if not self.sorted_columns:
            self.df = self.dataframe
        else:
            # Apply multi-column sort
            sort_cols = list(self.sorted_columns.keys())
            descending_flags = list(self.sorted_columns.values())
            self.df = self.dataframe.sort(
                sort_cols, descending=descending_flags, nulls_last=True
            )

        self.total_rows = len(self.df)

        # Recreate the table for display
        self._setup_columns()
        self._load_rows(INITIAL_BATCH_SIZE)

        # Restore cursor position on the sorted column
        self.table.move_cursor(column=col_idx, row=0)

        sort_by = ", ".join(
            f"[on $primary]{col}[/] ({'desc' if desc else 'asc'})"
            for col, desc in self.sorted_columns.items()
        )
        self.notify(f"Sorted by {sort_by}", title="Sort")

    def _save_to_file(self) -> None:
        """Open save file dialog."""
        self.push_screen(
            SaveFileScreen(self.filename or "dataframe.csv"),
            callback=self._on_save_file_screen,
        )

    def _on_save_file_screen(self, filename: str | None) -> None:
        """Handle result from SaveFileScreen."""
        if filename is None:
            return

        # Check if file exists
        if os.path.exists(filename):
            self.push_screen(
                OverwriteFileScreen(filename), callback=self._on_overwrite_screen
            )
            self._pending_filename = filename
        else:
            self._do_save(filename)

    def _on_overwrite_screen(self, should_overwrite: bool) -> None:
        """Handle result from OverwriteFileScreen."""
        if should_overwrite:
            self._do_save(self._pending_filename)
        else:
            # Go back to SaveFileScreen to allow user to enter a different name
            self.push_screen(
                SaveFileScreen(self._pending_filename),
                callback=self._on_save_file_screen,
            )

    def _do_save(self, filename: str) -> None:
        """Actually save the dataframe to a file."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in (".tsv", ".tab"):
            separator = "\t"
        else:
            separator = ","
        try:
            self.df.write_csv(filename, separator=separator)
            self.filename = filename
            self.notify(f"Saved to {filename}", title="Save")
        except Exception as e:
            self.notify(f"Failed to save: {str(e)}", title="Error")

    def _show_frequency(self) -> None:
        """Show frequency distribution for the current column."""
        col_idx = self.table.cursor_column
        if col_idx >= len(self.visible_columns):
            return

        col_name = self.visible_columns[col_idx]
        col_dtype = self.df.schema[col_name]

        # Push the frequency modal screen
        self.push_screen(FrequencyScreen(col_name, col_dtype, self.df))


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
    filename = ""

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
    app = DataFrameViewer(df, filename)
    app.run()
