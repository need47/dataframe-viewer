"""Modal screens for displaying data in tables (row details and frequency)."""

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from textual.widgets.data_table import ColumnKey

if TYPE_CHECKING:
    from .data_frame_table import DataFrameTable
    from .keybindings import KeyBindingRegistry

from functools import partial

import polars as pl
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.events import Key
from textual.renderables.bar import Bar
from textual.screen import ModalScreen
from textual.widgets import DataTable

from .commands import Scope
from .common import (
    BAR_COLUMN_WIDTH,
    COLUMN_WIDTH_CAP,
    CURSOR_TYPES,
    HIGHLIGHT_COLOR,
    NULL,
    NULL_DISPLAY,
    RID,
    DtypeConfig,
    format_float,
    get_next_item,
)
from .file_picker_screen import SaveFileScreen
from .text_screen import TextScreen
from .yes_no_screen import JoinTableScreen


class TableModalScreen(ModalScreen):
    """Base class for modal screens displaying data in a DataTable.

    Provides common functionality for screens that show tabular data with
    keyboard shortcuts and styling.
    """

    DEFAULT_CSS = """
        TableModalScreen {
            align: center middle;
        }

        TableModalScreen > DataTable {
            width: auto;
            max-width: 100%;
            height: auto;
            min-width: 17; /* for LoadIndicator */
            min-height: 3; /* for LoadIndicator */
            border: solid $primary;
            overflow: auto;
        }
    """

    def __init__(self, df: pl.DataFrame | None = None) -> None:
        """Initialize the table screen.

        Sets up the base modal screen and stores the DataFrame for display.

        Args:
            df: The DataFrame to display in this screen.
        """
        super().__init__()
        self.df = df  # DataFrame for this screen, to be set by subclasses
        self.thousand_separator = False  # Whether to use thousand separators in numbers
        self.selected_rows: set[int] = set()  # Track selected row indices for potential multi-row actions
        self.sorted_columns: dict[str, bool] = {}  # Track sorted columns and their sort order
        self.sort_ignore_last = False  # Whether to ignore the last column (e.g., Total) when sorting
        self.col_style = None
        self.col_justify = None
        self.filename: str | None = None

        self.registry: "KeyBindingRegistry" = self.app.key_registry
        self.leader = ""  # For command leader key sequences, if needed
        self.leader_timer = None

    def compose(self) -> ComposeResult:
        """Compose the table screen widget structure.

        Creates and yields a DataTable widget for displaying tabular data.
        Subclasses should override to customize table configuration.

        Yields:
            DataTable: The table widget for this screen.
        """
        self.table = DataTable(zebra_stripes=True)
        yield self.table

    def on_mount(self) -> None:
        self.app.notify("Ready")

    def on_key(self, event: Key) -> None:
        """Handle key press events in the table screen.

        Dispatches common modal-table commands and manages modal leader mode.

        Args:
            event: The key event object.
        """
        # Try dispatching table-screen-scoped commands
        if self.registry.dispatch(event.key, self.leader, Scope.TABLE_SCREEN, self):
            event.stop()
            event.prevent_default()
            self.reset_leader()
            return

        # Already in leader mode, no matching command found
        if self.leader:
            event.stop()
            event.prevent_default()
            self.notify(
                f"Command not found for [$warning]{self.leader}[/][$accent]{event.key}[/] key binding",
                title="Key Binding",
                severity="warning",
            )
            self.reset_leader()
            return

        # Enter leader mode on `g` or `z` key
        if event.key in ("g", "z"):
            event.stop()
            event.prevent_default()
            self.leader = event.key
            self.notify(
                f"Leader mode activated with [$success]{event.key}[/], waiting for next key in 3 seconds",
                title="Leader Mode",
            )
            self.leader_timer = self.set_timer(3, callback=lambda: self.reset_leader("Leader mode timed out"))
            return

    def reset_leader(self, message: str = "") -> None:
        """Cancel modal leader mode and reset the timeout timer."""
        self.leader = ""
        if message:
            self.notify(message, title="Leader Mode")

        if self.leader_timer:
            self.leader_timer.stop()
            self.leader_timer = None

    def cmd_table_close(self) -> None:
        """Close the modal table."""
        self.app.pop_screen()

    def cmd_table_toggle_thousand_separator(self) -> None:
        """Toggle thousand separators in the modal table."""
        self.thousand_separator = not self.thousand_separator
        self.build_table()

    def cmd_table_scroll_top(self) -> None:
        """Scroll the modal table to the top."""
        self.table.action_scroll_top()
        self.notify("Scrolled to [$success]top[/]", title="Scroll")

    def cmd_table_scroll_bottom(self) -> None:
        """Scroll the modal table to the bottom."""
        self.table.action_scroll_bottom()
        self.notify("Scrolled to [$success]bottom[/]", title="Scroll")

    def cmd_table_sort_ascending(self) -> None:
        """Sort the modal table by the current column ascending."""
        self.sort_by_column(descending=False)

    def cmd_table_sort_descending(self) -> None:
        """Sort the modal table by the current column descending."""
        self.sort_by_column(descending=True)

    def cmd_table_cycle_cursor_type(self) -> None:
        """Cycle the modal table cursor type."""
        next_type = get_next_item(CURSOR_TYPES, self.table.cursor_type)
        self.table.cursor_type = next_type

    def cmd_table_open_as_tab(self) -> None:
        """Open the modal table as a new tab."""
        self.open_as_tab()

    def cmd_table_save(self) -> None:
        """Save the modal table to a file."""
        self.save_table()

    def open_as_tab(self) -> None:
        """Open the current modal DataFrame as a new tab."""
        if self.df is None or self.df.is_empty():
            self.notify("No data to open as tab", severity="warning")
            return

        tabname, filename = "modal", "modal.csv"
        cols = [c for c in self.df.columns if c != RID]
        df_out = self.df.select(cols).filter((pl.nth(0) != pl.lit(RID)) | pl.nth(0).is_null())
        self.app.add_tab(df_out, filename=filename, tabname=tabname)
        self.app.pop_screen()
        self.notify(f"Opened as new tab: [$success]{tabname}[/]", title="New Tab")

    def build_table(self) -> None:
        """Build the table content.

        Subclasses should implement this method to populate the DataTable
        with appropriate columns and rows based on the specific screen's purpose.
        """
        raise NotImplementedError("Subclasses must implement build_table method.")

    def save_table(self) -> None:
        """Save the table to file."""
        filename = self.filename or "untitled.csv"

        self.app.push_screen(
            SaveFileScreen(filename=filename),
            callback=partial(self.app.save_to_file, all_tabs=False, use_df=self.df),
        )

    def df2table(
        self,
        col_style: str | dict[str, str | list[str]] | None = None,
        col_justify: str | dict[str, str] | None = None,
    ) -> None:
        """Convert a Polars DataFrame to a DataTable for display.


        Args:
            col_style: Optional style(s) to apply to all cells. Can be a single string, or a dict mapping column names to styles (which can be strings or lists for per-row styling).
            col_justify: Optional justification(s) for all cells. Can be a single string, or a dict mapping column names to justification.
        """
        if self.df is None:
            return

        if col_style is None:
            col_style = self.col_style
        else:
            self.col_style = col_style

        if col_justify is None:
            col_justify = self.col_justify
        else:
            self.col_justify = col_justify

        # Store the current cursor coordinate to restore it after rebuilding the table
        row_idx, col_idx = self.table.cursor_coordinate

        self.table.clear(columns=True)

        # Add columns with proper justification based on data types
        for cidx, (col, dtype) in enumerate(zip(self.df.columns, self.df.dtypes)):
            if col == RID:
                continue

            dc = DtypeConfig(dtype)
            justify = col_justify.get(col) if isinstance(col_justify, dict) else col_justify
            justify = dc.justify if justify is None else justify

            for c in self.sorted_columns:
                if c == col:
                    # Add sort indicator to column header
                    descending = self.sorted_columns[col]
                    sort_indicator = " ▼" if descending else " ▲"
                    cell_value = col + sort_indicator
                    break
            else:  # No break occurred, so column is not sorted
                cell_value = col

            self.table.add_column(Text(cell_value, justify=justify), key=col)

        # Add rows with proper formatting based on data types
        for ridx, row in enumerate(self.df.iter_rows()):
            # Skip the row containing the RID value
            if row[0] == RID or (isinstance(row[0], Text) and row[0].plain == RID):
                continue

            is_selected = ridx in self.selected_rows

            formatted_row = []
            for cidx, c in enumerate(row):
                col = self.df.columns[cidx]
                if col == RID:
                    continue
                dc = DtypeConfig(self.df.dtypes[cidx])
                style = col_style.get(col) if isinstance(col_style, dict) else col_style
                justify = col_justify.get(col) if isinstance(col_justify, dict) else col_justify

                formatted_row.append(
                    dc.format(
                        NULL_DISPLAY if c is None else c,
                        style=HIGHLIGHT_COLOR if is_selected else style[ridx] if isinstance(style, list) else style,
                        justify=justify,
                        thousand_separator=self.thousand_separator,
                    )
                )

            self.table.add_row(*formatted_row, key=str(ridx), label=str(ridx + 1))

        # Restore the old cursor coordinate
        self.table.move_cursor(row=row_idx, column=col_idx)

    def sort_by_column_key(self, col_key: ColumnKey, descending: bool) -> None:
        """Sort the table by the specified column."""
        if self.sort_ignore_last and self.table.row_count > 1:
            # Detach the last row, sort the rest, then re-append it
            last_row = self.table.ordered_rows[-1]
            last_cells = self.table.get_row(last_row.key)
            self.table.remove_row(last_row.key)
            self.table.sort(col_key, key=lambda c: c.plain if isinstance(c, Text) else c, reverse=descending)
            self.table.add_row(*last_cells, key=last_row.key.value, label=last_row.label)
        else:
            self.table.sort(col_key, key=lambda c: c.plain if isinstance(c, Text) else c, reverse=descending)

    def sort_by_column(self, descending: bool) -> None:
        """Sort the table by the current column.

        Args:
            descending: Whether to sort in descending order. Defaults to False (ascending).
        """
        # Get the current column index and name
        ridx, cidx = self.table.cursor_coordinate
        col_key = self.table.coordinate_to_cell_key((ridx, cidx)).column_key
        col_name = col_key.value

        # Already sorted by this column in the same order, do nothing
        if self.sorted_columns.get(col_name) == descending:
            return

        # Update sorted columns tracking and sort the dataframe
        self.sorted_columns.clear()
        self.sorted_columns[col_name] = descending

        # If no DataFrame is available (e.g., not yet populated), use built-in sort to sort the table directly
        if self.df is None:
            self.sort_by_column_key(col_key, descending)
            return

        if self.df.is_empty():
            return

        try:
            self.df = self.df.sort(col_name, descending=descending, nulls_last=True)
        except Exception as e:
            self.log(f"Error sorting by column '{col_name}': {e}")

            # Fallback to built-in sort if dataframe sorting fails (e.g. due to unsupported data types)
            self.sort_by_column_key(col_key, descending)
            return

        # Rebuild the table
        self.df2table()

        # Move cursor back to the same column and approximate row position after sorting
        self.table.move_cursor(row=ridx, column=cidx)

    def toggele_row_selection(self) -> None:
        """Toggle selection of the currently focused row."""
        ridx, _ = self.table.cursor_coordinate
        if ridx in self.selected_rows:
            self.selected_rows.remove(ridx)
        else:
            self.selected_rows.add(ridx)

    def _on_calc_ready(self) -> None:
        self.build_table()
        self.table.focus()
        self.table.loading = False


class TableScreen(TableModalScreen):
    """Base class for modal screens displaying data in a DataTable.

    Provides common functionality for screens that show tabular data with
    keyboard shortcuts and styling.
    """

    DEFAULT_CSS = TableModalScreen.DEFAULT_CSS.replace("TableModalScreen", "TableScreen")

    def __init__(self, dftable: "DataFrameTable") -> None:
        """Initialize the table screen.

        Sets up the base modal screen with reference to the main DataFrameTable widget
        and stores the DataFrame for display.

        Args:
            dftable: Reference to the parent DataFrameTable widget, if applicable.
        """
        super().__init__()
        self.dftable = dftable

    def sort_by_column(self, descending: bool) -> None:
        # Override to disable sorting in TableScreen, but subclasses can still implement their own sorting if desired.
        pass

    def filter_or_collect_selected_value(self, col_name: str, values: Any | list[Any], action: str = "filter") -> None:
        """Apply filter or collect action by the selected value.

        Filter or collect rows in the main table based on a selected value from
        this table (typically frequency or row detail). Updates the main table's display
        and notifies the user of the action.

        Args:
            col_name: Column name.
            values: Selected value(s) to filter/collect by.
            action: Either "filter" to filter rows, or "collect" to collect rows. Defaults to "filter".
        """
        # Create expression for NULL values
        if values is None or values == NULL:
            expr = pl.col(col_name).is_null()
            value_display = f"[$success]{NULL_DISPLAY}[/]"
        # Create expression for the selected value
        else:
            if isinstance(values, list):
                expr = pl.col(col_name).is_in(values)
                value_display = f"[$success]{','.join(values)}[/]"
            else:
                expr = pl.col(col_name) == values
                value_display = f"[$success]{values}[/]"

        df_filtered = self.dftable.df.lazy().filter(expr).collect()

        ok_rids = set(df_filtered[RID].to_list())
        if not ok_rids:
            self.notify(
                f"No matches found for [$warning]{col_name}[/] == {value_display}",
                title="No Matches",
                severity="warning",
            )
            return

        cidx = self.dftable.get_cidx(col_name)
        if cidx is None:
            self.notify(f"Column [$warning]{col_name}[/] not found", title="Filter Value", severity="warning")
            return

        # Dismiss modal screen(s) to return to main table first: filtering/collecting
        # pushes a transient BusyScreen, so popping afterward would remove that instead.
        while len(self.app._screen_stack) > 1:
            self.app.pop_screen()

        # Action collect
        if action == "collect":
            self.dftable.cmd_collect_rows(col_name=col_name, term=values)

        # Action filter
        else:
            self.dftable.cmd_filter_rows(
                {
                    "term": expr,
                    "col_name": col_name,
                    "match_nocase": False,
                    "match_whole": True,
                    "match_literal": True,
                    "match_reverse": False,
                }
            )

        self.dftable.move_cursor(column=cidx)

    def show_frequency(self, *col_names: str) -> None:
        """Show frequency for the selected columns.

        Args:
            *col_names: Column names.
        """
        if not col_names:
            return

        # Show frequency screen
        self.dftable.cmd_show_frequency(*col_names)

    def show_statistics(self, col_name: str | None = None) -> None:
        """Show statistics for the selected column.

        Args:
            col_name: Column name.
        """
        if col_name is None:
            return

        # Show statistics screen
        self.dftable.cmd_show_statistics(col_name)


class RowDetailScreen(TableScreen):
    """Modal screen to display a single row's details."""

    def __init__(self, dftable: "DataFrameTable", ridx: int) -> None:
        super().__init__(dftable)
        self.ridx = ridx

    def on_mount(self) -> None:
        """Initialize the row detail screen.

        Populates the table with column names and values from the selected row
        of the main DataFrame. Sets the table cursor type to "row".
        """
        super().on_mount()
        self.build_table()

    def on_key(self, event: Key) -> None:
        """Handle key press events on the row detail screen.

        Dispatches row-detail commands through the key binding registry, then
        falls back to the common modal table shortcuts.

        Args:
            event: The key event object.
        """
        if self.registry.dispatch(event.key, self.leader, Scope.ROW_DETAIL_SCREEN, self):
            event.stop()
            event.prevent_default()
            self.reset_leader()
            return

        super().on_key(event)

    def cmd_row_detail_filter_value(self) -> None:
        """Filter the main table by the selected row-detail value."""
        col_name, value = self.get_col_name_value()
        self.filter_or_collect_selected_value(col_name, value, action="filter")

    def cmd_row_detail_collect_value(self) -> None:
        """Collect the selected row-detail value in the main table to a new tab."""
        col_name, value = self.get_col_name_value()
        self.filter_or_collect_selected_value(col_name, value, action="collect")

    def cmd_row_detail_previous_row(self) -> None:
        """Show the previous dataframe row in the row-detail screen."""
        ridx = self.ridx - 1
        if ridx >= 0:
            self.ridx = ridx
            self.dftable.move_cursor_to(self.ridx)
            self.build_table()

    def cmd_row_detail_next_row(self) -> None:
        """Show the next dataframe row in the row-detail screen."""
        ridx = self.ridx + 1
        if ridx < len(self.dftable.df):
            self.ridx = ridx
            self.dftable.move_cursor_to(self.ridx)
            self.build_table()

    def cmd_row_detail_show_frequency(self) -> None:
        """Show frequency for the selected row-detail value."""
        col_name, _ = self.get_col_name_value()
        self.show_frequency(col_name)

    def cmd_row_detail_show_statistics(self) -> None:
        """Show statistics for the selected row-detail value."""
        col_name, _ = self.get_col_name_value()
        self.show_statistics(col_name)

    def cmd_row_detail_drill_down(self) -> None:
        """Show cell details for the selected row-detail value."""
        ridx = self.ridx
        cidx = self.table.cursor_row
        col_name = self.dftable.df.columns[cidx]
        dtype = self.dftable.df.dtypes[cidx]
        cell_value = self.dftable.df.item(ridx, cidx)

        if dtype == pl.String:
            # String contains the delimiter '|' (indicating a potential list of values)
            if "|" in cell_value:
                self.app.push_screen(CellDetailScreen(col_name, dtype, cell_value))
            # Show long string in a text screen for better readability
            elif len(cell_value) > COLUMN_WIDTH_CAP:
                self.app.push_screen(TextScreen(cell_value))

        # Show cell detail screen if the value is a non-empty list
        elif dtype == pl.List and not cell_value.is_empty():
            if len(cell_value) == 1 and isinstance(cell_value[0], str) and len(cell_value[0]) > COLUMN_WIDTH_CAP:
                self.app.push_screen(TextScreen(cell_value[0]))
            else:
                self.app.push_screen(CellDetailScreen(col_name, dtype, cell_value))

        # or a non-empty dict (struct)
        elif dtype == pl.Struct and cell_value:
            self.app.push_screen(CellDetailScreen(col_name, dtype, cell_value))

    def build_table(self) -> None:
        """Build the row detail table."""
        self.df = pl.DataFrame(
            {
                "Column": self.dftable.df.columns,
                "Value": [NULL_DISPLAY if c is None else str(c) for c in self.dftable.df.row(self.ridx)],
            }
        )

        col2style = defaultdict(list)
        col2style["Column"] = ""  # No specific style for the "Column" header
        for dtype in self.dftable.df.dtypes:
            col2style["Value"].append(DtypeConfig(dtype).style)

        self.df2table(col_style=col2style)

        self.table.cursor_type = "row"

    def get_col_name_value(self) -> tuple[str, Any]:
        """Get the current column info."""
        cidx = self.table.cursor_row
        col_name = self.dftable.df.columns[cidx]
        col_value = self.dftable.df.item(self.ridx, cidx)
        return col_name, col_value


class StatisticsScreen(TableScreen):
    """Modal screen to display statistics for a column or entire dataframe."""

    def __init__(self, dftable: "DataFrameTable", col_name: str = "") -> None:
        super().__init__(dftable)
        self.col_name = col_name

    def on_mount(self) -> None:
        """Create the statistics table."""
        super().on_mount()
        self.table.loading = True
        self.calculate_statistics()

    @work(thread=True)
    def calculate_statistics(self) -> None:
        """Calculate statistics."""
        self.build_df()
        self.app.call_from_thread(self._on_calc_ready)

    def build_table(self) -> None:
        """Build the statistics table."""
        self.table.clear(columns=True)

        # Add columns
        for col_name, col_dtype in zip(self.df.columns, self.df.dtypes):
            if col_name == "statistic":  # no styling
                self.table.add_column("Statistic", key=col_name)
                continue

            dc = DtypeConfig(col_dtype)
            self.table.add_column(Text(col_name, justify=dc.justify), key=col_name)

        # Add rows
        for ridx, row in enumerate(self.df.iter_rows()):
            formatted_row = []
            is_selected = ridx in self.selected_rows

            # Format remaining values with appropriate styling
            for idx, stat_value in enumerate(row):
                # First element is the statistic label, no styling needed
                if idx == 0:
                    formatted_row.append(Text(stat_value, style=HIGHLIGHT_COLOR if is_selected else ""))
                    continue

                col_dtype = self.df.dtypes[idx]
                dc = DtypeConfig(col_dtype)

                if ridx < 4 and col_dtype == pl.String and self.thousand_separator:
                    stat_value = f"{int(stat_value):,}"

                formatted_row.append(
                    dc.format(
                        stat_value,
                        style=HIGHLIGHT_COLOR if is_selected else None,
                        justify=dc.justify,
                        thousand_separator=self.thousand_separator,
                    )
                )

            self.table.add_row(*formatted_row, key=str(ridx), label=str(ridx + 1))

        # Set cursor type based on whether this is dataframe stats (column cursor) or column stats (row cursor)
        self.table.cursor_type = "column" if not self.col_name else "row"

    def build_df(self) -> None:
        """Get the dataframe to use for statistics."""
        # all columns
        if not self.col_name:
            lf = self.dftable.df.lazy().select(pl.exclude(RID))

            # Apply only to non-hidden columns
            if self.dftable.hidden_columns:
                lf = lf.select(pl.exclude(self.dftable.hidden_columns))

            source_schema = lf.collect_schema()

            # Get dataframe statistics
            stats_df = lf.describe()

            # total
            df_n_total = lf.select(pl.all().len()).collect()
            df_n_total.insert_column(0, pl.Series("statistic", ["n_total"]))
            df_n_total = df_n_total.cast(stats_df.schema)

            # unique count for each column
            df_n_unique = lf.select(pl.all().n_unique()).collect()
            df_n_unique.insert_column(0, pl.Series("statistic", ["n_unique"]))
            df_n_unique = df_n_unique.cast(stats_df.schema)

            # sum
            sum_exprs: list[pl.Expr] = [pl.lit("sum").alias("statistic")]
            for col_name, dtype in source_schema.items():
                dc = DtypeConfig(dtype)
                if dc.gtype in ("integer", "float"):
                    sum_exprs.append(pl.col(col_name).sum().alias(col_name))
                else:
                    sum_exprs.append(pl.lit(None).alias(col_name))

            df_sum = lf.select(sum_exprs).collect().cast(stats_df.schema)

            # min_length and max_length
            min_length_exprs: list[pl.Expr] = [pl.lit("min_length").alias("statistic")]
            max_length_exprs: list[pl.Expr] = [pl.lit("max_length").alias("statistic")]
            for col_name, dtype in source_schema.items():
                if dtype == pl.String:
                    min_length_exprs.append(pl.col(col_name).str.len_chars().min().alias(col_name))
                    max_length_exprs.append(pl.col(col_name).str.len_chars().max().alias(col_name))
                elif dtype == pl.List:
                    min_length_exprs.append(pl.col(col_name).list.len().min().alias(col_name))
                    max_length_exprs.append(pl.col(col_name).list.len().max().alias(col_name))
                else:
                    try:
                        min_length_exprs.append(pl.col(col_name).cast(pl.String).str.len_chars().min().alias(col_name))
                        max_length_exprs.append(pl.col(col_name).cast(pl.String).str.len_chars().max().alias(col_name))
                    except Exception:
                        min_length_exprs.append(pl.lit(None).alias(col_name))
                        max_length_exprs.append(pl.lit(None).alias(col_name))

            df_min_length = lf.select(min_length_exprs).collect().cast(stats_df.schema)
            df_max_length = lf.select(max_length_exprs).collect().cast(stats_df.schema)

            # fill rate
            fill_exprs: list[pl.Expr] = [pl.lit("fill%").alias("statistic")]
            for col_name, dtype in source_schema.items():
                fill_expr = pl.col(col_name).count().truediv(pl.len()) * 100
                dc = DtypeConfig(dtype)
                if dc.gtype in ("integer", "float"):
                    fill_exprs.append(fill_expr.alias(col_name))
                else:
                    fill_exprs.append(fill_expr.round(1).cast(pl.String).alias(col_name))

            df_fill = lf.select(fill_exprs).collect().cast(stats_df.schema)

            # vstack
            self.df = pl.concat(
                [
                    df_n_unique,
                    df_n_total,
                    stats_df,
                    df_sum,
                    df_min_length,
                    df_max_length,
                    df_fill,
                ]
            )

        # single column
        else:
            col_name = self.col_name
            dtype = self.dftable.df.schema[col_name]
            this_col = self.dftable.df[col_name]
            lf = self.dftable.df.lazy()

            # Get column statistics
            stats_df = lf.select(pl.col(col_name)).describe()
            if len(stats_df) == 0:
                return

            # unique count
            n_unique = this_col.n_unique()
            df_n_unique = pl.DataFrame({"statistic": ["n_unique"], col_name: n_unique}, schema=stats_df.schema)

            # total count
            n_total = len(this_col)
            df_n_total = pl.DataFrame({"statistic": ["n_total"], col_name: n_total}, schema=stats_df.schema)

            # sum
            dc = DtypeConfig(dtype)
            if dc.gtype in ("integer", "float"):
                sum_value = this_col.sum()
                df_sum = pl.DataFrame({"statistic": ["sum"], col_name: sum_value}, schema=stats_df.schema)
            else:
                df_sum = pl.DataFrame({"statistic": ["sum"], col_name: None}, schema=stats_df.schema)

            # min_length and max_length
            if dtype == pl.String:
                min_length = this_col.str.len_chars().min()
                max_length = this_col.str.len_chars().max()
            elif dtype == pl.List:
                min_length = this_col.list.len().min()
                max_length = this_col.list.len().max()
            else:
                try:
                    min_length = this_col.cast(pl.String).str.len_chars().min()
                    max_length = this_col.cast(pl.String).str.len_chars().max()
                except Exception:
                    min_length = None
                    max_length = None

            df_min_length = pl.DataFrame({"statistic": ["min_length"], col_name: min_length}, schema=stats_df.schema)
            df_max_length = pl.DataFrame({"statistic": ["max_length"], col_name: max_length}, schema=stats_df.schema)

            # fill rate
            fill_rate = this_col.count() / n_total * 100
            if dc.gtype in ("integer", "float"):
                df_fill = pl.DataFrame({"statistic": ["fill%"], col_name: fill_rate}, schema=stats_df.schema)
            else:
                df_fill = pl.DataFrame(
                    {"statistic": ["fill%"], col_name: str(round(fill_rate, 1))}, schema=stats_df.schema
                )

            # total first, then n_unique, then describe stats
            self.df = pl.concat(
                [
                    df_n_unique,
                    df_n_total,
                    stats_df,
                    df_sum,
                    df_min_length,
                    df_max_length,
                    df_fill,
                ]
            )

        # Rename the first column to "Statistic" for better display
        self.df = self.df.rename({"statistic": "Statistic"})

        # No specific styling for the "Statistics" header
        self.col_style = {"Statistic": ""}


class FrequencyScreen(TableScreen):
    """Modal screen to display frequency of values in a column."""

    def __init__(self, dftable: "DataFrameTable", *col_names: str) -> None:
        super().__init__(dftable)
        self.col_names = list(col_names)
        self.is_multi_column = len(self.col_names) > 1
        self.sorted_columns["Count"] = True  # Count sort by default
        self.total_count = len(dftable.df)
        self.columns = [(col_name, col_name) for col_name in self.col_names] + [
            ("Count", "Count"),
            ("%", "%"),
            ("Histogram", "Histogram"),
        ]

    def on_mount(self) -> None:
        """Start frequency calculation."""
        super().on_mount()
        self.table.loading = True
        self._calculate_frequency()

    @work(thread=True)
    def _calculate_frequency(self) -> None:
        """Calculate frequency."""
        if self.is_multi_column:
            self.df = (
                self.dftable.df.lazy()
                .group_by(self.col_names, maintain_order=True)
                .len(name="count")
                .sort("count", descending=True, nulls_last=True)
                .collect()
            )
        else:
            col_name = self.col_names[0]
            self.df = self.dftable.df.lazy().select(pl.col(col_name).value_counts(sort=True)).unnest(col_name).collect()

        # Add percentage column
        self.df = self.df.with_columns((pl.col("count") / self.total_count * 100).round(3).alias("%"))

        self.app.call_from_thread(self._on_calc_ready)

    def on_key(self, event: Key) -> None:
        """Handle key press events on the frequency screen."""
        if self.registry.dispatch(event.key, self.leader, Scope.FREQUENCY_SCREEN, self):
            event.stop()
            event.prevent_default()
            self.reset_leader()
            return

        super().on_key(event)

    def cmd_frequency_filter_value(self) -> None:
        """Filter the main table by selected frequency value(s)."""
        self.filter_or_collect_selected_values(action="filter")

    def cmd_frequency_collect_value(self) -> None:
        """Collect selected frequency value(s) to a new tab."""
        self.filter_or_collect_selected_values(action="collect")

    def cmd_frequency_toggle_selection(self) -> None:
        """Select or deselect the current frequency row."""
        self.toggele_row_selection()
        self.build_table()

    def build_table(self) -> None:
        """Build the frequency table."""
        # Save cursor position
        row_idx, col_idx = self.table.cursor_coordinate

        self.table.clear(columns=True)

        # Create frequency table
        dcs = {col: DtypeConfig(self.dftable.df.schema[col]) for col in self.col_names}

        for display_name, col_name in self.columns:
            # Check if this column is sorted and add indicator
            if col_name in self.sorted_columns:
                descending = self.sorted_columns[col_name]
                sort_indicator = " ▼" if descending else " ▲"
                header_text = display_name + sort_indicator
            else:
                header_text = display_name

            justify = dcs[col_name].justify if col_name in dcs else "right"
            self.table.add_column(Text(header_text, justify=justify), key=col_name)

        # Get style config for Int64 and Float64
        dc_int = DtypeConfig(pl.Int64)
        dc_float = DtypeConfig(pl.Float64)
        bar_width = BAR_COLUMN_WIDTH

        # Add rows to the frequency table
        for ridx, row in enumerate(self.df.iter_rows()):
            values = row[: len(self.col_names)]
            count = row[-2]
            percentage = row[-1]

            is_selected = ridx in self.selected_rows
            style = HIGHLIGHT_COLOR if is_selected else None

            value_cells = [
                dcs[col].format(value, style=style) for col, value in zip(self.col_names, values, strict=True)
            ]

            self.table.add_row(
                *value_cells,
                dc_int.format(count, style=style, thousand_separator=self.thousand_separator),
                dc_float.format(percentage, style=style, thousand_separator=self.thousand_separator),
                Bar(
                    highlight_range=(0.0, percentage / 100 * bar_width),
                    width=bar_width,
                ),
                key=str(ridx),
                label=str(ridx + 1),
            )

        # Add a total row
        total_cells = [Text("", style="bold", justify=dcs[col].justify) for col in self.col_names]
        total_cells[0] = Text("Total", style="bold", justify=dcs[self.col_names[0]].justify)

        self.table.add_row(
            *total_cells,
            Text(
                f"{self.total_count:,}" if self.thousand_separator else str(self.total_count),
                style="bold",
                justify="right",
            ),
            Text(
                format_float(100.0, self.thousand_separator),
                style="bold",
                justify="right",
            ),
            Bar(
                highlight_range=(0.0, bar_width),
                width=bar_width,
            ),
            key="total",
        )

        # Restore cursor position
        self.table.move_cursor(row=row_idx, column=col_idx)

    def sort_by_column(self, descending: bool) -> None:
        """Sort the dataframe by the selected column and refresh the main table."""
        row_idx, col_idx = self.table.cursor_coordinate
        col_sort = self.columns[col_idx][1]

        if self.sorted_columns.get(col_sort) == descending:
            return

        self.sorted_columns.clear()
        self.sorted_columns[col_sort] = descending

        # Count/%/Histogram sort by Count; value columns sort by their own column.
        if col_sort in {"Count", "%", "Histogram"}:
            col_name = "Count"
        else:
            col_name = col_sort

        self.df = self.df.sort(col_name, descending=descending, nulls_last=True)

        # Rebuild the frequency table
        self.build_table()

        self.table.move_cursor(row=row_idx, column=col_idx)

    def get_values(self) -> list[Any] | list[dict[str, Any]] | None:
        if self.is_multi_column:
            values: list[dict[str, Any]] = []
            if self.selected_rows:
                row_indices = self.selected_rows
            else:
                ridx = self.table.cursor_row
                if ridx >= len(self.df):
                    return None
                row_indices = {ridx}

            for ridx in row_indices:
                if ridx >= len(self.df):
                    continue  # Skip the last `Total` row
                values.append({col: self.df[col][ridx] for col in self.col_names})

            return values or None

        col_name = self.col_names[0]
        values: list[Any] = []
        if self.selected_rows:
            for ridx in self.selected_rows:
                if ridx >= len(self.df):
                    continue  # Skip the last `Total` row
                values.append(self.df[col_name][ridx])
        else:
            ridx = self.table.cursor_row
            if ridx >= len(self.df):
                return None  # Skip the last `Total` row
            values.append(self.df[col_name][ridx])

        return values or None

    def _values_to_expr(self, values: list[Any] | list[dict[str, Any]] | None) -> pl.Expr | None:
        """Convert selected frequency row value(s) into a dataframe filter expression."""

        if not values:
            return None

        if not self.is_multi_column:
            col_name = self.col_names[0]
            expr: pl.Expr | None = None

            for value in values:
                if value is None or value == NULL:
                    this_expr = pl.col(col_name).is_null()
                else:
                    this_expr = pl.col(col_name) == value
                expr = this_expr if expr is None else expr | this_expr

            return expr

        value_maps = [value for value in values if isinstance(value, dict)]
        if not value_maps:
            return None

        expr: pl.Expr | None = None

        for value_map in value_maps:
            per_row_expr: pl.Expr | None = None
            for col in self.col_names:
                value = value_map.get(col)
                if value is None or value == NULL:
                    this_expr = pl.col(col).is_null()
                elif isinstance(value, list):
                    this_expr = pl.col(col) == value
                else:
                    this_expr = pl.col(col) == value

                per_row_expr = this_expr if per_row_expr is None else per_row_expr & this_expr

            if per_row_expr is not None:
                expr = per_row_expr if expr is None else expr | per_row_expr

        return expr

    def filter_or_collect_selected_values(self, action: str = "filter") -> None:
        """Filter or collect rows in the main table using selected frequency row values."""
        values = self.get_values()
        expr = self._values_to_expr(values)
        if expr is None:
            return

        col_name = self.col_names[0]
        cidx = self.dftable.get_cidx(col_name)
        if cidx is None:
            self.notify(f"Column [$warning]{col_name}[/] not found", title="Frequency", severity="warning")
            return

        # Dismiss modal screen(s) to return to main table first: filtering/collecting
        # pushes a transient BusyScreen, so popping afterward would remove that instead.
        while len(self.app._screen_stack) > 1:
            self.app.pop_screen()

        if action == "collect":
            self.dftable.cmd_collect_rows(col_name=col_name, term=expr)
        else:
            self.dftable.cmd_filter_rows(
                {
                    "term": expr,
                    "col_name": col_name,
                    "match_nocase": False,
                    "match_whole": True,
                    "match_literal": True,
                    "match_reverse": False,
                }
            )

        self.dftable.move_cursor(column=cidx)


class HistogramScreen(TableScreen):
    """Modal screen to display histogram of values in a column."""

    def __init__(
        self, dftable: "DataFrameTable", bins: list[float] | None = None, bin_count: int | None = None
    ) -> None:
        super().__init__(dftable)
        self.cidx = dftable.cursor_cidx
        self.bins = bins
        self.bin_count = bin_count
        self.total_count = len(dftable.df)

    def on_mount(self) -> None:
        """Start histogram calculation."""
        super().on_mount()
        self.table.loading = True
        self._calculate_histogram()

    @work(thread=True)
    def _calculate_histogram(self) -> None:
        """Calculate histogram."""
        col = self.dftable.df.columns[self.cidx]
        self.df = (
            self.dftable.df.lazy()
            .select(col)
            .collect()[col]
            .hist(bins=self.bins, bin_count=self.bin_count, include_breakpoint=False)
        ).rename({"category": col, "count": "Count"})
        self.app.call_from_thread(self._on_calc_ready)

    def build_table(self) -> None:
        """Build the histogram table."""
        self.table.clear(columns=True)

        # Create histogram table
        column = self.dftable.df.columns[self.cidx]
        dtype = self.dftable.df.dtypes[self.cidx]
        dc = DtypeConfig(dtype)

        # Add column headers with sort indicators
        columns = [
            (column, "Value", 0),
            ("Count", "Count", 1),
            ("%", "%", 2),
            ("Histogram", "Histogram", 3),
        ]

        for display_name, key, col_idx_num in columns:
            header_text = display_name

            justify = dc.justify if col_idx_num == 0 else ("right" if col_idx_num in (1, 2) else "left")
            self.table.add_column(Text(header_text, justify=justify), key=key)

        # Get style config for Int64 and Float64
        dc_int = DtypeConfig(pl.Int64)
        dc_float = DtypeConfig(pl.Float64)
        bar_width = BAR_COLUMN_WIDTH

        # Add rows to the histogram table
        for ridx, row in enumerate(self.df.iter_rows()):
            column, count = row
            percentage = (count / self.total_count) * 100

            self.table.add_row(
                Text(column, style=dc.style, justify=dc.justify),
                dc_int.format(count, thousand_separator=self.thousand_separator),
                dc_float.format(percentage, thousand_separator=self.thousand_separator),
                Bar(
                    highlight_range=(0.0, percentage / 100 * bar_width),
                    width=bar_width,
                ),
                key=str(ridx),
                label=str(ridx + 1),
            )

        # Add a total row
        self.table.add_row(
            Text("Total", style="bold", justify=dc.justify),
            Text(
                f"{self.total_count:,}" if self.thousand_separator else str(self.total_count),
                style="bold",
                justify="right",
            ),
            Text(
                format_float(100.0, self.thousand_separator),
                style="bold",
                justify="right",
            ),
            Bar(
                highlight_range=(0.0, bar_width),
                width=bar_width,
            ),
            key="total",
        )


class MetaColumnScreen(TableScreen):
    """Modal screen to display metadata about the columns in the dataframe."""

    def on_ready(self) -> None:
        """Initialize the column metadata screen.

        Populates the table with information about each column in the dataframe,
        including ID (1-based index), Name, and Type.
        """
        self.build_table()

    def on_key(self, event: Key) -> None:
        """Handle key press events on the column metadata screen.

        Dispatches metadata commands through the key binding registry, then
        falls back to the common modal table shortcuts.

        Args:
            event: The key event object.
        """
        if self.registry.dispatch(event.key, self.leader, Scope.META_COLUMN_SCREEN, self):
            event.stop()
            event.prevent_default()
            self.reset_leader()
            return

        super().on_key(event)

    def cmd_meta_column_jump(self) -> None:
        """Jump to the selected column in the main table and close metadata."""
        cidx = self.get_cidx()
        self.app.pop_screen()
        self.dftable.move_cursor(column=cidx)

    def cmd_meta_column_show_frequency(self) -> None:
        """Show frequency for the selected column."""
        col_name = self.dftable.df.columns[self.get_cidx()]
        self.show_frequency(col_name)

    def cmd_meta_column_show_statistics(self) -> None:
        """Show statistics for the selected column."""
        col_name = self.dftable.df.columns[self.get_cidx()]
        self.show_statistics(col_name)

    def cmd_meta_column_move_right(self) -> None:
        """Move the selected column right."""
        self.move_selected_column("right")

    def cmd_meta_column_move_left(self) -> None:
        """Move the selected column left."""
        self.move_selected_column("left")

    def move_selected_column(self, direction: str) -> None:
        """Move the selected column and refresh metadata."""
        row_idx, col_idx = self.table.cursor_coordinate
        col_name = self.dftable.df.columns[row_idx]

        if direction == "right":
            self.dftable._move_column("right", col_name=col_name)
            new_row_idx = min(row_idx + 1, len(self.dftable.df.columns) - 1)
        else:
            self.dftable._move_column("left", col_name=col_name)
            new_row_idx = max(row_idx - 1, 0)

        self.build_table()
        self.table.move_cursor(row=new_row_idx, column=col_idx)

    def cmd_meta_column_edit(self) -> None:
        """Rename, cast, or resize the selected column based on cursor column."""
        row_idx = self.table.cursor_row
        col_idx = self.table.cursor_column
        self._resume_row_idx = row_idx
        self._resume_col_idx = col_idx

        if col_idx == 0:
            col_name = self.dftable.df.columns[row_idx]
            self.dftable.cmd_rename_column(col_name=col_name)
        elif col_idx == 1:
            col_name = self.dftable.df.columns[row_idx]
            self.dftable._cast_column_dtype(col_name=col_name)
        elif col_idx == 2:
            col_name = self.dftable.df.columns[row_idx]
            self.dftable.cmd_resize_column(col_name=col_name)

    def cmd_meta_column_delete(self) -> None:
        """Delete the selected column and refresh metadata."""
        row_idx = self.table.cursor_row

        self.dftable.move_cursor(column=row_idx)
        col_name = self.dftable.df.columns[row_idx]
        self.dftable.cmd_delete_column(col_name=col_name)

        if not self.dftable.visible_columns:
            self.app.pop_screen()
            return

        self.build_table()
        self.table.move_cursor(row=row_idx)

    def on_screen_resume(self) -> None:
        """Rebuild metadata after returning from stacked screens (e.g., rename dialog)."""
        row_idx = getattr(self, "_resume_row_idx", self.table.cursor_row)
        col_idx = getattr(self, "_resume_col_idx", self.table.cursor_column)

        # Remove old table and remount a fresh one to avoid stale cached column widths
        self.table.remove()
        self.table = DataTable(zebra_stripes=True)
        self.mount(self.table)

        self.build_table()
        self.table.focus()
        self.table.move_cursor(row=row_idx, column=col_idx)

    def build_table(self) -> None:
        """Build the column metadata table."""
        self.df = pl.DataFrame(
            {
                "Column": self.dftable.df.columns,
                "Type": [str(dtype) for dtype in self.dftable.df.dtypes],
                "Width": [col.width if (col := self.dftable.columns.get(c)) else 0 for c in self.dftable.df.columns],
            }
        )

        col2style = defaultdict(list)
        col2style["Column"] = ""  # No specific style for the "Column" header
        for dtype in self.dftable.df.dtypes:
            col2style["Type"].append(DtypeConfig(dtype).style)

        self.df2table(col_style=col2style)

    def get_cidx(self) -> int:
        """Get the current column index."""
        cidx = self.table.cursor_row
        return cidx


class CellDetailScreen(TableModalScreen):
    """Modal screen to display details of a cell value, including support for nested structures like lists and dicts."""

    def __init__(self, col_name: str, dtype: pl.DataType, cell_value: Any, delimiter: str | None = "|") -> None:
        super().__init__()
        self.col_name = col_name
        self.dtype = dtype
        self.cell_value = cell_value
        self.delimiter = delimiter

    def on_mount(self) -> None:
        """Initialize the cell detail screen."""
        super().on_mount()
        self.build_table()

    def on_key(self, event: Key) -> None:
        """Handle key press events on the column metadata screen.

        Args:
            event: The key event object.
        """
        if self.registry.dispatch(event.key, self.leader, Scope.CELL_DETAIL_SCREEN, self):
            event.stop()
            event.prevent_default()
            self.reset_leader()
            return

        super().on_key(event)

    def cmd_cell_detail_drill_down(self) -> None:
        """Drill into the selected cell-detail value."""
        cidx = self.table.cursor_column
        ridx = self.table.cursor_row
        col_name = self.df.columns[cidx]
        dtype = self.df.dtypes[cidx]
        cell_value = self.df.item(ridx, cidx)

        if dtype == pl.String:
            # String contains the delimiter '|' (indicating a potential list of values)
            if "|" in cell_value:
                self.app.push_screen(CellDetailScreen(col_name, dtype, cell_value))
            # Show long string in a text screen for better readability
            elif len(cell_value) > COLUMN_WIDTH_CAP:
                self.app.push_screen(TextScreen(cell_value))

        # Show cell detail screen if the value is a non-empty list
        elif dtype == pl.List and not cell_value.is_empty():
            if len(cell_value) == 1 and isinstance(cell_value[0], str) and len(cell_value[0]) > COLUMN_WIDTH_CAP:
                self.app.push_screen(TextScreen(cell_value[0]))
            else:
                self.app.push_screen(CellDetailScreen(col_name, dtype, cell_value))

        # or a non-empty dict (struct)
        elif dtype == pl.Struct and cell_value:
            self.app.push_screen(CellDetailScreen(col_name, dtype, cell_value))

    def build_table(self) -> None:
        """Build the list table."""
        col_name = self.col_name
        dtype = self.dtype
        cell_value = self.cell_value

        # Get the column values as a list
        if isinstance(cell_value, pl.Series) and not cell_value.is_empty():
            self.df = pl.DataFrame({col_name: cell_value})
            self.df2table()
        elif isinstance(cell_value, dict) and cell_value:
            self.df = pl.DataFrame(
                {
                    f"{col_name} (Key)": pl.Series(cell_value.keys(), strict=False),
                    f"{col_name} (Value)": pl.Series(cell_value.values(), strict=False),
                }
            )
            self.df2table()
        elif dtype == pl.String and cell_value:
            self.df = pl.DataFrame({col_name: [c for c in cell_value.split(self.delimiter) if c]})
            self.df2table()
        else:
            self.df = pl.DataFrame({col_name: [cell_value]})
            self.df2table()


class SheetScreen(TableModalScreen):
    """Modal screen displaying information about all currently opened tabs."""

    def __init__(self, tabs: dict) -> None:
        """Initialize the SheetScreen.

        Args:
            tabs: Dict mapping TabPane to DataFrameTable instances.
        """
        super().__init__()
        self.tabs = tabs

    def on_ready(self) -> None:
        """Build the table after initial layout."""
        self.build_table()

    def on_key(self, event) -> None:
        """Handle key events."""
        if self.registry.dispatch(event.key, self.leader, Scope.SHEET_SCREEN, self):
            event.stop()
            event.prevent_default()
            self.reset_leader()
            return

        super().on_key(event)

    def cmd_sheet_switch_tab(self) -> None:
        """Switch to the tab under the cursor."""
        self._switch_to_cursor_tab()

    def cmd_sheet_close_tab(self) -> None:
        """Close the tab under the cursor."""
        self._close_cursor_tab()

    def cmd_sheet_rename_tab(self) -> None:
        """Rename the tab under the cursor."""
        self._rename_cursor_tab()

    def cmd_sheet_toggle_selection(self) -> None:
        """Select or deselect the current sheet row."""
        self.toggele_row_selection()
        self.build_table()

    def cmd_sheet_join_selected_tabs(self) -> None:
        """Join the selected tabs."""
        self._join_selected_tabs()

    def _get_selected_tables(self) -> list["DataFrameTable"]:
        """Return selected tables in the visible sheet order."""
        panes = list(self.tabs.keys())
        tables = []
        for row_idx in sorted(self.selected_rows):
            if 0 <= row_idx < len(panes):
                tables.append(self.tabs[panes[row_idx]])
        return tables

    def _join_selected_tabs(self) -> None:
        """Open JoinTableScreen with the two selected sheet rows pre-selected."""
        selected_tables = self._get_selected_tables()

        if len(selected_tables) != 2:
            self.notify("Select exactly 2 tables first with `[$warning]s[/]`", title="Join Tables", severity="warning")
            return

        left, right = selected_tables
        self.app.push_screen(JoinTableScreen(left=left, right=right), callback=self.app._join_table)

    def _switch_to_cursor_tab(self) -> None:
        """Close the SheetScreen and switch to the tab under the cursor."""
        row_idx = self.table.cursor_row
        panes = list(self.tabs.keys())
        if 0 <= row_idx < len(panes):
            target_pane = panes[row_idx]
            self.app.pop_screen()
            self.app.tabbed.active = target_pane.id
            self.tabs[target_pane].focus()

    def _close_cursor_tab(self) -> None:
        """Close the SheetScreen, activate the tab under the cursor, and close it."""
        row_idx = self.table.cursor_row
        panes = list(self.tabs.keys())
        if 0 <= row_idx < len(panes):
            target_pane = panes[row_idx]
            self.app.close_tab(target_pane)
            self.build_table()

    def _rename_cursor_tab(self) -> None:
        """Open rename screen for the tab under the cursor."""
        row_idx = self.table.cursor_row
        panes = list(self.tabs.keys())
        if 0 <= row_idx < len(panes):
            target_pane = panes[row_idx]
            try:
                content_tab = self.app.query_one(f"#--content-tab-{target_pane.id}")
            except Exception:
                return
            self.app.do_rename_tab(content_tab)

    def on_screen_resume(self) -> None:
        """Rebuild the table when returning from a stacked screen."""
        self.build_table()

    def build_table(self) -> None:
        """Build the sheets overview table."""
        rows = []
        for pane, dftable in self.tabs.items():
            # Collect the dataframe if it hasn't been loaded yet
            if dftable.df is None:
                dftable.df = dftable.lf.collect()

            rows.append(
                {
                    "Tab": dftable.tabname,
                    "#Rows": len(dftable.df),
                    "#Cols": len(dftable.df.select(pl.exclude(RID)).columns),
                    "Filename": dftable.filename,
                }
            )

        self.df = pl.DataFrame(rows)
        self.df2table()
