"""Modal screens with Yes/No buttons and their specialized variants."""

from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data_frame_table import DataFrameTable


import polars as pl
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.validation import ValidationResult, Validator
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    SelectionList,
    TabPane,
    TextArea,
)
from textual.widgets.selection_list import Selection
from textual.widgets.tabbed_content import ContentTab

from .commands import Scope
from .common import NULL, RID, DtypeClass, DtypeConfig, tentative_expr, validate_expr
from .keybindings import KeyBinding, format_key_display, parse_key_display


class NonEmptyValueValidator(Validator):
    """Validator that rejects empty or whitespace-only input values."""

    def validate(self, value: str) -> ValidationResult:
        """Return success when input contains non-whitespace characters."""
        if value.strip():
            return self.success()
        return self.failure("Value cannot be empty")


class YMNScreen(ModalScreen):
    """Base class for Yes/Maybe/No modal screens with customizable content and callbacks.

    This widget handles:
    - Yes/Maybe/No button responses
    - Enter key for Yes, Escape for No
    - Optional callback function for Yes action
    - Optional second callback for Maybe action
    """

    DEFAULT_CSS = """
        YMNScreen {
            align: center middle;
        }

        YMNScreen > Container {
            width: auto;
            height: auto;
            border: solid $primary;
            border-title-color: $primary;
            padding: 1 2;
            overflow: auto;
        }

        YMNScreen #button-container {
            margin: 1 0 0 0;
            width: 100%;
            height: auto;
            align: center middle;
        }

        YMNScreen Button {
            height: 3;
            margin: 0 2;
        }
    """

    def __init__(
        self,
        yes: str | dict | Button = "Yes",
        maybe: str | dict | Button | None = None,
        no: str | dict | Button = "No",
        on_yes_callback=None,
        on_maybe_callback=None,
    ) -> None:
        """Initialize the modal screen.

        Creates a customizable Yes/Maybe/No dialog with optional input fields, labels, and checkboxes.

        Args:
            yes: Text or dict for the Yes button. If None, hides the Yes button. Defaults to "Yes".
            maybe: Optional Maybe button text/dict. Defaults to None.
            no: Text or dict for the No button. If None, hides the No button. Defaults to "No".
            on_yes_callback: Optional callable that takes no args and returns the value to dismiss with when Yes is pressed. Defaults to None.
            on_maybe_callback: Optional callable that takes no args and returns the value to dismiss with when Maybe is pressed. Defaults to None.
        """

        super().__init__()
        self.yes = yes
        self.maybe = maybe
        self.no = no
        self.on_yes_callback = on_yes_callback
        self.on_maybe_callback = on_maybe_callback

    def compose(self) -> ComposeResult:
        """Compose the modal screen widget structure.

        Builds the widget hierarchy with optional title, labels, inputs, checkboxes,
        and action buttons based on initialization parameters.

        Yields:
            Widget: The components of the modal screen in rendering order.
        """
        with Horizontal(id="button-container"):
            if self.yes:
                if isinstance(self.yes, Button):
                    pass
                elif isinstance(self.yes, dict):
                    self.yes = Button(**self.yes, id="yes", variant="success", compact=True)
                else:
                    self.yes = Button(self.yes, id="yes", variant="success", compact=True)

                yield self.yes

            if self.maybe:
                if isinstance(self.maybe, Button):
                    pass
                elif isinstance(self.maybe, dict):
                    self.maybe = Button(**self.maybe, id="maybe", variant="warning", compact=True)
                else:
                    self.maybe = Button(self.maybe, id="maybe", variant="warning", compact=True)

                yield self.maybe

            if self.no:
                if isinstance(self.no, Button):
                    pass
                elif isinstance(self.no, dict):
                    self.no = Button(**self.no, id="no", variant="error", compact=True)
                else:
                    self.no = Button(self.no, id="no", variant="error", compact=True)

                yield self.no

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events in the Yes/No screen."""
        if event.button.id == "yes":
            self._handle_yes()
        elif event.button.id == "maybe":
            self._handle_maybe()
        elif event.button.id == "no":
            self.dismiss(None)

    def on_key(self, event) -> None:
        """Handle key press events in the modal screen."""
        if event.key in ("q", "escape"):
            event.stop()
            self.dismiss(None)
        elif event.key == "enter":
            event.stop()
            for button in self.query(Button):
                if button.has_focus:
                    if button.id == "yes":
                        self._handle_yes()
                    elif button.id == "maybe":
                        self._handle_maybe()
                    elif button.id == "no":
                        self.dismiss(None)
                    break
            else:
                self._handle_yes()

    def _handle_yes(self) -> None:
        """Handle Yes button/Enter key press."""
        if self.on_yes_callback:
            result = self.on_yes_callback()
            self.dismiss(result)
        else:
            self.dismiss(True)

    def _handle_maybe(self) -> None:
        """Handle Maybe button press."""
        if self.on_maybe_callback:
            result = self.on_maybe_callback()
            self.dismiss(result)
        else:
            self.dismiss(False)


class YesNoScreen(YMNScreen):
    """Reusable modal screen with Yes/Maybe/No buttons and customizable label and input."""

    # fmt: off
    DEFAULT_CSS = YMNScreen.DEFAULT_CSS.replace("YMNScreen", "YesNoScreen") + """
        YesNoScreen > Container {
            min-width: 48;
            max-width: 72;
        }

        YesNoScreen Label {
            margin: 0;
            width: 100%;
            text-wrap: wrap;
        }

        YesNoScreen Input {
            margin: 0 0 1 0;
        }

        YesNoScreen Input:blur {
            border: solid $secondary;
        }

        YesNoScreen #checkbox-container {
            margin: 0 0 1 0;
            height: auto;
            align: left middle;
        }

        YesNoScreen Checkbox {
            margin: 0;
        }

        YesNoScreen Checkbox:blur {
            border: solid $secondary;
        }
    """
    # fmt: on

    def __init__(
        self,
        title: str | None = None,
        label: str | dict | Label | None = None,
        input: str | dict | Input | None = None,
        label2: str | dict | Label | None = None,
        input2: str | dict | Input | None = None,
        label3: str | dict | Label | None = None,
        checkbox: str | dict | Checkbox | None = None,
        checkbox2: str | dict | Checkbox | None = None,
        checkbox3: str | dict | Checkbox | None = None,
        checkbox4: str | dict | Checkbox | None = None,
        yes: str | dict | Button = "Yes",
        maybe: str | dict | Button | None = None,
        no: str | dict | Button = "No",
        on_yes_callback=None,
        on_maybe_callback=None,
    ) -> None:
        """Initialize the modal screen.

        Creates a customizable Yes/No dialog with optional input fields, labels, and checkboxes.

        Args:
            title: The title to display in the border. Defaults to None.
            label: Optional label to display below title as a Label. Defaults to None.
            input: Optional input widget or value to pre-fill. If None, no Input is shown. Defaults to None.
            label2: Optional second label widget. Defaults to None.
            input2: Optional second input widget or value. Defaults to None.
            label3: Optional third label widget. Defaults to None.
            checkbox: Optional checkbox widget or label. Defaults to None.
            checkbox2: Optional second checkbox widget or label. Defaults to None.
            checkbox3: Optional third checkbox widget or label. Defaults to None.
            checkbox4: Optional fourth checkbox widget or label. Defaults to None.
            yes: Text or dict for the Yes button. If None, hides the Yes button. Defaults to "Yes".
            maybe: Optional Maybe button text/dict. Defaults to None.
            no: Text or dict for the No button. If None, hides the No button. Defaults to "No".
            on_yes_callback: Optional callable that takes no args and returns the value to dismiss with when Yes is pressed. Defaults to None.
            on_maybe_callback: Optional callable that takes no args and returns the value to dismiss with when Maybe is pressed. Defaults to None.
        """
        super().__init__(
            yes=yes,
            maybe=maybe,
            no=no,
            on_yes_callback=on_yes_callback,
            on_maybe_callback=on_maybe_callback,
        )
        self.title = title
        self.label = label
        self.input = input
        self.label2 = label2
        self.input2 = input2
        self.label3 = label3
        self.checkbox = checkbox
        self.checkbox2 = checkbox2
        self.checkbox3 = checkbox3
        self.checkbox4 = checkbox4

    def compose(self) -> ComposeResult:
        """Compose the modal screen widget structure.

        Builds the widget hierarchy with optional title, labels, inputs, checkboxes,
        and action buttons based on initialization parameters.

        Yields:
            Widget: The components of the modal screen in rendering order.
        """
        with Container(id="modal-container") as container:
            if self.title:
                container.border_title = self.title

            if self.label:
                if isinstance(self.label, Label):
                    pass
                elif isinstance(self.label, dict):
                    self.label = Label(**self.label)
                else:
                    self.label = Label(self.label)
                yield self.label

            if self.input is not None:
                if isinstance(self.input, Input):
                    pass
                elif isinstance(self.input, dict):
                    self.input = Input(**self.input)
                else:
                    self.input = Input(self.input)
                self.input.select_all()
                yield self.input

            if self.label2:
                if isinstance(self.label2, Label):
                    pass
                elif isinstance(self.label2, dict):
                    self.label2 = Label(**self.label2)
                else:
                    self.label2 = Label(self.label2)
                yield self.label2

            if self.input2 is not None:
                if isinstance(self.input2, Input):
                    pass
                elif isinstance(self.input2, dict):
                    self.input2 = Input(**self.input2)
                else:
                    self.input2 = Input(self.input2)
                self.input2.select_all()
                yield self.input2

            if self.label3:
                if isinstance(self.label3, Label):
                    pass
                elif isinstance(self.label3, dict):
                    self.label3 = Label(**self.label3)
                else:
                    self.label3 = Label(self.label3)
                yield self.label3

            if any([self.checkbox, self.checkbox2, self.checkbox3, self.checkbox4]):
                with Horizontal(id="checkbox-container"):
                    if self.checkbox:
                        if isinstance(self.checkbox, Checkbox):
                            pass
                        elif isinstance(self.checkbox, dict):
                            self.checkbox = Checkbox(**self.checkbox)
                        else:
                            self.checkbox = Checkbox(self.checkbox)
                        yield self.checkbox

                    if self.checkbox2:
                        if isinstance(self.checkbox2, Checkbox):
                            pass
                        elif isinstance(self.checkbox2, dict):
                            self.checkbox2 = Checkbox(**self.checkbox2)
                        else:
                            self.checkbox2 = Checkbox(self.checkbox2)
                        yield self.checkbox2

                    if self.checkbox3:
                        if isinstance(self.checkbox3, Checkbox):
                            pass
                        elif isinstance(self.checkbox3, dict):
                            self.checkbox3 = Checkbox(**self.checkbox3)
                        else:
                            self.checkbox3 = Checkbox(self.checkbox3)
                        yield self.checkbox3

                    if self.checkbox4:
                        if isinstance(self.checkbox4, Checkbox):
                            pass
                        elif isinstance(self.checkbox4, dict):
                            self.checkbox4 = Checkbox(**self.checkbox4)
                        else:
                            self.checkbox4 = Checkbox(self.checkbox4)
                        yield self.checkbox4

            if self.yes or self.no or self.maybe:
                yield from super().compose()


class ConfirmScreen(YesNoScreen):
    """Modal screen to ask for confirmation."""

    CSS = """
        ConfirmScreen > Container {
            min-width: 64;
        }
    """

    def __init__(
        self, title: str | None = None, label=None, input=None, yes="Confirm", maybe: str | None = None, no="Cancel"
    ):
        super().__init__(
            title=title,
            label=label,
            input=input,
            yes=yes,
            maybe=maybe,
            no=no,
            on_yes_callback=self._get_input,
        )

    def _get_input(self):
        """Get input value when Yes is pressed."""
        # Do not strip to preserve spaces
        return self.input.value if self.input else True


class KeyCaptureScreen(ModalScreen):
    """Modal screen that captures a key for a Commands tab cell."""

    DEFAULT_CSS = """
        KeyCaptureScreen {
            align: center middle;
        }

        KeyCaptureScreen > Container {
            min-width: 56;
            max-width: 72;
            height: auto;
            border: solid $primary;
            border-title-color: $primary;
            padding: 1 2;
        }

        KeyCaptureScreen Label {
            width: 100%;
            text-wrap: wrap;
        }

        KeyCaptureScreen #button-container {
            margin: 1 0 0 0;
            width: 100%;
            height: auto;
            align: center middle;
        }

        KeyCaptureScreen #status-label {
            margin: 1 0 1 0;
        }

        KeyCaptureScreen Button {
            height: 3;
            margin: 0 2;
        }
    """

    def __init__(
        self,
        command_id: str,
        column_name: str,
        current_value: str = "",
        leader: str = "",
        key: str = "",
        scope: str = "",
    ) -> None:
        """Initialize the key capture modal.

        Args:
            command_id: The command whose key binding is being edited.
            column_name: The keybinding column being edited.
            current_value: The current display value for the selected cell.
            leader: The row's current leader display value.
            key: The row's current key display value.
            scope: The row's command scope display value.
        """
        super().__init__()
        self.command_id = command_id
        self.column_name = column_name
        self.current_value = current_value
        self.leader = leader or ""
        self.key = key or ""
        self.scope = scope
        self.status_label: Label | None = None

    def compose(self) -> ComposeResult:
        """Compose the key capture modal."""
        with Container(id="modal-container") as container:
            container.border_title = "Capture Key Binding"
            yield Label(f"Command: [$success]{self.command_id}[/]")
            yield Label(f"Cell: [$success]{self.column_name}[/]")
            if self.current_value:
                yield Label(f"Current: [$accent]{self.current_value}[/]")
            self.status_label = Label("Press a key to use as this cell's value.", id="status-label")
            yield self.status_label
            with Horizontal(id="button-container"):
                yield Button("Cancel", id="cancel", variant="error", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Cancel capture when the Cancel button is pressed."""
        if event.button.id == "cancel":
            self.dismiss(None)

    def on_key(self, event: Key) -> None:
        """Capture the next key event as a binding."""
        event.stop()
        event.prevent_default()

        key_display = format_key_display(event.key)
        if not self.validate_keybinding(key_display):
            return

        self.dismiss(key_display)

    def validate_keybinding(self, key_display: str) -> bool:
        """Check whether a captured key can be used for this keybinding cell."""
        if self.column_name == "Leader" and key_display not in {"g", "z"}:
            self.set_status("Leader must be [$success]g[/] or [$success]z[/]. Press another key.")
            return False

        leader = key_display if self.column_name == "Leader" else self.leader
        key = key_display if self.column_name == "Key" else self.key
        if not key:
            return True

        try:
            binding = KeyBinding(
                leader=leader,
                key=parse_key_display(key),
                scope=Scope(self.scope),
                command_id=self.command_id,
            )
        except ValueError as e:
            self.set_status(f"Invalid key binding scope: [$error]{e}[/].")
            return False

        existing_binding = self.app.key_registry.lookup(binding.key, binding.leader, binding.scope)
        if existing_binding is not None and existing_binding.command_id != binding.command_id:
            self.set_status(
                f"[$warning]{binding.display_key}[/] is already bound to "
                f"[$accent]{existing_binding.command_id}[/]. Press another key."
            )
            return False

        return True

    def set_status(self, message: str) -> None:
        """Update the modal status text."""
        if self.status_label is not None:
            self.status_label.update(message)


class EditCellScreen(YesNoScreen):
    """Modal screen to edit a single cell value."""

    def __init__(self, ridx: int, col_name: str, df: pl.DataFrame) -> None:
        """Initialize the edit-cell modal for a dataframe cell.

        Args:
            ridx: The dataframe row index to edit.
            col_name: The dataframe column name to edit.
            df: The dataframe containing the editable cell.
        """
        self.ridx = ridx
        self.col_name = col_name
        self.dtype = df.schema[col_name]

        # Label
        content = f"[$success]{col_name}[/] ([$accent]{self.dtype}[/])"

        # Input
        df_value = df.item(ridx, col_name)
        if df_value is None:
            self.input_value = NULL
        elif isinstance(self.dtype, pl.List) and isinstance(df_value, pl.Series):
            self.input_value = "[" + ", ".join(repr(v) for v in df_value) + "]"
        else:
            self.input_value = str(df_value)

        super().__init__(
            title="Edit Cell",
            label=content,
            input={
                "value": self.input_value,
                "type": DtypeConfig(self.dtype).itype,
            },
            yes="Apply",
            no="Cancel",
            on_yes_callback=self._validate_input,
        )

    def _validate_input(self) -> tuple[int, str, Any | None] | None:
        """Validate and save the edited value."""
        new_value_str = self.input.value  # Do not strip to preserve spaces
        new_value: str | None = None

        # Handle empty input
        if not new_value_str:
            new_value = ""
            self.notify(
                "Empty value provided. If you want to clear the cell, press [$warning]Delete[/].",
                title="Edit Cell",
                severity="warning",
            )
        # Check if value changed
        elif new_value_str == self.input_value:
            new_value = None
            self.notify("No changes made", title="Edit Cell", severity="warning")
        else:
            # Parse and validate based on column dtype
            try:
                if isinstance(self.dtype, pl.List):
                    inner_dtype = self.dtype.inner
                    inner_convert = DtypeConfig(inner_dtype).convert
                    # Accept "1, 2, 3" or "[1, 2, 3]" or "['a', 'b']"
                    stripped = new_value_str.strip()
                    if stripped.startswith("[") and stripped.endswith("]"):
                        items = eval(stripped)
                        new_value = [inner_convert(v) for v in items]
                    else:
                        new_value = [inner_convert(v.strip()) for v in stripped.split(",") if v.strip()]
                else:
                    new_value = DtypeConfig(self.dtype).convert(new_value_str)
            except Exception as e:
                self.notify(
                    f"Failed to convert [$error]{new_value_str}[/] to [$accent]{self.dtype}[/]: {e}",
                    title="Edit Cell",
                    severity="error",
                )
                return None

        # New value
        return self.ridx, self.col_name, new_value


class SearchScreen(YesNoScreen):
    """Modal screen to search by value or expression."""

    CSS = YesNoScreen.DEFAULT_CSS.replace("YesNoScreen", "SearchScreen").replace("max-width: 60", "max-width: 70")

    def __init__(self, title: str, col_name: str, term: str):
        self.col_name = col_name

        EXPR = f"{NULL}, Fire, $1 > 50, $name == 'text', $_ > 100, $a < $b"
        label = f"By value or Polars expression, e.g., {EXPR}"

        super().__init__(
            title=title,
            label=label,
            input=term,
            label2="Match options:",
            checkbox=Checkbox("Nocase", id="checkbox-nocase", tooltip="Ignore letter case when matching"),
            checkbox2=Checkbox("Whole", id="checkbox-whole", tooltip="Match whole words only"),
            checkbox3=Checkbox("Literal", id="checkbox-literal", tooltip="Treat input as plain text instead of regex"),
            checkbox4=Checkbox("Reverse", id="checkbox-reverse", tooltip="Invert the match result"),
            yes="Search",
            no="Cancel",
            on_yes_callback=self._get_input,
        )

    def _get_input(self) -> dict:
        """Get input."""
        term = self.input.value  # Do not strip to preserve spaces
        match_nocase = self.checkbox.value
        match_whole = self.checkbox2.value
        match_literal = self.checkbox3.value
        match_reverse = self.checkbox4.value

        return {
            "term": term,
            "col_name": self.col_name,
            "match_nocase": match_nocase,
            "match_whole": match_whole,
            "match_literal": match_literal,
            "match_reverse": match_reverse,
        }


class FreezeScreen(YesNoScreen):
    """Modal screen to pin rows and columns.

    Accepts one value for fixed rows, or two space-separated values for fixed rows and columns.
    """

    def __init__(self):
        super().__init__(
            title="Freeze Rows / Columns",
            label="Enter number of fixed rows",
            input={"value": "0", "type": "number"},
            label2="Enter number of fixed columns",
            input2={"value": "0", "type": "number"},
            yes="Apply",
            no="Cancel",
            on_yes_callback=self._get_input,
        )

    def _get_input(self) -> tuple[int, int] | None:
        """Parse and validate the pin input.

        Returns:
            Tuple of (fixed_rows, fixed_columns) or None if invalid.
        """
        fixed_rows = int(self.input.value.strip())
        fixed_cols = int(self.input2.value.strip())

        if fixed_rows < 0 or fixed_cols < 0:
            self.notify("Values must be non-negative", title="Pin", severity="error")
            return None

        return fixed_rows, fixed_cols


class RenameColumnScreen(YesNoScreen):
    """Modal screen to rename a column."""

    def __init__(self, col_name: str, existing_columns: list[str]):
        self.col_name = col_name
        self.existing_columns = [c for c in existing_columns if c != col_name]

        # Label
        content = f"Rename header [$success]{col_name}[/]"

        super().__init__(
            title="Rename Column",
            label=content,
            input=Input(col_name, validators=NonEmptyValueValidator()),
            yes="Rename",
            no="Cancel",
            on_yes_callback=self._validate_input,
        )

    def _validate_input(self) -> tuple[str, Any]:
        """Validate and save the new column name."""
        new_name = self.input.value.strip()

        # Check if name is empty
        if not new_name:
            self.notify("Column name cannot be empty", title="Rename", severity="error")

        # Check if name changed
        elif new_name == self.col_name:
            self.notify("No changes made", title="Rename", severity="warning")
            new_name = None

        # Check if name already exists
        elif new_name in self.existing_columns:
            self.notify(
                f"Column [$error]{new_name}[/] already exists",
                title="Rename",
                severity="error",
            )
            new_name = None

        # Return new name
        return self.col_name, new_name


class EditColumnScreen(YesNoScreen):
    """Modal screen to edit an entire column with an expression."""

    def __init__(self, col_name: str, df: pl.DataFrame):
        self.col_name = col_name
        self.df = df
        super().__init__(
            title="Edit Column",
            label=f"By value or Polars expression, e.g., abc, pl.lit(7), {NULL}, $_ * 2, $1 + $2, $_.str.to_uppercase(), pl.arange(0, pl.len())",
            input="$_",
            yes="Apply",
            no="Cancel",
            on_yes_callback=self._get_input,
        )

    def _get_input(self) -> tuple[str, str]:
        """Get input."""
        term = self.input.value  # Do not strip to preserve spaces
        return term, self.col_name


class AddColumnScreen(YesNoScreen):
    """Modal screen to add a new column with an expression."""

    def __init__(self, col_name: str, df: pl.DataFrame, link: bool = False):
        self.col_name = col_name
        self.df = df
        self.link = link
        self.existing_columns = set(df.columns)

        label2 = (
            "Link template, e.g., https://example.com/$1/id/$_, PC/compound/$cid"
            if link
            else f"Value or Polars expression, e.g., abc, pl.lit(123), {NULL}, $_ * 2, $1 + $total, $_ + '_suffix', $_.str.to_uppercase()"
        )

        super().__init__(
            title="Add Column",
            label="Column name",
            input="Link" if link else "New column",
            label2=label2,
            input2=Input(placeholder="Link template" if link else "Value or Polars expression"),
            yes="Add",
            no="Cancel",
            on_yes_callback=self._get_input,
        )

    def _get_input(self) -> tuple[str, str, Any] | None:
        """Validate and return the new column configuration."""
        new_col_name = self.input.value.strip()
        term = self.input2.value  # Do not strip to preserve spaces

        # Validate column name
        if not new_col_name:
            self.notify("Column name cannot be empty", title="Add Column", severity="error")
            return None

        if new_col_name in self.existing_columns:
            self.notify(
                f"Column [$error]{new_col_name}[/] already exists",
                title="Add Column",
                severity="error",
            )
            return None

        if term == NULL:
            return self.col_name, new_col_name, pl.lit(None)
        elif self.link:
            # Treat as link template
            return self.col_name, new_col_name, term
        elif tentative_expr(term):
            try:
                expr = validate_expr(term, self.df.columns, self.col_name, self.df)
                return self.col_name, new_col_name, expr
            except Exception as e:
                self.notify(f"Invalid expression [$error]{term}[/]: {e}", title="Add Column", severity="error")
            return None
        else:
            # Treat as literal value
            dtype = self.df.schema[self.col_name]
            try:
                value = DtypeConfig(dtype).convert(term)
                return self.col_name, new_col_name, pl.lit(value)
            except Exception as e:
                self.notify(
                    f"Unable to convert [$warning]{term}[/] to [$accent]{dtype}[/]: {e}. Cast to string.",
                    title="Add Column",
                    severity="warning",
                )
                return self.col_name, new_col_name, pl.lit(term)


class AddLinkScreen(AddColumnScreen):
    """Modal screen to add a new link column with user-provided expressions.

    Allows user to specify a column name and a value or Polars expression that will be
    evaluated to create links. A new column is created with the resulting link values.
    Inherits column name and expression validation from AddColumnScreen.
    """

    def __init__(self, col_name: str, df: pl.DataFrame):
        super().__init__(col_name, df, link=True)


class FindReplaceScreen(YesNoScreen):
    """Modal screen to replace column values with an expression."""

    def __init__(self, title: str, dftable: "DataFrameTable"):
        if (cursor_value := dftable.cursor_value) is None:
            term_find = NULL
        else:
            term_find = str(cursor_value)

        super().__init__(
            title=title,
            label="Find",
            input=term_find,
            label2="Replace with",
            input2="new value or expression",
            label3="Match options:",
            checkbox=Checkbox("Nocase", id="checkbox-nocase", tooltip="Ignore letter case when matching"),
            checkbox2=Checkbox("Whole", id="checkbox-whole", tooltip="Match whole words only"),
            checkbox3=Checkbox("Literal", id="checkbox-literal", tooltip="Treat input as plain text instead of regex"),
            yes="Replace",
            maybe="Replace All",
            no="Cancel",
            on_yes_callback=self._get_input,
            on_maybe_callback=partial(self._get_input, replace_all=True),
        )

    def _get_input(self, replace_all: bool = False) -> dict:
        """Get input."""
        term_find = self.input.value  # Do not strip to preserve spaces
        term_replace = self.input2.value  # Do not strip to preserve spaces
        match_nocase = self.checkbox.value
        match_whole = self.checkbox2.value
        match_literal = self.checkbox3.value

        return {
            "replace_all": replace_all,
            "term_find": term_find,
            "term_replace": term_replace,
            "match_nocase": match_nocase,
            "match_whole": match_whole,
            "match_literal": match_literal,
        }


class RenameTabScreen(YesNoScreen):
    """Modal screen to rename a tab."""

    def __init__(self, content_tab: ContentTab, existing_tabs: list[TabPane]):
        self.content_tab = content_tab
        self.existing_tabs = existing_tabs
        tab_name = content_tab.label_text

        super().__init__(
            title="Rename Tab",
            label="New tab name",
            input={"value": tab_name},
            yes="Rename",
            no="Cancel",
            on_yes_callback=self._validate_input,
        )

    def _validate_input(self) -> tuple[ContentTab, str] | None:
        """Validate and save the new tab name."""
        new_name = self.input.value.strip()

        # Check if name is empty
        if not new_name:
            self.notify("Tab name cannot be empty", title="Rename Tab", severity="error")
            return None

        # Check if name changed
        if new_name == self.content_tab.label_text:
            self.notify("No changes made", title="Rename Tab", severity="warning")
            return None

        # Check if name already exists
        if new_name in self.existing_tabs:
            self.notify(f"Tab [$error]{new_name}[/] already exists", title="Rename Tab", severity="error")
            return None

        # Return new name
        return self.content_tab, new_name


class CustomBinScreen(YesNoScreen):
    """Modal screen to specify custom bins for histogram."""

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(
            title="Custom Bins",
            label="Enter number of bins or bin breakpoints (e.g., 5 or 0 10 20 30)",
            input={"value": "10"},
            yes="Apply",
            no="Cancel",
            on_yes_callback=self._get_input,
        )

    def _get_input(self) -> tuple[int | None, list[float] | None] | None:
        """Get and validate the row index input."""
        row_str = self.input.value.strip()

        bin_count, bins = None, None

        try:
            bin_count = int(row_str)
        except ValueError:
            try:
                bins = set()

                for b in row_str.split():
                    # 0:10 -> 0, 1, 2, ..., 10
                    # 0:100:10 -> 0, 10, 20, ..., 100
                    if ":" in b:
                        toks = [float(t) for t in b.split(":")]
                        start = self.min_value if toks[0] < self.min_value else toks[0]
                        end = self.max_value if toks[1] > self.max_value else toks[1]
                        step = 1 if len(toks) == 2 else toks[2]

                        while start <= end:
                            bins.add(start)
                            start += step

                            if start > end:
                                bins.add(end)
                                break
                    else:
                        val = float(b)
                        bins.add(val)

                bins = sorted(bins)
                if len(bins) < 2:
                    raise ValueError("At least two bin breakpoints are required")

                if bins[0] > self.min_value:
                    bins = [self.min_value] + bins
                if bins[-1] < self.max_value:
                    bins = bins + [self.max_value]

                bins = [float(b) for b in bins]
            except ValueError:
                self.notify(
                    "Please enter a valid integer for bin count or space-separated numbers for bin breakpoints",
                    title="Custom Bins",
                    severity="error",
                    timeout=10,
                )
                return None

        return (bin_count, bins)


class SimpleSqlScreen(YMNScreen):
    """Simple SQL query screen."""

    CSS = """
        SimpleSqlScreen SelectionList {
            width: auto;
            min-width: 60;
            margin: 0 0 1 0;
        }

        SimpleSqlScreen SelectionList:blur {
            border: solid $secondary;
        }

        SimpleSqlScreen Label {
            width: auto;
        }

        SimpleSqlScreen Input {
            width: auto;
        }

        SimpleSqlScreen Input:blur {
            border: solid $secondary;
        }
    """

    def __init__(self, dftable: "DataFrameTable") -> None:
        """Initialize the simple SQL screen.

        Sets up the modal screen with reference to the main DataFrameTable widget
        and stores the DataFrame for display.

        Args:
            dftable: Reference to the parent DataFrameTable widget.
        """
        super().__init__(
            yes="Query",
            maybe="Query to Tab",
            no="Cancel",
            on_yes_callback=self.handle_simple,
            on_maybe_callback=partial(self.handle_simple, new_tab=True),
        )
        self.dftable = dftable  # DataFrameTable

    def compose(self) -> ComposeResult:
        """Compose the simple SQL screen widget structure."""
        with Container(id="sql-container") as container:
            container.border_title = "SQL Query Builder"
            yield Label("SELECT columns (default to all if none selected)", id="select-label")
            yield SelectionList(
                *[
                    Selection(col, col)
                    for col in self.dftable.df.columns
                    if col not in self.dftable.hidden_columns and col != RID
                ],
                id="column-selection",
            )
            yield Label("WHERE condition (optional)", id="where-label")
            yield Input(placeholder="e.g., age > 30 and height < 180", id="where-input")
            yield from super().compose()

    def handle_simple(self, new_tab: bool = False) -> tuple[str, str, bool]:
        """Build and return the (columns, where_clause, new_tab) tuple from widget state."""
        selections = self.query_one(SelectionList).selected
        if not selections:
            selections = [
                col for col in self.dftable.df.columns if col not in self.dftable.hidden_columns and col != RID
            ]

        columns = ", ".join(f"`{s}`" for s in selections)
        where = self.query_one(Input).value.strip()

        return columns, where, new_tab


class AdvancedSqlScreen(YMNScreen):
    """Advanced SQL query screen."""

    CSS = """
        AdvancedSqlScreen TextArea {
            width: auto;
            height: auto;
            min-width: 60;
            min-height: 10;
        }
    """

    def __init__(self, dftable: "DataFrameTable") -> None:
        """Initialize the advanced SQL screen.

        Args:
            dftable: Reference to the parent DataFrameTable widget.
        """
        super().__init__(
            yes="Query",
            maybe="Query to Tab",
            no="Cancel",
            on_yes_callback=self.handle_advanced,
            on_maybe_callback=partial(self.handle_advanced, new_tab=True),
        )
        self.dftable = dftable  # DataFrameTable

    def compose(self) -> ComposeResult:
        """Compose the advanced SQL screen widget structure."""
        with Container(id="sql-container") as container:
            container.border_title = "Advanced SQL Query Builder"
            yield TextArea.code_editor(
                placeholder="Enter SQL query, e.g., \n\nSELECT * \nFROM self \nWHERE age > 30\n\n* use 'self' as the table name\n* use backtick (`) to quote column name with spaces",
                id="sql-textarea",
                language="sql",
                tab_behavior="focus",  # Tab to focus next and allow Esc to exit
            )
            yield from super().compose()

    def handle_advanced(self, new_tab: bool = False) -> tuple[str, bool]:
        """Return the (sql_query, new_tab) tuple from the TextArea."""
        return self.query_one(TextArea).text.strip(), new_tab


class NewTabScreen(YMNScreen):
    """A screen for creating a new tab based on an input Polars expression."""

    CSS = """
        NewTabScreen TextArea {
            width: auto;
            height: auto;
            min-width: 60;
            min-height: 14;
        }
    """

    def __init__(self) -> None:
        """Initialize the new tab screen."""

        super().__init__(
            yes="Create",
            no="Cancel",
            on_yes_callback=self._get_input,
        )

    def compose(self) -> ComposeResult:
        """Compose the new tab screen widget structure."""
        with Container(id="new-tab-container") as container:
            container.border_title = "New Tab from Polars expression"
            yield TextArea.code_editor(
                placeholder="Enter expression, e.g., \n\n- $2 > 30\n\n- self.select('name', 'age')\n\n- self.filter($age > 30)\n\n* use $1, $2, ... for column references by index\n* use $column_name for column references by name\n* use 'self' to reference the current dataframe",
                id="new-tab-textarea",
                language="python",
                tab_behavior="focus",  # Tab to focus next and allow Esc to exit
            )
            yield from super().compose()

    def _get_input(self) -> str:
        """Return the Polars expression text entered in the TextArea."""
        return self.query_one(TextArea).text.strip()


class FilterNumericScreen(YMNScreen):
    """A screen for filtering a numeric column."""

    CSS = """
        FilterNumericScreen .condition-row {
            width: auto;
            height: auto;
            min-width: 50;
        }

        FilterNumericScreen .condition-row Label {
            width: 2;
            margin: 1;
        }

        FilterNumericScreen .condition-row Input {
            width: auto;
            min-width: 48;
        }
    """

    def __init__(self, s: pl.Series, col_name: str, dc: DtypeClass, cursor_value: int | float | None) -> None:
        """Initialize the filter numeric column screen."""
        super().__init__(
            yes="Filter",
            no="Cancel",
            on_yes_callback=self._get_input,
        )
        self.s = s
        self.col_name = col_name
        self.dc = dc
        self.placeholder = NULL if cursor_value is None else str(cursor_value)

    def compose(self) -> ComposeResult:
        """Compose the filter numeric column screen widget structure."""
        min_value = self.s.min()
        max_value = self.s.max()

        with Container(id="filter-numeric-column-container") as container:
            container.border_title = "Filter Rows by Column Value"
            yield Horizontal(
                Label("="),
                Input(
                    placeholder=self.placeholder,
                    id="condition-eq",
                    tooltip="Enter a value to filter rows where the column is equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(
                Label("!="),
                Input(
                    placeholder=self.placeholder,
                    id="condition-neq",
                    tooltip="Enter a value to filter rows where the column is not equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(
                Label("<"),
                Input(placeholder=f"{max_value}", id="condition-lt", type=self.dc.itype, valid_empty=True),
                classes="condition-row",
            )
            yield Horizontal(
                Label("<="),
                Input(placeholder=f"{max_value}", id="condition-lte", type=self.dc.itype, valid_empty=True),
                classes="condition-row",
            )
            yield Horizontal(
                Label(">="),
                Input(placeholder=f"{min_value}", id="condition-gte", type=self.dc.itype, valid_empty=True),
                classes="condition-row",
            )
            yield Horizontal(
                Label(">"),
                Input(placeholder=f"{min_value}", id="condition-gt", type=self.dc.itype, valid_empty=True),
                classes="condition-row",
            )
            yield from super().compose()

    def _get_input(self) -> tuple[pl.Expr | None, str]:
        """Build and return a Polars filter expression from the numeric condition inputs.

        Returns:
            A tuple of (filter_expression, col_name), where filter_expression is None
            if no conditions were entered.
        """
        col = self.col_name
        expr: pl.Expr | None = None

        eq = self.query_one("#condition-eq", Input).value.strip()
        if eq:
            expr = pl.col(col).is_null() if eq == NULL else pl.col(col) == self.dc.convert(eq)
        else:
            neq = self.query_one("#condition-neq", Input).value.strip()
            if neq:
                e = pl.col(col).is_not_null() if neq == NULL else pl.col(col) != self.dc.convert(neq)
                expr = e if expr is None else expr & e

            lt = self.query_one("#condition-lt", Input).value.strip()
            if lt:
                e = pl.col(col) < self.dc.convert(lt)
                expr = e if expr is None else expr & e

            lte = self.query_one("#condition-lte", Input).value.strip()
            if lte:
                e = pl.col(col) <= self.dc.convert(lte)
                expr = e if expr is None else expr & e

            gte = self.query_one("#condition-gte", Input).value.strip()
            if gte:
                e = pl.col(col) >= self.dc.convert(gte)
                expr = e if expr is None else expr & e

            gt = self.query_one("#condition-gt", Input).value.strip()
            if gt:
                e = pl.col(col) > self.dc.convert(gt)
                expr = e if expr is None else expr & e

        return expr, self.col_name


class FilterTemporalScreen(YMNScreen):
    """A screen for filtering a temporal column."""

    CSS = """
        FilterTemporalScreen .condition-row {
            width: auto;
            height: auto;
            min-width: 50;
        }

        FilterTemporalScreen .condition-row Label {
            width: 2;
            margin: 1;
        }

        FilterTemporalScreen .condition-row Input {
            width: auto;
            min-width: 48;
        }
    """

    def __init__(self, s: pl.Series, col_name: str, dc: DtypeClass, cursor_value: Any | None) -> None:
        """Initialize the filter temporal column screen.

        Args:
            s: The source series for the selected temporal column.
            col_name: The selected column name.
            dc: Data type configuration for the temporal column.
            cursor_value: The current cell value used as the default placeholder.
        """
        super().__init__(
            yes="Filter",
            no="Cancel",
            on_yes_callback=self._get_input,
        )
        self.s = s
        self.col_name = col_name
        self.dc = dc
        self.placeholder = NULL if cursor_value is None else str(cursor_value)

    def compose(self) -> ComposeResult:
        """Compose the filter temporal column screen widget structure."""
        min_value = self.s.min()
        max_value = self.s.max()

        with Container(id="filter-temporal-column-container") as container:
            container.border_title = "Filter Rows by Column Value"
            yield Horizontal(
                Label("="),
                Input(
                    placeholder=self.placeholder,
                    id="condition-eq",
                    tooltip="Enter a value to filter rows where the column is equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(
                Label("!="),
                Input(
                    placeholder=self.placeholder,
                    id="condition-neq",
                    tooltip="Enter a value to filter rows where the column is not equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(
                Label("<"),
                Input(placeholder=f"{max_value}", id="condition-lt", valid_empty=True),
                classes="condition-row",
            )
            yield Horizontal(
                Label("<="),
                Input(placeholder=f"{max_value}", id="condition-lte", valid_empty=True),
                classes="condition-row",
            )
            yield Horizontal(
                Label(">="),
                Input(placeholder=f"{min_value}", id="condition-gte", valid_empty=True),
                classes="condition-row",
            )
            yield Horizontal(
                Label(">"),
                Input(placeholder=f"{min_value}", id="condition-gt", valid_empty=True),
                classes="condition-row",
            )
            yield from super().compose()

    def _temporal_literal(self, value: str) -> pl.Expr | None:
        """Build a typed temporal literal for comparisons.

        Args:
            value: User-provided temporal value as text.

        Returns:
            A Polars expression parsed to the same dtype as the source series.
        """
        if self.s.dtype == pl.Date:
            return pl.lit(value).str.to_date()
        elif self.s.dtype == pl.Time:
            return pl.lit(value).str.to_time()
        elif self.s.dtype == pl.Datetime:
            return pl.lit(value).str.to_datetime()
        else:
            self.notify(f"Unsupported temporal dtype: [$warning]{self.s.dtype}[/]", severity="warning")
            return None

    def _get_input(self) -> tuple[pl.Expr | None, str]:
        """Build and return a Polars filter expression from the temporal condition inputs.

        Returns:
            A tuple of (filter_expression, col_name), where filter_expression is None
            if no conditions were entered.
        """
        col = self.col_name
        expr: pl.Expr | None = None

        eq = self.query_one("#condition-eq", Input).value.strip()
        if eq and (t := self._temporal_literal(eq)) is not None:
            expr = pl.col(col).is_null() if eq == NULL else pl.col(col) == t
        else:
            neq = self.query_one("#condition-neq", Input).value.strip()
            if neq and (t := self._temporal_literal(neq)) is not None:
                e = pl.col(col).is_not_null() if neq == NULL else pl.col(col) != t
                expr = e if expr is None else expr & e

            lt = self.query_one("#condition-lt", Input).value.strip()
            if lt and (t := self._temporal_literal(lt)) is not None:
                e = pl.col(col) < t
                expr = e if expr is None else expr & e

            lte = self.query_one("#condition-lte", Input).value.strip()
            if lte and (t := self._temporal_literal(lte)) is not None:
                e = pl.col(col) <= t
                expr = e if expr is None else expr & e

            gte = self.query_one("#condition-gte", Input).value.strip()
            if gte and (t := self._temporal_literal(gte)) is not None:
                e = pl.col(col) >= t
                expr = e if expr is None else expr & e

            gt = self.query_one("#condition-gt", Input).value.strip()
            if gt and (t := self._temporal_literal(gt)) is not None:
                e = pl.col(col) > t
                expr = e if expr is None else expr & e

        return expr, self.col_name


class FilterListScreen(YMNScreen):
    """A screen for filtering a list column."""

    CSS = """
        FilterListScreen Label {
            width: auto;
            min-width: 12;
        }

        FilterListScreen .condition-row {
            width: auto;
            height: auto;
            min-width: 50;
        }

        FilterListScreen .condition-row Label {
            width: 10;
            margin: 1;
        }

        FilterListScreen .condition-row Input {
            width: auto;
            min-width: 40;
        }
    """

    def __init__(self, s: pl.Series, col_name: str, cursor_value: Any | None) -> None:
        """Initialize the filter list column screen.

        Args:
            s: The source series for the selected list column.
            col_name: The selected column name.
            cursor_value: The current cell value used as the default placeholder.
        """
        super().__init__(
            yes="Filter",
            no="Cancel",
            on_yes_callback=self._get_input,
        )
        self.s = s
        self.col_name = col_name
        self.cursor_value = cursor_value

    def compose(self) -> ComposeResult:
        """Compose the filter list column screen widget structure."""
        with Container(id="filter-list-column-container") as container:
            container.border_title = "Filter Rows by Column Value"
            yield Horizontal(
                Label("Equals to"),
                Input(
                    id="condition-eq",
                    tooltip="Enter a value to filter rows where the column is equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(
                Label("Not equal"),
                Input(
                    id="condition-neq",
                    tooltip="Enter a value to filter rows where the column is not equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(Label("Contains"), Input(id="condition-contains"), classes="condition-row")
            yield Horizontal(Label("Not contains"), Input(id="condition-not-contains"), classes="condition-row")
            yield from super().compose()

    def _convert_list_item(self, value: str) -> Any:
        """Convert user input to the list item dtype when possible.

        Args:
            value: User-provided text value.

        Returns:
            The converted value, or the original string if no stronger conversion applies.
        """
        inner_dtype = getattr(self.s.dtype, "inner", None)
        if inner_dtype is None:
            return value

        try:
            if inner_dtype == pl.Date:
                return pl.Series([value]).str.to_date().item()
            if inner_dtype == pl.Time:
                return pl.Series([value]).str.to_time().item()
            if inner_dtype == pl.Datetime:
                return pl.Series([value]).str.to_datetime().item()
            return DtypeConfig(inner_dtype).convert(value)
        except Exception:
            return value

    def _parse_list_literal(self, value: str) -> Any:
        """Parse a bracketed list literal for exact-list comparisons.

        Args:
            value: User-provided string expected to look like a Python list.

        Returns:
            Parsed list value, or the original string if parsing fails.
        """
        try:
            parsed_value = eval(value)
        except Exception:
            return value

        return parsed_value if isinstance(parsed_value, list) else value

    def _get_input(self) -> tuple[pl.Expr | None, str]:
        """Build and return a Polars filter expression from the list condition inputs.

        Returns:
            A tuple of (filter_expression, col_name), where filter_expression is None
            if no conditions were entered.
        """
        col = self.col_name
        expr: pl.Expr | None = None

        eq = self.query_one("#condition-eq", Input).value.strip()
        if eq:
            if eq == NULL:
                expr = pl.col(col).is_null()
            elif eq.startswith("[") and eq.endswith("]"):
                expr = pl.col(col) == self._parse_list_literal(eq)
            else:
                expr = pl.col(col).list.contains(self._convert_list_item(eq))
        else:
            neq = self.query_one("#condition-neq", Input).value.strip()
            if neq:
                if neq == NULL:
                    e = pl.col(col).is_not_null()
                elif neq.startswith("[") and neq.endswith("]"):
                    e = pl.col(col) != self._parse_list_literal(neq)
                else:
                    e = ~pl.col(col).list.contains(self._convert_list_item(neq))
                expr = e if expr is None else expr & e

            contains = self.query_one("#condition-contains", Input).value.strip()
            if contains:
                e = pl.col(col).list.contains(self._convert_list_item(contains))
                expr = e if expr is None else expr & e

            not_contains = self.query_one("#condition-not-contains", Input).value.strip()
            if not_contains:
                e = ~pl.col(col).list.contains(self._convert_list_item(not_contains))
                expr = e if expr is None else expr & e

        return expr, self.col_name


class FilterStringScreen(YMNScreen):
    """A screen for filtering a string column."""

    CSS = """
        FilterStringScreen Label {
            width: auto;
            min-width: 12;
        }

        FilterStringScreen Label#match-options-label {
            margin: 1 0 0 0;
        }

        FilterStringScreen .condition-row {
            width: auto;
            height: auto;
            min-width: 50;
        }

        FilterStringScreen .condition-row Label {
            width: 2;
            margin: 1;
        }

        FilterStringScreen .condition-row Input {
            width: auto;
            min-width: 48;
        }

        FilterStringScreen #checkbox-container {
            margin: 0 0 1 0;
            height: auto;
            width: auto;
        }

        FilterStringScreen Checkbox {
            margin: 0;
        }

        FilterStringScreen Checkbox:blur {
            border: solid $secondary;
        }
    """

    def __init__(self, s: pl.Series, col_name: str, cursor_value: int | float | None) -> None:
        """Initialize the filter string column screen."""
        super().__init__(
            yes="Filter",
            no="Cancel",
            on_yes_callback=self._get_input,
        )
        self.s = s
        self.col_name = col_name
        self.placeholder = NULL if cursor_value is None else str(cursor_value)

    def compose(self) -> ComposeResult:
        """Compose the filter string column screen widget structure."""
        with Container(id="filter-string-column-container") as container:
            container.border_title = "Filter Rows by Column Value"
            yield Horizontal(
                Label("Equals to"),
                Input(
                    placeholder=self.placeholder,
                    id="condition-eq",
                    tooltip="Enter a value to filter rows where the column is equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(
                Label("Not equal to"),
                Input(
                    placeholder=self.placeholder,
                    id="condition-neq",
                    tooltip="Enter a value to filter rows where the column is not equal to that value. Use NULL to filter null values.",
                ),
                classes="condition-row",
            )
            yield Horizontal(
                Label("Contains"),
                Input(placeholder=self.placeholder, id="condition-contains"),
                classes="condition-row",
            )
            yield Horizontal(
                Label("Starts with"),
                Input(placeholder=self.placeholder, id="condition-startswith"),
                classes="condition-row",
            )
            yield Horizontal(
                Label("Ends with"),
                Input(placeholder=self.placeholder, id="condition-endswith"),
                classes="condition-row",
            )
            yield Horizontal(
                Label("Regex"),
                Input(placeholder=self.placeholder, id="condition-regex"),
                classes="condition-row",
            )
            yield Label("Match options:", id="match-options-label")
            with Horizontal(id="checkbox-container"):
                yield Checkbox("Nocase", id="checkbox-nocase", tooltip="Ignore letter case when matching")
                yield Checkbox("Literal", id="checkbox-literal", tooltip="Treat input as plain text instead of regex")
                yield Checkbox("Reverse", id="checkbox-reverse", tooltip="Invert the match result")
            yield from super().compose()

    def _get_input(self) -> tuple[pl.Expr | None, str]:
        """Build and return a Polars filter expression from the string condition inputs.

        Returns:
            A tuple of (filter_expression, col_name), where filter_expression is None
            if no conditions were entered.
        """
        col = self.col_name
        expr: pl.Expr | None = None

        match_nocase = self.query_one("#checkbox-nocase", Checkbox).value
        match_literal = self.query_one("#checkbox-literal", Checkbox).value
        match_reverse = self.query_one("#checkbox-reverse", Checkbox).value

        eq = self.query_one("#condition-eq", Input).value
        if eq:
            if match_nocase:
                eq = f"(?i)^{eq}$"
            expr = (
                pl.col(col).is_null()
                if eq == NULL
                else pl.col(col).str.contains(eq, literal=match_literal)
                if match_nocase
                else pl.col(col) == eq
            )
        else:
            neq = self.query_one("#condition-neq", Input).value
            if neq:
                if match_nocase:
                    neq = f"(?i)^{neq}$"
                e = (
                    pl.col(col).is_not_null()
                    if neq == NULL
                    else ~pl.col(col).str.contains(neq, literal=match_literal)
                    if match_nocase
                    else pl.col(col) != neq
                )
                expr = e if expr is None else expr & e

            contains = self.query_one("#condition-contains", Input).value
            if contains:
                if match_nocase:
                    contains = f"(?i){contains}"
                e = pl.col(col).str.contains(contains, literal=match_literal)
                expr = e if expr is None else expr & e

            startswith = self.query_one("#condition-startswith", Input).value
            if startswith:
                startswith = f"^{startswith}"
                if match_nocase:
                    startswith = f"(?i){startswith}"
                e = pl.col(col).str.contains(startswith, literal=match_literal)
                expr = e if expr is None else expr & e

            endswith = self.query_one("#condition-endswith", Input).value
            if endswith:
                endswith = f"{endswith}$"
                if match_nocase:
                    endswith = f"(?i){endswith}"
                e = pl.col(col).str.contains(endswith, literal=match_literal)
                expr = e if expr is None else expr & e

            regex = self.query_one("#condition-regex", Input).value
            if regex:
                e = pl.col(col).str.contains(regex, literal=match_literal)
                expr = e if expr is None else expr & e

        if match_reverse and expr is not None:
            expr = ~expr

        return expr, self.col_name


class FilterBooleanScreen(YMNScreen):
    """A screen for filtering a boolean column."""

    CSS = """
        FilterBooleanScreen > #filter-boolean-column-container {
            width: auto;
            height: auto;
            max-width: 50;
        }

        FilterBooleanScreen #radio-container {
            height: auto;
        }

        FilterBooleanScreen #boolean-radio-set {
            height: auto;
        }
    """

    def __init__(self, s: pl.Series, col_name: str, cursor_value: bool | None) -> None:
        """Initialize the filter boolean column screen."""
        super().__init__(
            yes="Filter",
            no="Cancel",
            on_yes_callback=self._get_input,
        )
        self.s = s
        self.col_name = col_name
        self.cursor_value = cursor_value

    def compose(self) -> ComposeResult:
        """Compose the filter boolean column screen widget structure."""
        has_null = self.s.null_count() > 0

        with Container(id="filter-boolean-column-container") as container:
            container.border_title = "Filter Rows by Column Value"
            with RadioSet(id="boolean-radio-set"):
                yield RadioButton("True", id="radio-true", value=self.cursor_value is True)
                yield RadioButton("False", id="radio-false", value=self.cursor_value is False)
                if has_null:
                    yield RadioButton(NULL, id="radio-null", value=self.cursor_value is None)
            yield from super().compose()

    def _get_input(self) -> tuple[pl.Expr | None, str]:
        """Build and return a Polars filter expression from the selected radio button.

        Returns:
            A tuple of (filter_expression, col_name), where filter_expression is None
            if no radio button is selected.
        """
        col = self.col_name
        radio_set = self.query_one("#boolean-radio-set", RadioSet)

        pressed = radio_set.pressed_button
        if pressed is None:
            return None, self.col_name

        # RadioButton.value is the checked state, not a payload; map from id.
        if pressed.id == "radio-true":
            selected_value = True
        elif pressed.id == "radio-false":
            selected_value = False
        elif pressed.id == "radio-null":
            selected_value = NULL
        else:
            return None, self.col_name

        if selected_value == NULL:
            expr = pl.col(col).is_null()
        else:
            expr = pl.col(col) == selected_value

        return expr, self.col_name


class JoinTableScreen(YMNScreen):
    """A screen for joining two tables from the current app.

    Provides two vertical panels (left and right) each containing an OptionList
    to select a table and a SelectionList of columns that updates dynamically
    based on the selected table. Also includes a join type selector.
    """

    JOIN_TYPES = {
        "join-inner": "inner",
        "join-left": "left",
        "join-right": "right",
        "join-full": "full",
        "join-semi": "semi",
        "join-anti": "anti",
    }

    # fmt: off
    CSS = YMNScreen.DEFAULT_CSS.replace("YMNScreen", "JoinTableScreen") + """
        JoinTableScreen > Container {
            min-width: 64;
            max-width: 80;
            max-height: 80%;
            padding: 1;
        }

        JoinTableScreen #join-panels {
            height: auto;
            max-height: 16;
            margin-bottom: 1;
        }

        JoinTableScreen .join-panel {
            width: 1fr;
            height: auto;
            padding: 0 1;
        }

        JoinTableScreen Select {
            width: 100%;
            margin: 0 0 1 0;
        }

        JoinTableScreen SelectionList {
            height: auto;
            max-height: 10;
            margin: 0 0 1 0;
        }

        JoinTableScreen SelectionList:blur {
            border: solid $secondary;
        }

        JoinTableScreen Label {
            margin: 0;
            width: 100%;
        }

        JoinTableScreen #join-type-set {
            layout: horizontal;
            height: auto;
            margin: 0 0 1 0;
        }

        JoinTableScreen RadioButton {
            margin: 0 1 0 0;
        }

        JoinTableScreen #button-container {
            margin: 1 0 0 0;
        }

        JoinTableScreen Button {
            height: 3;
            margin: 0 2;
        }
    """
    # fmt: on

    def __init__(self, left: "DataFrameTable | None" = None, right: "DataFrameTable | None" = None) -> None:
        """Initialize the join table screen.

        Args:
            left: Optional DataFrameTable to pre-select as the left table.
                  Defaults to the active tab's DataFrameTable.
            right: Optional DataFrameTable to pre-select as the right table.
                   Defaults to the first table that is not the left table.
        """
        super().__init__(
            yes="Join",
            no="Cancel",
            on_yes_callback=self._join_two_tables,
        )

        # Build table name -> DataFrameTable mapping from app tabs
        self.dftables: dict[str, "DataFrameTable"] = {}
        for dftable in self.app.tabs.values():
            self.dftables[dftable.tabname] = dftable

        # Store left/right DataFrameTable selections
        self.left: "DataFrameTable" = left if left is not None else self.app.active_table

        if right is not None:
            self.right: "DataFrameTable" | None = right
        else:
            self.right = None
            for dftable in self.app.tabs.values():
                if dftable is not self.left:
                    self.right = dftable
                    break
            else:
                self.right = self.left

    def compose(self) -> ComposeResult:
        """Compose the join table screen widget structure.

        Creates two vertical panels (left and right), each with an OptionList
        for table selection and a SelectionList for column selection.
        Also includes a RadioSet for join type.

        Yields:
            Widget: The components of the join table screen.
        """
        table_names = list(self.dftables.keys())
        table_options = [(name, name) for name in table_names]
        left_default = self.left.tabname if self.left else Select.BLANK
        right_default = self.right.tabname if self.right else Select.BLANK

        with Container(id="join-table-container") as container:
            container.border_title = "Join Tables"

            with Horizontal(id="join-panels"):
                with Vertical(classes="join-panel"):
                    yield Label("Left table:")
                    yield Select(table_options, value=left_default, id="left-table-selection")
                    yield Label("Left keys:")
                    yield SelectionList(id="left-key-selection")

                with Vertical(classes="join-panel"):
                    yield Label("Right table:")
                    yield Select(table_options, value=right_default, id="right-table-selection")
                    yield Label("Right keys:")
                    yield SelectionList(id="right-key-selection")

            yield Label("Join type:")
            with RadioSet(id="join-type-set"):
                yield RadioButton("Inner", id="join-inner", value=True)
                yield RadioButton("Left", id="join-left")
                yield RadioButton("Right", id="join-right")
                yield RadioButton("Full", id="join-full")
                yield RadioButton("Semi", id="join-semi")
                yield RadioButton("Anti", id="join-anti")

            yield from super().compose()

    def on_mount(self) -> None:
        """Initialize column lists based on pre-selected left/right tables."""
        left_select = self.query_one("#left-table-selection", Select)
        right_select = self.query_one("#right-table-selection", Select)

        if left_select.value != Select.BLANK:
            self._update_columns("left", str(left_select.value))
        if right_select.value != Select.BLANK:
            self._update_columns("right", str(right_select.value))

    def _update_columns(self, side: str, table_name: str) -> None:
        """Update the SelectionList for the given side based on the selected table.

        Args:
            side: Either "left" or "right" to indicate which panel to update.
            table_name: The name of the selected table.
        """
        selection_list = self.query_one(f"#{side}-key-selection", SelectionList)
        dftable = self.dftables.get(table_name)
        if dftable is None:
            return

        selection_list.clear_options()

        for col in dftable.df.columns:
            if col == RID:
                continue
            selection_list.add_option(Selection(col, col, initial_state=False))

    @on(Select.Changed, "#left-table-selection")
    def _on_left_table_changed(self, event: Select.Changed) -> None:
        """Update left column list when a different table is selected.

        Args:
            event: The select changed event.
        """
        if event.value != Select.BLANK:
            self._update_columns("left", str(event.value))

    @on(Select.Changed, "#right-table-selection")
    def _on_right_table_changed(self, event: Select.Changed) -> None:
        """Update right column list when a different table is selected.

        Args:
            event: The select changed event.
        """
        if event.value != Select.BLANK:
            self._update_columns("right", str(event.value))

    def _get_selected_table(self, side: str) -> "DataFrameTable | None":
        """Get the DataFrameTable for the currently selected table on the given side.

        Args:
            side: Either "left" or "right".

        Returns:
            The DataFrameTable for the selected table, or None if not found.
        """
        select = self.query_one(f"#{side}-table-selection", Select)
        if select.value == Select.BLANK:
            return None
        return self.dftables.get(str(select.value))

    def _get_selected_columns(self, side: str) -> list[str]:
        """Get the selected column names from the SelectionList on the given side.

        Args:
            side: Either "left" or "right".

        Returns:
            A list of selected column names.
        """
        selection_list = self.query_one(f"#{side}-key-selection", SelectionList)
        return list(selection_list.selected)

    def _get_join_type(self) -> str:
        """Get the selected join type from the RadioSet.

        Returns:
            The Polars join type string (e.g., "inner", "left", "full").
        """
        radio_set = self.query_one("#join-type-set", RadioSet)
        pressed = radio_set.pressed_button
        if pressed and pressed.id:
            return self.JOIN_TYPES.get(pressed.id, "inner")
        return "inner"

    def _join_two_tables(self) -> pl.DataFrame | None:
        """Perform the join operation on the two selected tables.

        Returns:
            The joined DataFrame, or None if validation fails.
        """
        left_table = self._get_selected_table("left")
        right_table = self._get_selected_table("right")

        if left_table is None or right_table is None:
            self.notify("Please select both left and right tables.", severity="error")
            return None

        left_keys = self._get_selected_columns("left")
        right_keys = self._get_selected_columns("right")

        if not left_keys or not right_keys:
            self.notify("Please select at least one key column on each side.", severity="error")
            return None

        if len(left_keys) != len(right_keys):
            self.notify(
                f"Key column count mismatch: left has {len(left_keys)}, right has {len(right_keys)}.",
                severity="error",
            )
            return None

        join_type = self._get_join_type()

        left_df = left_table.df.select([c for c in left_table.df.columns if c != RID])
        right_df = right_table.df.select([c for c in right_table.df.columns if c != RID])

        try:
            result: pl.DataFrame = left_df.join(
                right_df,
                left_on=left_keys,
                right_on=right_keys,
                how=join_type,
            )
        except Exception as e:
            self.notify(f"Join failed: {e}", severity="error")
            return None

        return result
