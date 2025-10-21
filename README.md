# DataFrame Viewer

A fast, interactive terminal-based CSV viewer built with Python, Polars, and Textual. Inspired by VisiData, this tool provides smooth keyboard navigation and a clean interface for exploring CSV data directly in your terminal.

> **Note:** This is the Textual version (simpler, ~100 lines). For the Rich library version with custom box styles, see `main_rich.py` and [COMPARISON.md](COMPARISON.md).

## Features

- 🚀 **Fast CSV Loading** - Powered by Polars for efficient data handling
- 🎨 **Rich Terminal UI** - Beautiful, color-coded columns with automatic type detection
- ⌨️ **Keyboard Navigation** - Intuitive controls for browsing large datasets
- 📊 **Flexible Input** - Read from files or stdin (pipes/redirects)
- 🔄 **Live Updates** - Smooth, flicker-free screen rendering
- 📄 **Page-based Display** - Shows ~50 rows per page with automatic terminal size detection

## Installation

### Using uv (recommended)

```bash
# Clone or download the project
cd dataframe-viewer

# Run directly with uv
uv run python main.py <csv_file>
```

### Using pip

```bash
pip install polars rich
python main.py <csv_file>
```

## Usage

### Basic Usage

```bash
# View a CSV file
python main.py pokemon.csv

# Or with uv
uv run python main.py pokemon.csv
```

### Using the Rich version (with box styles)

```bash
# Rich version with custom box styles
python main_rich.py pokemon.csv --box rounded
python main_rich.py pokemon.csv --box heavy
python main_rich.py pokemon.csv --box minimal
```

### Reading from stdin

```bash
# Pipe CSV data
cat data.csv | python main.py

# Redirect from file
python main.py < data.csv

# Chain with other commands
grep "Fire" pokemon.csv | python main.py
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `↑` / `↓` | Move up/down one row |
| `PageDown` | Next page |
| `PageUp` | Previous page |
| `Home` | Jump to first row |
| `End` | Jump to last row |
| `q` | Quit viewer |

## Features in Detail

### Color-Coded Columns

Columns are automatically styled based on their data type:
- **Integers** (Int64, Int32, UInt32): Cyan
- **Floats** (Float64, Float32): Magenta
- **Strings** (Utf8): Green
- **Booleans**: Yellow
- **Dates/Datetime**: Blue

### Automatic Scrolling

- Built-in scrolling with DataTable widget
- Smooth navigation with arrow keys
- Page up/down for quick navigation
- Home/End keys to jump to first/last row

## Examples

```bash
# View Pokemon dataset
uv run python main.py pokemon.csv

# Filter and view
grep "Legendary" pokemon.csv | uv run python main.py

# View with custom processing
cut -d',' -f1,2,3 pokemon.csv | uv run python main.py
```

## Dependencies

- **polars**: Fast DataFrame library for CSV processing
- **rich**: Terminal rendering and styling

## Technical Details

- Built with Rich's Live display for flicker-free updates
- Uses Polars for efficient CSV parsing and slicing
- Direct `/dev/tty` access for keyboard input when reading from stdin
- SIMPLE box style for minimal, clean table rendering
- Configurable padding and terminal size detection

## Requirements

- Python 3.8+
- POSIX-compatible terminal (macOS, Linux, WSL)
- Terminal supporting ANSI escape sequences

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Areas for enhancement:
- Additional keyboard shortcuts (search, filter, sort)
- Column width adjustment
- Export/save filtered views
- Support for other data formats (JSON, Parquet, Excel)
- Cross-platform key handling improvements

## Troubleshooting

**Issue**: Headers not showing
- Ensure your terminal supports ANSI colors and box-drawing characters

**Issue**: Navigation keys not working
- Try alternative keys: `Ctrl+B`/`Ctrl+F` for paging, `Enter` for next page

**Issue**: Can't read from pipe
- Make sure you have proper permissions to read from `/dev/tty`

## Acknowledgments

- Inspired by [VisiData](https://visidata.org/)
- Built with [Polars](https://www.pola.rs/) and [Rich](https://rich.readthedocs.io/)
- **All code created by GitHub Copilot agent** through iterative conversation and refinement
