# DataFrame Viewer

A fast, interactive terminal-based CSV viewer built with Python, Polars, and Rich. Inspired by VisiData, this tool provides smooth keyboard navigation and a clean interface for exploring CSV data directly in your terminal.

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

# Choose a different box style
python main.py pokemon.csv --box rounded
python main.py pokemon.csv --box heavy
python main.py pokemon.csv --box minimal

# Or with uv
uv run python main.py pokemon.csv
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
| `Enter` | Page down (next page) |
| `PageDown` / `Ctrl+F` | Next page |
| `PageUp` / `Ctrl+B` | Previous page |
| `Home` | Jump to first page |
| `End` | Jump to last page |
| `q` | Quit viewer |

**Note:** Pressing `Enter` on the last page will exit the viewer.

## Features in Detail

### Box Styles

Customize the appearance of table borders with the `--box` option. The viewer accepts any box style name from the Rich library (case-insensitive):

**Popular styles:**
- **simple** (default): Minimal single-line borders
- **rounded**: Smooth rounded corners
- **minimal**: Very subtle borders
- **heavy**: Bold, thick borders
- **double**: Double-line borders
- **square**: Sharp square corners
- **ascii**: ASCII-only characters (for maximum compatibility)
- **minimal_double_head**: Minimal with double-line header
- **heavy_head**: Heavy border with emphasized header

Examples:
```bash
python main.py data.csv --box rounded
python main.py data.csv --box minimal_double_head
python main.py data.csv --box heavy_edge
```

To see all available box styles, run with an invalid style name:
```bash
python main.py data.csv --box help
```

### Color-Coded Columns

Columns are automatically styled based on their data type:
- **Integers** (Int64, Int32, UInt32): Bold cyan, right-aligned
- **Floats** (Float64, Float32): Magenta, right-aligned
- **Strings** (Utf8): Green, left-aligned
- **Booleans**: Yellow, center-aligned
- **Dates/Datetime**: Blue, center-aligned

### Status Bar

The bottom status bar shows:
- **Left**: Filename or "stdin"
- **Right**: Current row range and total rows (e.g., "rows 1-50 / 163")

### Smart Pagination

- Automatically calculates optimal rows per page based on terminal height
- Last page may show fewer rows to avoid overlap
- Smooth navigation with no duplicate rows between pages

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
