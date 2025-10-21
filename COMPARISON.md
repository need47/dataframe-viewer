# CSV Viewer Comparison: Rich vs Textual

## Overview

This project now has two implementations:
- **`main_rich.py`**: Custom implementation using Rich library
- **`main.py`**: Implementation using Textual framework (default)

## Comparison

### Rich Implementation (`main_rich.py`)

**Pros:**
- Full control over every aspect of rendering
- Smaller dependency footprint
- Custom keyboard handling gives precise control
- Learned low-level terminal concepts (raw mode, escape sequences, etc.)
- More lightweight and faster startup

**Cons:**
- ~290 lines of code
- Manual keyboard event handling required
- Had to implement pagination logic manually
- Terminal mode handling (raw mode, /dev/tty) is complex
- More code to maintain

**Key Features:**
- Custom box styles (simple, rounded, minimal, etc.)
- Manual pagination with calculated page sizes
- Custom status bar with filename and row count
- Handles stdin and file input
- Raw terminal input handling

### Textual Implementation (`main.py`)

**Pros:**
- **Much simpler**: ~100 lines vs ~290 lines
- Built-in keyboard navigation (arrows, page up/down work automatically)
- Built-in widgets (DataTable, Header, Footer)
- Automatic scrolling and virtualization for large datasets
- Better abstraction - focuses on "what" not "how"
- CSS-like styling system
- More maintainable and easier to extend

**Cons:**
- Larger dependency (Textual brings more libraries)
- Less control over fine details
- Slightly heavier (~6 additional packages)
- Learning curve for Textual-specific concepts

**Key Features:**
- Built-in DataTable widget with automatic scrolling
- Header and Footer widgets
- Keyboard bindings system (`BINDINGS`)
- Automatic cursor management
- Row highlighting

## Code Size Comparison

| Aspect | Rich | Textual |
|--------|------|---------|
| Lines of code | ~290 | ~100 |
| Keyboard handling | Manual (50+ lines) | Automatic (5 lines for bindings) |
| Table rendering | Manual (50+ lines) | Built-in widget (10 lines) |
| Navigation logic | Manual (40+ lines) | Built-in (2 action methods) |
| Status bar | Custom (20 lines) | Built-in Footer widget |

## Which to Use?

**Use Rich (`main_rich.py`) if you want:**
- Learning experience with low-level terminal programming
- Maximum control over rendering
- Minimal dependencies
- Custom styling options (box styles)
- Slightly faster startup

**Use Textual (`main.py`) if you want:**
- Rapid development
- Less code to maintain
- Built-in features (scrolling, highlighting, etc.)
- Better defaults out of the box
- Easier to extend with more features

## Running

```bash
# Textual version (default)
uv run python main.py pokemon.csv
cat pokemon.csv | uv run python main.py

# Rich version
uv run python main_rich.py pokemon.csv
uv run python main_rich.py pokemon.csv --box rounded
```

## Conclusion

Both implementations work well! The Rich version taught us a lot about terminal programming, while the Textual version shows how a higher-level framework can dramatically reduce code complexity.

**Textual is probably better for production use** due to its simplicity, maintainability, and built-in features. The Rich version is excellent for learning and when you need fine-grained control.
