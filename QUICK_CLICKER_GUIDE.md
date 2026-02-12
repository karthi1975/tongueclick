# Quick Guide: Random Window Clicker

One simple script that clicks Chrome, Adobe, and cursor position randomly! 🖱️

## Setup (First Time Only)

### Step 1: Open Your Applications
1. Open **Chrome browser**
2. Open **Adobe application** (Acrobat, Photoshop, etc.)
3. Position them where you can see them

### Step 2: Record Window Positions

```bash
# Activate virtual environment
source venv/bin/activate

# Run setup
python random_window_clicker.py --setup
```

Follow the prompts:
1. Move mouse to Chrome → Press Enter
2. Move mouse to Adobe → Press Enter
3. Positions saved automatically!

## Usage

### Basic - 20 Random Clicks (1-5 second delays)

```bash
python random_window_clicker.py
```

This will:
- ✓ Click Chrome randomly
- ✓ Click Adobe randomly
- ✓ Click current cursor position randomly
- ✓ Random delays 1-5 seconds between clicks
- ✓ Total 20 clicks

### Custom Number of Clicks

```bash
# 50 clicks
python random_window_clicker.py --clicks 50

# 100 clicks
python random_window_clicker.py --clicks 100
```

### Custom Time Delays

```bash
# 30 clicks with 2-10 second delays
python random_window_clicker.py --clicks 30 --min-delay 2 --max-delay 10

# 20 clicks with 0.5-2 second delays (faster)
python random_window_clicker.py --clicks 20 --min-delay 0.5 --max-delay 2
```

### Continuous Mode

```bash
# Run for 2 minutes
python random_window_clicker.py --continuous --duration 120

# Run for 10 minutes
python random_window_clicker.py --continuous --duration 600

# Run forever (until Ctrl+C)
python random_window_clicker.py --continuous --duration 0
```

### Only Click Chrome and Adobe (No Cursor)

```bash
python random_window_clicker.py --clicks 30 --no-cursor
```

## Examples

### Example 1: Quick Test
```bash
# 10 clicks, fast (1-2 seconds)
python random_window_clicker.py --clicks 10 --min-delay 1 --max-delay 2
```

### Example 2: Long Random Session
```bash
# 100 clicks with 3-8 second random delays
python random_window_clicker.py --clicks 100 --min-delay 3 --max-delay 8
```

### Example 3: Continuous Background Task
```bash
# Keep clicking for 30 minutes
python random_window_clicker.py --continuous --duration 1800
```

### Example 4: Only Application Windows
```bash
# 50 clicks, only Chrome/Adobe, no cursor clicks
python random_window_clicker.py --clicks 50 --no-cursor --min-delay 2 --max-delay 6
```

## What It Does

The script will:
1. **Randomly select** Chrome, Adobe, or cursor position
2. **Click** the selected target
3. **Wait** a random time (between min-delay and max-delay)
4. **Repeat** until complete

### Random Selection Logic:
- 50% chance: Click one of the saved windows (Chrome or Adobe)
- 50% chance: Click at current cursor position (if enabled)
- Which window: Randomly chosen each time

### Random Timing:
- Each click has different delay
- Unpredictable pattern
- Looks more "human"

## Quick Reference

```bash
# First time setup
python random_window_clicker.py --setup

# Default (20 clicks, 1-5s delays)
python random_window_clicker.py

# Custom clicks
python random_window_clicker.py --clicks 50

# Custom delays
python random_window_clicker.py --min-delay 2 --max-delay 10

# Continuous 5 minutes
python random_window_clicker.py --continuous --duration 300

# No cursor clicks
python random_window_clicker.py --no-cursor

# All options combined
python random_window_clicker.py --clicks 100 --min-delay 1 --max-delay 3 --no-cursor
```

## Safety

### Abort Anytime:
- **Move mouse to top-left corner** = Instant abort
- **Ctrl+C** = Stop gracefully

### Before Running:
- Save your work!
- Test with small number first (--clicks 5)
- Make sure windows are visible

## Troubleshooting

### "window_positions.json not found"
**Solution**: Run `python random_window_clicker.py --setup` first

### Wrong positions
**Solution**: Run setup again to re-record:
```bash
python random_window_clicker.py --setup
```

### Too fast/slow
**Solution**: Adjust delays:
```bash
# Slower
python random_window_clicker.py --min-delay 5 --max-delay 10

# Faster
python random_window_clicker.py --min-delay 0.5 --max-delay 2
```

### Clicks wrong spot
**Solution**:
1. Make sure Chrome/Adobe are in same position as during setup
2. Or re-run setup if windows moved

## Output Example

```
============================================================
RANDOM WINDOW CLICKING STARTED
============================================================
Total clicks: 20
Delay range: 1.0-5.0 seconds
Include cursor position: True

Press Ctrl+C to stop early
Move mouse to corner to abort
============================================================

Starting in 3 seconds...

[1/20] Clicking: Chrome Browser at (500, 200)
  → Waiting 3.2 seconds...

[2/20] Clicking: Current Cursor at (800, 400)
  → Waiting 1.8 seconds...

[3/20] Clicking: Adobe Application at (1200, 300)
  → Waiting 4.5 seconds...

...

============================================================
✓ Completed 20 clicks!
============================================================
```

## Full Command Reference

```bash
python random_window_clicker.py [OPTIONS]

Options:
  --setup              Record window positions (first time)
  --clicks N           Number of clicks (default: 20)
  --min-delay N        Min seconds between clicks (default: 1.0)
  --max-delay N        Max seconds between clicks (default: 5.0)
  --no-cursor          Don't click cursor position
  --continuous         Continuous mode
  --duration N         Duration for continuous (0=forever, default: 60)
  --config FILE        Custom config file (default: window_positions.json)
```

---

**That's it! Simple random clicking for Chrome, Adobe, and cursor position!** 🎯
