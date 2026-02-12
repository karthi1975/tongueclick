# Mouse Automation with Tongue Click Control 🖱️👅

Automate mouse movements and clicks using Python. Includes integration with tongue click detection for hands-free control!

## Features

### 🖱️ Mouse Automation (`mouse_automation.py`)
- **Random Clicking**: Click random positions on screen
- **Position Lists**: Click specific windows/buttons in random order
- **Pattern Drawing**: Draw shapes (square, circle, zigzag, star)
- **Position Finder**: Interactive tool to record window coordinates
- **Configurable Delays**: Control timing between actions
- **Safety Mode**: Move mouse to corner to abort

### 👅 Tongue Click Integration (`tongue_click_mouse_integration.py`)
- **Current Position Mode**: Click wherever your mouse is pointing
- **Random Position Mode**: Click random screen locations
- **Position Cycle Mode**: Cycle through predefined positions
- **Random List Mode**: Randomly select from position list
- **Hands-Free Control**: Trigger clicks with tongue sounds!

## Quick Start

### 1. Test Installation

```bash
python test_mouse.py
```

Should show:
```
✓ ALL TESTS PASSED
Mouse automation is ready to use!
```

### 2. Find Window Positions (Interactive)

```bash
python mouse_automation.py --demo finder
```

This will help you:
1. Move mouse to windows you want to click
2. Press Enter to record each position
3. Get code you can copy/paste

### 3. Try Mouse Automation Examples

**Click 5 random positions:**
```bash
python mouse_automation.py --demo basic
```

**Draw patterns:**
```bash
python mouse_automation.py --demo patterns
```

### 4. Try Tongue Click Control

**Click at current mouse position:**
```bash
python tongue_click_mouse_integration.py --mode current --duration 15
```

**Click random positions:**
```bash
python tongue_click_mouse_integration.py --mode random --duration 15
```

## Usage Guide

### Mouse Automation Only

```python
from mouse_automation import MouseAutomation

# Initialize
auto = MouseAutomation(safe_mode=True, delay=0.5)

# Click 5 random positions
auto.click_random_positions(5)

# Click specific windows in random order
windows = [
    {'name': 'Browser', 'x': 300, 'y': 100},
    {'name': 'Terminal', 'x': 700, 'y': 400},
    {'name': 'Editor', 'x': 900, 'y': 400},
]
auto.click_windows_random_order(windows)

# Draw a pattern
auto.draw_pattern('circle', size=200)
```

### Tongue Click + Mouse Integration

```python
from tongue_click_mouse_integration import TongueClickMouseControl

# Initialize
controller = TongueClickMouseControl(threshold=0.3, click_delay=0.5)

# Mode 1: Click at current position
controller.start_current_position_mode(duration=30)

# Mode 2: Click random positions
controller.start_random_position_mode(duration=30)

# Mode 3: Cycle through positions
positions = [(400, 300), (800, 300), (400, 600)]
controller.start_position_cycle_mode(positions, duration=30)

# Mode 4: Random from list
controller.start_random_from_list_mode(positions, duration=30)
```

## Command-Line Reference

### Mouse Automation

```bash
# Run all demos
python mouse_automation.py --demo all

# Interactive position finder
python mouse_automation.py --demo finder

# Basic clicking demo
python mouse_automation.py --demo basic

# Pattern drawing demo
python mouse_automation.py --demo patterns

# Click specific windows demo
python mouse_automation.py --demo windows
```

### Tongue Click Integration

```bash
# Click at current mouse position
python tongue_click_mouse_integration.py --mode current --duration 15

# Click random positions
python tongue_click_mouse_integration.py --mode random --duration 15

# Cycle through predefined positions
python tongue_click_mouse_integration.py --mode cycle --duration 20

# Random from predefined list
python tongue_click_mouse_integration.py --mode random-list --duration 20

# Adjust sensitivity
python tongue_click_mouse_integration.py --mode current --threshold 0.2 --duration 20
```

## Use Cases

### 1. Automated Testing
```python
# Click through UI elements in random order for testing
auto = MouseAutomation()

test_buttons = [
    {'name': 'Submit', 'x': 500, 'y': 400},
    {'name': 'Cancel', 'x': 600, 'y': 400},
    {'name': 'Reset', 'x': 700, 'y': 400},
]

auto.click_windows_random_order(test_buttons)
```

### 2. Accessibility Tool
```python
# Hands-free clicking with tongue clicks
controller = TongueClickMouseControl(threshold=0.3)

# Move mouse with trackball/eye-tracking
# Click with tongue clicks
controller.start_current_position_mode(duration=60)
```

### 3. Game Automation
```python
# Click through game menus in random order
positions = [
    (640, 300),  # Play button
    (640, 400),  # Settings
    (640, 500),  # Quit
]

auto = MouseAutomation(delay=1.0)
auto.click_positions_random_order(positions)
```

### 4. Presentation Control
```python
# Advance slides with tongue clicks
controller = TongueClickMouseControl(threshold=0.3)

# Position mouse on "Next" button
# Each tongue click advances the slide
controller.start_current_position_mode(duration=300)
```

## Safety Features

### Fail-Safe Corner
- Move mouse to **top-left corner** to immediately abort
- Enabled by default with `safe_mode=True`
- Prevents runaway automation

### Delays
- Configurable delays between clicks prevent overwhelming systems
- Default: 0.5 seconds between actions
- Tongue click mode: 0.5 second minimum between clicks

### Visual Feedback
- All actions print to console
- Shows coordinates and action type
- Click counts displayed

## Finding Window Positions

### Method 1: Interactive Finder (Recommended)

```bash
python mouse_automation.py --demo finder
```

1. Run the command
2. Move mouse to desired window
3. Press Enter to record
4. Repeat for all windows
5. Type 'done' when finished
6. Copy the generated code

### Method 2: Get Current Position

```python
from mouse_automation import MouseAutomation

auto = MouseAutomation()
x, y = auto.get_current_position()
# Move mouse where you want, then run this
```

### Method 3: PyAutoGUI Display

```bash
python -m pyautogui.displayMousePosition
```

Shows live coordinates as you move the mouse.

## API Reference

### MouseAutomation Class

#### `__init__(safe_mode=True, delay=0.5)`
Initialize mouse automation.

#### `click_at(x, y, clicks=1, button='left')`
Click at specific coordinates.

#### `click_random_positions(num_clicks, region=None)`
Click random positions. Optional region to limit area.

#### `click_positions_random_order(positions)`
Click list of (x, y) tuples in random order.

#### `click_windows_random_order(window_positions)`
Click windows defined as dicts with 'name', 'x', 'y'.

#### `draw_pattern(pattern='square', size=200)`
Draw patterns: 'square', 'circle', 'zigzag', 'star'.

#### `get_current_position()`
Get current mouse (x, y) coordinates.

#### `screenshot(filename=None)`
Take a screenshot.

### TongueClickMouseControl Class

#### `__init__(threshold=0.3, click_delay=0.5)`
Initialize tongue click mouse control.

#### `start_current_position_mode(duration=30)`
Click at current mouse position on each tongue click.

#### `start_random_position_mode(duration=30)`
Click random positions on each tongue click.

#### `start_position_cycle_mode(positions, duration=30)`
Cycle through positions on each tongue click.

#### `start_random_from_list_mode(positions, duration=30)`
Click random position from list on each tongue click.

## Tips & Best Practices

### 1. Test with Short Durations
- Start with 10-15 seconds
- Increase once you're comfortable
- Always know where the abort corner is!

### 2. Adjust Sensitivity
- **Too many false clicks**: Increase threshold (0.4)
- **Missing real clicks**: Decrease threshold (0.2)
- Default 0.3 works for most cases

### 3. Position Recording
- Use the interactive finder tool
- Record positions with window maximized
- Test positions before long automation runs

### 4. Delays Matter
- Slow systems: Increase delay (1.0+ seconds)
- Fast systems: Can use 0.3-0.5 seconds
- Web apps: Add extra delay for loading

### 5. Safety First
- Always enable safe_mode
- Test in safe environment first
- Don't automate irreversible actions without supervision

## Troubleshooting

### Issue: Clicks not registering
- **Check coordinates**: Use position finder
- **Increase delay**: Give apps time to respond
- **Window focus**: Ensure window is active

### Issue: Too many false tongue clicks
- **Increase threshold**: Use 0.4 or 0.5
- **Check microphone**: Reduce background noise
- **Increase click_delay**: Prevent rapid clicking

### Issue: Mouse moves too fast
- **Increase duration**: In `move_to()` calls
- **Add delays**: Between movements
- **Slow patterns**: Use larger delay values

### Issue: Can't find pyautogui
- **Install**: `pip install pyautogui`
- **Use venv**: `source venv/bin/activate`
- **Verify**: `python test_mouse.py`

## Platform Notes

### macOS
- May need to grant Accessibility permissions
- System Preferences → Security & Privacy → Accessibility
- Add Terminal or Python to allowed apps

### Linux
- Install dependencies: `sudo apt-get install python3-tk python3-dev`
- May need xdotool: `sudo apt-get install xdotool`

### Windows
- Should work out of the box
- May need to run as Administrator for some apps

## Examples

### Example 1: Automated Form Filling

```python
auto = MouseAutomation(delay=0.5)

fields = [
    {'name': 'Name field', 'x': 400, 'y': 200},
    {'name': 'Email field', 'x': 400, 'y': 250},
    {'name': 'Phone field', 'x': 400, 'y': 300},
    {'name': 'Submit button', 'x': 500, 'y': 400},
]

# Click each field
for field in fields:
    auto.click_at(field['x'], field['y'])
    time.sleep(0.3)
    # Type something (you'd add this)
```

### Example 2: Hands-Free Drawing

```python
# Use tongue clicks to draw
controller = TongueClickMouseControl(threshold=0.3)

# Open a drawing app, select pen tool
# Move mouse with trackpad/mouse
# Click to draw with tongue clicks
controller.start_current_position_mode(duration=120)
```

### Example 3: Random UI Stress Test

```python
auto = MouseAutomation(delay=0.2)

# Click 100 random positions in a region
auto.click_random_positions(
    num_clicks=100,
    region=(0, 0, 1920, 1080)  # Full HD screen
)
```

## Dependencies

- `pyautogui` - Mouse control and automation
- `numpy` - Already installed for tongue click detection
- Platform-specific (auto-installed):
  - macOS: pyobjc-core, pyobjc-framework-Quartz
  - Linux: python3-xlib
  - Windows: No additional requirements

## Credits

Built using:
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse automation
- Integrated with tongue click detection system
- Cross-platform compatibility

## License

This project is provided as-is for educational and automation purposes.

---

**Ready to automate!** 🖱️🚀
