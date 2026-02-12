"""
Mouse Automation Script

Automates mouse movements and clicks on windows in random order.
Can be used for testing, automation, or integrated with tongue click detection.
"""

import pyautogui
import time
import random
from typing import List, Tuple, Optional
import platform


class MouseAutomation:
    """
    Automates mouse movements and clicks with various patterns.
    """

    def __init__(self, safe_mode=True, delay=0.5):
        """
        Initialize mouse automation.

        Args:
            safe_mode (bool): Enable safety features (fail-safe corner)
            delay (float): Delay between actions in seconds
        """
        self.safe_mode = safe_mode
        self.delay = delay

        # Enable PyAutoGUI fail-safe (move to corner to abort)
        pyautogui.FAILSAFE = safe_mode

        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        print(f"Screen size: {self.screen_width}x{self.screen_height}")
        if safe_mode:
            print("Safety mode ON: Move mouse to corner to abort")

    def move_to(self, x: int, y: int, duration: float = 0.5):
        """
        Move mouse to specific coordinates.

        Args:
            x (int): X coordinate
            y (int): Y coordinate
            duration (float): Movement duration in seconds
        """
        pyautogui.moveTo(x, y, duration=duration)
        print(f"Moved to ({x}, {y})")

    def click_at(self, x: int, y: int, clicks: int = 1, button: str = 'left'):
        """
        Click at specific coordinates.

        Args:
            x (int): X coordinate
            y (int): Y coordinate
            clicks (int): Number of clicks (1=single, 2=double)
            button (str): 'left', 'right', or 'middle'
        """
        pyautogui.click(x, y, clicks=clicks, button=button)
        print(f"Clicked at ({x}, {y}) - {clicks} {button} click(s)")
        time.sleep(self.delay)

    def click_random_positions(self, num_clicks: int, region: Optional[Tuple[int, int, int, int]] = None):
        """
        Click at random positions on screen.

        Args:
            num_clicks (int): Number of random clicks
            region (tuple): Optional (x, y, width, height) to limit clicking area
        """
        print(f"\nClicking {num_clicks} random positions...")

        if region:
            x, y, width, height = region
        else:
            x, y = 0, 0
            width, height = self.screen_width, self.screen_height

        for i in range(num_clicks):
            rand_x = random.randint(x, x + width - 1)
            rand_y = random.randint(y, y + height - 1)

            print(f"Click {i+1}/{num_clicks}:", end=" ")
            self.click_at(rand_x, rand_y)
            time.sleep(self.delay)

    def click_positions_random_order(self, positions: List[Tuple[int, int]]):
        """
        Click a list of positions in random order.

        Args:
            positions (list): List of (x, y) tuples
        """
        shuffled = positions.copy()
        random.shuffle(shuffled)

        print(f"\nClicking {len(shuffled)} positions in random order...")

        for i, (x, y) in enumerate(shuffled):
            print(f"Click {i+1}/{len(shuffled)}:", end=" ")
            self.click_at(x, y)

    def click_windows_random_order(self, window_positions: List[dict]):
        """
        Click on specific windows in random order.

        Args:
            window_positions (list): List of dicts with 'name', 'x', 'y', 'clicks'
        """
        shuffled = window_positions.copy()
        random.shuffle(shuffled)

        print(f"\nClicking {len(shuffled)} windows in random order...")

        for i, window in enumerate(shuffled):
            name = window.get('name', f'Window {i+1}')
            x = window['x']
            y = window['y']
            clicks = window.get('clicks', 1)
            button = window.get('button', 'left')

            print(f"Window {i+1}/{len(shuffled)}: {name}")
            self.move_to(x, y, duration=0.5)
            time.sleep(0.2)
            self.click_at(x, y, clicks=clicks, button=button)

    def draw_pattern(self, pattern: str = 'square', size: int = 200):
        """
        Draw a pattern with mouse movements.

        Args:
            pattern (str): 'square', 'circle', 'zigzag', 'star'
            size (int): Pattern size in pixels
        """
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2

        print(f"\nDrawing {pattern} pattern (size: {size})...")

        if pattern == 'square':
            points = [
                (center_x - size//2, center_y - size//2),
                (center_x + size//2, center_y - size//2),
                (center_x + size//2, center_y + size//2),
                (center_x - size//2, center_y + size//2),
                (center_x - size//2, center_y - size//2),
            ]
        elif pattern == 'circle':
            import math
            points = []
            for angle in range(0, 360, 10):
                rad = math.radians(angle)
                x = center_x + int(size//2 * math.cos(rad))
                y = center_y + int(size//2 * math.sin(rad))
                points.append((x, y))
        elif pattern == 'zigzag':
            points = []
            for i in range(5):
                x = center_x + (i - 2) * size//4
                y = center_y + (size//4 if i % 2 == 0 else -size//4)
                points.append((x, y))
        elif pattern == 'star':
            import math
            points = []
            for i in range(10):
                angle = (i * 36) - 90
                rad = math.radians(angle)
                radius = size//2 if i % 2 == 0 else size//4
                x = center_x + int(radius * math.cos(rad))
                y = center_y + int(radius * math.sin(rad))
                points.append((x, y))
            points.append(points[0])  # Close the star
        else:
            print(f"Unknown pattern: {pattern}")
            return

        for x, y in points:
            self.move_to(x, y, duration=0.3)
            time.sleep(0.1)

    def type_text(self, text: str, interval: float = 0.1):
        """
        Type text at current cursor position.

        Args:
            text (str): Text to type
            interval (float): Interval between keystrokes
        """
        print(f"Typing: {text}")
        pyautogui.write(text, interval=interval)

    def get_current_position(self) -> Tuple[int, int]:
        """Get current mouse position."""
        x, y = pyautogui.position()
        print(f"Current position: ({x}, {y})")
        return x, y

    def screenshot(self, filename: str = None) -> str:
        """
        Take a screenshot.

        Args:
            filename (str): Output filename (default: timestamp)

        Returns:
            str: Screenshot filename
        """
        if not filename:
            filename = f"screenshot_{int(time.time())}.png"

        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"Screenshot saved: {filename}")
        return filename

    def wait_for_user(self, message: str = "Press Enter to continue..."):
        """Wait for user input before continuing."""
        input(f"\n{message}")


def demo_basic_clicking():
    """Demo: Basic clicking functionality."""
    print("=" * 60)
    print("DEMO: Basic Mouse Clicking")
    print("=" * 60)

    auto = MouseAutomation(safe_mode=True, delay=0.5)

    # Get current position
    auto.get_current_position()

    # Click 5 random positions
    auto.click_random_positions(5)


def demo_window_clicking():
    """Demo: Click specific windows in random order."""
    print("\n" + "=" * 60)
    print("DEMO: Click Windows in Random Order")
    print("=" * 60)

    auto = MouseAutomation(safe_mode=True, delay=0.8)

    # Define window positions (you'll need to adjust these for your screen)
    # You can use auto.get_current_position() to find coordinates
    windows = [
        {'name': 'Browser Tab 1', 'x': 300, 'y': 100, 'clicks': 1},
        {'name': 'Browser Tab 2', 'x': 500, 'y': 100, 'clicks': 1},
        {'name': 'Terminal', 'x': 700, 'y': 400, 'clicks': 1},
        {'name': 'Editor', 'x': 900, 'y': 400, 'clicks': 1},
    ]

    print("\nDefined windows:")
    for w in windows:
        print(f"  - {w['name']} at ({w['x']}, {w['y']})")

    auto.wait_for_user("\nPress Enter to start clicking windows in random order...")

    auto.click_windows_random_order(windows)


def demo_patterns():
    """Demo: Draw patterns with mouse."""
    print("\n" + "=" * 60)
    print("DEMO: Draw Patterns")
    print("=" * 60)

    auto = MouseAutomation(safe_mode=True, delay=0.3)

    patterns = ['square', 'circle', 'zigzag', 'star']

    for pattern in patterns:
        auto.wait_for_user(f"\nPress Enter to draw {pattern} pattern...")
        auto.draw_pattern(pattern, size=200)


def demo_interactive_position_finder():
    """Interactive tool to find window positions."""
    print("\n" + "=" * 60)
    print("INTERACTIVE: Position Finder")
    print("=" * 60)
    print("\nMove your mouse to a window and press Enter to record position.")
    print("Type 'done' when finished.\n")

    auto = MouseAutomation(safe_mode=True)
    positions = []

    while True:
        response = input("Press Enter to record position (or 'done' to finish): ").strip()

        if response.lower() == 'done':
            break

        x, y = pyautogui.position()
        name = input(f"Name for position ({x}, {y}): ").strip() or f"Position {len(positions) + 1}"

        positions.append({'name': name, 'x': x, 'y': y})
        print(f"✓ Recorded: {name} at ({x}, {y})\n")

    if positions:
        print("\n" + "=" * 60)
        print("Recorded Positions:")
        print("=" * 60)
        for p in positions:
            print(f"  {{'name': '{p['name']}', 'x': {p['x']}, 'y': {p['y']}}},")

        print("\nYou can copy these positions and use them in your script!")

        test = input("\nWant to test clicking these in random order? (y/n): ").strip().lower()
        if test == 'y':
            auto.click_windows_random_order(positions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mouse Automation Tool")
    parser.add_argument('--demo', choices=['basic', 'windows', 'patterns', 'finder', 'all'],
                       default='all', help='Demo to run')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MOUSE AUTOMATION TOOL")
    print("=" * 60)
    print("\nSAFETY: Move mouse to top-left corner to abort!")
    print("=" * 60)

    try:
        if args.demo == 'basic' or args.demo == 'all':
            demo_basic_clicking()

        if args.demo == 'windows' or args.demo == 'all':
            demo_window_clicking()

        if args.demo == 'patterns' or args.demo == 'all':
            demo_patterns()

        if args.demo == 'finder':
            demo_interactive_position_finder()

    except KeyboardInterrupt:
        print("\n\nAborted by user!")
    except Exception as e:
        print(f"\n\nError: {e}")
