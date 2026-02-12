#!/usr/bin/env python3
"""
Quick test for mouse automation functionality.
Tests basic imports and functions without actually clicking.
"""

def test_imports():
    """Test that pyautogui can be imported."""
    print("Testing imports...")

    try:
        import pyautogui
        print("✓ pyautogui imported")
    except ImportError as e:
        print(f"✗ pyautogui import failed: {e}")
        return False

    return True


def test_mouse_automation_import():
    """Test that mouse_automation module can be imported."""
    print("\nTesting mouse_automation module...")

    try:
        from mouse_automation import MouseAutomation
        print("✓ MouseAutomation class imported")

        # Initialize (but don't execute any actions)
        auto = MouseAutomation(safe_mode=True, delay=0.5)
        print("✓ MouseAutomation initialized")

        return True
    except Exception as e:
        print(f"✗ MouseAutomation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_import():
    """Test that integration module can be imported."""
    print("\nTesting integration module...")

    try:
        from tongue_click_mouse_integration import TongueClickMouseControl
        print("✓ TongueClickMouseControl class imported")

        # Initialize (but don't start detection)
        controller = TongueClickMouseControl(threshold=0.3)
        print("✓ TongueClickMouseControl initialized")

        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_screen_info():
    """Test getting screen information."""
    print("\nTesting screen info...")

    try:
        import pyautogui

        width, height = pyautogui.size()
        print(f"✓ Screen size: {width}x{height}")

        x, y = pyautogui.position()
        print(f"✓ Current mouse position: ({x}, {y})")

        return True
    except Exception as e:
        print(f"✗ Screen info test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MOUSE AUTOMATION - BASIC TESTS")
    print("=" * 60 + "\n")

    all_passed = True

    if not test_imports():
        print("\n✗ Import test failed. Please install pyautogui:")
        print("  pip install pyautogui")
        all_passed = False

    if not test_mouse_automation_import():
        all_passed = False

    if not test_integration_import():
        all_passed = False

    if not test_screen_info():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nMouse automation is ready to use!")
        print("\nNext steps:")
        print("  - Run: python mouse_automation.py --demo finder")
        print("  - Or: python tongue_click_mouse_integration.py --mode current")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
