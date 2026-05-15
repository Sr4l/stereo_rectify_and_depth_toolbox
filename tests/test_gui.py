#!/usr/bin/env python3
"""
GUI component tests for Stereo Camera Calibration & Depth Toolbox (Qt6/PySide6).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gui_imports():
    """Test that all GUI modules can be imported."""
    print("Testing GUI imports...")

    try:
        from gui.qt_param_panel import CameraParamPanel
        print("  ✓ CameraParamPanel imported")
    except Exception as e:
        print(f"  ✗ CameraParamPanel import failed: {e}")
        return False

    try:
        from gui.qt_image_panel import ImagePanel, ThumbnailPanel
        print("  ✓ ImagePanel and ThumbnailPanel imported")
    except Exception as e:
        print(f"  ✗ ImagePanel/ThumbnailPanel import failed: {e}")
        return False

    try:
        from gui.qt_main_window import StereoCalibrationGUI
        print("  ✓ StereoCalibrationGUI imported")
    except Exception as e:
        print(f"  ✗ StereoCalibrationGUI import failed: {e}")
        return False

    return True


def test_camera_param_panel():
    """Test CameraParamPanel creation and basic functionality."""
    print("Testing CameraParamPanel...")

    try:
        import numpy as np
        from PySide6.QtWidgets import QApplication
        from gui.qt_param_panel import CameraParamPanel

        # Get or create QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        def on_change():
            pass

        panel = CameraParamPanel(None, title="Test Camera", on_change=on_change)

        K = panel.get_K()
        assert K.shape == (3, 3), f"K shape should be (3,3), got {K.shape}"
        print("  ✓ get_K() works")

        dist = panel.get_distortion()
        assert dist.shape == (5,), f"dist shape should be (5,), got {dist.shape}"
        print("  ✓ get_distortion() works")

        R = panel.get_R()
        assert R.shape == (3, 3), f"R shape should be (3,3), got {R.shape}"
        print("  ✓ get_R() works")

        T = panel.get_T()
        assert T.shape == (3,), f"T shape should be (3,), got {T.shape}"
        print("  ✓ get_T() works")

        test_K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
        panel.set_K(test_K)
        retrieved_K = panel.get_K()
        assert np.allclose(retrieved_K, test_K), "set_K/get_K mismatch"
        print("  ✓ set_K() works")

        test_dist = np.array([-0.2, 0.1, 0, 0, 0], dtype=np.float64)
        panel.set_distortion(test_dist)
        retrieved_dist = panel.get_distortion()
        assert np.allclose(retrieved_dist, test_dist), "set_distortion/get_distortion mismatch"
        print("  ✓ set_distortion() works")

        test_R = np.array([[0.98, -0.1, 0], [0.1, 0.98, 0], [0, 0, 1]], dtype=np.float64)
        panel.set_R(test_R)
        retrieved_R = panel.get_R()
        assert np.allclose(retrieved_R, test_R), "set_R/get_R mismatch"
        print("  ✓ set_R() works")

        test_T = np.array([50, 5, 0], dtype=np.float64)
        panel.set_T(test_T)
        retrieved_T = panel.get_T()
        assert np.allclose(retrieved_T, test_T), "set_T/get_T mismatch"
        print("  ✓ set_T() works")

        panel.reset_to_identity()
        print("  ✓ reset_to_identity() works")

        print("  ✓ CameraParamPanel tests passed")
        return True

    except Exception as e:
        print(f"  ✗ CameraParamPanel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thumbnail_panel():
    """Test ThumbnailPanel creation and basic functionality."""
    print("Testing ThumbnailPanel...")

    try:
        import numpy as np
        from PySide6.QtWidgets import QApplication
        from gui.qt_image_panel import ThumbnailPanel

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        panel = ThumbnailPanel(None, title="Test Thumbnail", size=(200, 150))

        panel.set_image(None)
        print("  ✓ set_image(None) works")

        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        panel.set_image(test_img)
        print("  ✓ set_image() works")

        panel.clear()
        assert panel._image is None, "Image should be None after clear"
        print("  ✓ clear() works")

        print("  ✓ ThumbnailPanel tests passed")
        return True

    except Exception as e:
        print(f"  ✗ ThumbnailPanel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_panel():
    """Test ImagePanel creation and basic functionality."""
    print("Testing ImagePanel...")

    try:
        import numpy as np
        from PySide6.QtWidgets import QApplication
        from gui.qt_image_panel import ImagePanel

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        panel = ImagePanel(None, title="Test Image", show_controls=True)

        panel.set_image(None)
        print("  ✓ set_image(None) works")

        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        panel.set_image(test_img)
        assert panel._image is not None, "Image should be set"
        print("  ✓ set_image() works")

        panel._fit_to_window()
        print("  ✓ _fit_to_window() works")

        panel._reset_zoom()
        assert panel._zoom_factor == 1.0, "Zoom should be 1.0 after reset"
        print("  ✓ _reset_zoom() works")

        panel._reset_view()
        print("  ✓ _reset_view() works")

        panel.clear()
        assert panel._image is None, "Image should be None after clear"
        print("  ✓ clear() works")

        print("  ✓ ImagePanel tests passed")
        return True

    except Exception as e:
        print(f"  ✗ ImagePanel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_window_initialization():
    """Test StereoCalibrationGUI initialization."""
    print("Testing StereoCalibrationGUI initialization...")

    try:
        from PySide6.QtWidgets import QApplication
        from gui.qt_main_window import StereoCalibrationGUI

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        window = StereoCalibrationGUI()

        assert window.rectifier is not None, "Rectifier should exist"
        assert window.depth_estimator is not None, "Depth estimator should exist"

        assert window._left_param_panel is not None, "Left param panel should exist"
        assert window._right_param_panel is not None, "Right param panel should exist"

        assert window._rectified_left_panel is not None, "Rectified left panel should exist"
        assert window._rectified_right_panel is not None, "Rectified right panel should exist"
        assert window._depth_panel is not None, "Depth panel should exist"

        window.close()

        print("  ✓ StereoCalibrationGUI initialization passed")
        return True

    except Exception as e:
        print(f"  ✗ StereoCalibrationGUI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_gui_tests():
    """Run all GUI tests."""
    print("=" * 60)
    print("GUI Component Test Suite (PySide6)")
    print("=" * 60)
    print()

    tests = [
        test_gui_imports,
        test_camera_param_panel,
        test_thumbnail_panel,
        test_image_panel,
        test_main_window_initialization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_gui_tests()
    sys.exit(0 if success else 1)