#!/usr/bin/env python3
"""
GUI component tests for Stereo Camera Calibration & Depth Toolbox.
"""

import sys
import os
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gui_imports():
    """Test that all GUI modules can be imported."""
    print("Testing GUI imports...")
    
    try:
        from gui.param_panel import CameraParamPanel
        print("  ✓ CameraParamPanel imported")
    except Exception as e:
        print(f"  ✗ CameraParamPanel import failed: {e}")
        return False
    
    try:
        from gui.image_panel import ImagePanel, ThumbnailPanel
        print("  ✓ ImagePanel and ThumbnailPanel imported")
    except Exception as e:
        print(f"  ✗ ImagePanel/ThumbnailPanel import failed: {e}")
        return False
    
    try:
        from gui.main_window import StereoCalibrationGUI
        print("  ✓ StereoCalibrationGUI imported")
    except Exception as e:
        print(f"  ✗ StereoCalibrationGUI import failed: {e}")
        return False
    
    return True


def test_camera_param_panel():
    """Test CameraParamPanel creation and basic functionality."""
    print("Testing CameraParamPanel...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        import numpy as np
        from gui.param_panel import CameraParamPanel
        
        change_count = [0]
        
        def on_change():
            change_count[0] += 1
        
        panel = CameraParamPanel(root, title="Test Camera", on_change=on_change)
        panel.pack()
        
        root.update_idletasks()
        
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
        assert change_count[0] > 0, "on_change not called during reset"
        print(f"  ✓ reset_to_identity() works (callbacks: {change_count[0]})")
        
        root.destroy()
        print("  ✓ CameraParamPanel tests passed")
        return True
        
    except Exception as e:
        root.destroy()
        print(f"  ✗ CameraParamPanel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thumbnail_panel():
    """Test ThumbnailPanel creation and basic functionality."""
    print("Testing ThumbnailPanel...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        import numpy as np
        from gui.image_panel import ThumbnailPanel
        
        panel = ThumbnailPanel(root, title="Test Thumbnail", size=(200, 150))
        panel.pack()
        
        root.update_idletasks()
        
        panel.set_image(None)
        print("  ✓ set_image(None) works")
        
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        panel.set_image(test_img)
        print("  ✓ set_image() works")
        
        panel.clear()
        assert panel.image is None, "Image should be None after clear"
        print("  ✓ clear() works")
        
        root.destroy()
        print("  ✓ ThumbnailPanel tests passed")
        return True
        
    except Exception as e:
        root.destroy()
        print(f"  ✗ ThumbnailPanel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_panel():
    """Test ImagePanel creation and basic functionality."""
    print("Testing ImagePanel...")
    
    root = tk.Tk()
    root.withdraw()
    
    try:
        import numpy as np
        from gui.image_panel import ImagePanel
        
        panel = ImagePanel(root, title="Test Image", show_controls=True)
        panel.pack()
        
        root.update_idletasks()
        
        panel.set_image(None)
        print("  ✓ set_image(None) works")
        
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        panel.set_image(test_img)
        assert panel.image is not None, "Image should be set"
        print("  ✓ set_image() works")
        
        panel._fit_to_window()
        print("  ✓ _fit_to_window() works")
        
        panel._reset_zoom()
        assert panel.zoom == 1.0, "Zoom should be 1.0 after reset"
        print("  ✓ _reset_zoom() works")
        
        panel._reset_view()
        print("  ✓ _reset_view() works")
        
        panel.clear()
        assert panel.image is None, "Image should be None after clear"
        print("  ✓ clear() works")
        
        root.destroy()
        print("  ✓ ImagePanel tests passed")
        return True
        
    except Exception as e:
        root.destroy()
        print(f"  ✗ ImagePanel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_window_initialization():
    """Test StereoCalibrationGUI initialization."""
    print("Testing StereoCalibrationGUI initialization...")
    
    try:
        from gui.main_window import StereoCalibrationGUI
        
        app = StereoCalibrationGUI()
        
        assert app.root is not None, "Root window should exist"
        assert app.rectifier is not None, "Rectifier should exist"
        assert app.depth_estimator is not None, "Depth estimator should exist"
        
        assert app.left_param_panel is not None, "Left param panel should exist"
        assert app.right_param_panel is not None, "Right param panel should exist"
        
        assert app.rectified_left_panel is not None, "Rectified left panel should exist"
        assert app.rectified_right_panel is not None, "Rectified right panel should exist"
        assert app.depth_panel is not None, "Depth panel should exist"
        
        app.root.destroy()
        
        print("  ✓ StereoCalibrationGUI initialization passed")
        return True
        
    except Exception as e:
        print(f"  ✗ StereoCalibrationGUI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_window_with_example():
    """Test main window with example images."""
    print("Testing StereoCalibrationGUI with example images...")
    
    try:
        from gui.main_window import StereoCalibrationGUI
        import cv2
        import numpy as np
        
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
        left_path = os.path.join(examples_dir, 'sphere_left.png')
        right_path = os.path.join(examples_dir, 'sphere_right.png')
        
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print("  ⚠ Example images not found, skipping")
            return True
        
        app = StereoCalibrationGUI()
        
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        app.rectifier.set_left_image(left_img)
        app.rectifier.set_right_image(right_img)
        
        assert app.rectifier.left_image is not None, "Left image not set"
        assert app.rectifier.right_image is not None, "Right image not set"
        
        import json
        calib_path = os.path.join(examples_dir, 'sphere_calibration.json')
        with open(calib_path, 'r') as f:
            calib = json.load(f)
        
        app.left_param_panel.set_K(np.array(calib['left_camera']['K']))
        app.left_param_panel.set_distortion(np.array(calib['left_camera']['distortion']))
        app.right_param_panel.set_K(np.array(calib['right_camera']['K']))
        app.right_param_panel.set_distortion(np.array(calib['right_camera']['distortion']))
        app.right_param_panel.set_R(np.array(calib['right_camera']['R']))
        app.right_param_panel.set_T(np.array(calib['right_camera']['T']))
        
        app._update_rectification()
        
        assert app.rectifier.rectified_left is not None, "Rectified left should exist"
        assert app.rectifier.rectified_right is not None, "Rectified right should exist"
        
        assert app.depth_estimator.disparity is not None, "Disparity should exist"
        
        app.root.destroy()
        
        print("  ✓ StereoCalibrationGUI with example passed")
        return True
        
    except Exception as e:
        print(f"  ✗ StereoCalibrationGUI with example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_gui_tests():
    """Run all GUI tests."""
    print("=" * 60)
    print("GUI Component Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_gui_imports,
        test_camera_param_panel,
        test_thumbnail_panel,
        test_image_panel,
        test_main_window_initialization,
        test_main_window_with_example,
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
