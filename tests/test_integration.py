#!/usr/bin/env python3
"""
Integration tests for Stereo Camera Calibration & Depth Toolbox.
Tests the complete workflow from loading images to generating depth maps.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_complete_workflow_sphere():
    """Test complete workflow with sphere example."""
    print("Testing complete workflow with sphere example...")
    
    from core.rectifier import StereoRectifier
    from core.depth import DepthEstimator
    import json
    
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    
    left_img = cv2.imread(os.path.join(examples_dir, 'sphere_left.png'))
    right_img = cv2.imread(os.path.join(examples_dir, 'sphere_right.png'))
    
    with open(os.path.join(examples_dir, 'sphere_calibration.json'), 'r') as f:
        calib = json.load(f)
    
    rectifier = StereoRectifier()
    rectifier.set_left_image(left_img)
    rectifier.set_right_image(right_img)
    rectifier.set_left_intrinsics(
        np.array(calib['left_camera']['K']),
        np.array(calib['left_camera']['distortion'])
    )
    rectifier.set_right_intrinsics(
        np.array(calib['right_camera']['K']),
        np.array(calib['right_camera']['distortion'])
    )
    rectifier.set_extrinsics(
        np.array(calib['right_camera']['R']),
        np.array(calib['right_camera']['T'])
    )
    
    rect_left, rect_right = rectifier.rectify()
    assert rect_left is not None, "Rectification failed"
    assert rect_right is not None, "Rectification failed"
    assert rect_left.shape == left_img.shape, "Rectified shape mismatch"
    
    depth_estimator = DepthEstimator()
    disparity = depth_estimator.compute_disparity(rect_left, rect_right)
    stats = depth_estimator.get_disparity_stats()
    assert stats['max'] >= 0, "Disparity should have non-negative values"
    
    depth_map = depth_estimator.compute_depth()
    assert depth_map is not None, "Depth computation failed"
    
    colored = depth_estimator.apply_colormap(disparity)
    assert colored is not None, "Colormap application failed"
    
    print(f"  ✓ Sphere workflow passed (disparity max: {stats['max']:.2f})")
    return True


def test_complete_workflow_all_examples():
    """Test complete workflow with all example pairs."""
    print("Testing complete workflow with all examples...")
    
    from core.rectifier import StereoRectifier
    from core.depth import DepthEstimator
    import json
    
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    test_pairs = ['sphere', 'circles', 'dots', 'checkerboard', 'indoor']
    
    results = {}
    
    for pair_name in test_pairs:
        left_img = cv2.imread(os.path.join(examples_dir, f'{pair_name}_left.png'))
        right_img = cv2.imread(os.path.join(examples_dir, f'{pair_name}_right.png'))
        
        with open(os.path.join(examples_dir, f'{pair_name}_calibration.json'), 'r') as f:
            calib = json.load(f)
        
        rectifier = StereoRectifier()
        rectifier.set_left_image(left_img)
        rectifier.set_right_image(right_img)
        rectifier.set_left_intrinsics(
            np.array(calib['left_camera']['K']),
            np.array(calib['left_camera']['distortion'])
        )
        rectifier.set_right_intrinsics(
            np.array(calib['right_camera']['K']),
            np.array(calib['right_camera']['distortion'])
        )
        rectifier.set_extrinsics(
            np.array(calib['right_camera']['R']),
            np.array(calib['right_camera']['T'])
        )
        
        rect_left, rect_right = rectifier.rectify()
        
        depth_estimator = DepthEstimator()
        disparity = depth_estimator.compute_disparity(rect_left, rect_right)
        stats = depth_estimator.get_disparity_stats()
        
        results[pair_name] = stats
    
    print(f"  ✓ All examples processed successfully")
    for name, stats in results.items():
        print(f"    {name:12s} - disparity: min={stats['min']:6.2f}, max={stats['max']:6.2f}, mean={stats['mean']:6.2f}")
    
    return True


def test_bm_parameters_sensitivity():
    """Test sensitivity to BM parameters."""
    print("Testing BM parameter sensitivity...")
    
    from core.depth import DepthEstimator
    from core.rectifier import StereoRectifier
    
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    left_img = cv2.imread(os.path.join(examples_dir, 'sphere_left.png'))
    right_img = cv2.imread(os.path.join(examples_dir, 'sphere_right.png'))
    
    rectifier = StereoRectifier()
    rectifier.set_left_image(left_img)
    rectifier.set_right_image(right_img)
    
    import json
    with open(os.path.join(examples_dir, 'sphere_calibration.json'), 'r') as f:
        calib = json.load(f)
    
    rectifier.set_left_intrinsics(
        np.array(calib['left_camera']['K']),
        np.array(calib['left_camera']['distortion'])
    )
    rectifier.set_right_intrinsics(
        np.array(calib['right_camera']['K']),
        np.array(calib['right_camera']['distortion'])
    )
    rectifier.set_extrinsics(
        np.array(calib['right_camera']['R']),
        np.array(calib['right_camera']['T'])
    )
    
    rect_left, rect_right = rectifier.rectify()
    
    param_sets = [
        {'numDisparities': 16, 'blockSize': 9},
        {'numDisparities': 32, 'blockSize': 15},
        {'numDisparities': 64, 'blockSize': 21},
    ]
    
    results = []
    
    for params in param_sets:
        estimator = DepthEstimator()
        estimator.set_bm_params(**params)
        
        disparity = estimator.compute_disparity(rect_left, rect_right)
        stats = estimator.get_disparity_stats()
        results.append((params, stats))
    
    print(f"  ✓ BM parameter sensitivity test passed")
    for params, stats in results:
        print(f"    numDisp={params['numDisparities']:2d}, blockSize={params['blockSize']:2d} -> "
              f"max disparity: {stats['max']:6.2f}")
    
    return True


def test_calibration_save_load():
    """Test saving and loading calibration data (JSON serialization only, no GUI)."""
    print("Testing calibration save/load...")
    
    import json
    import tempfile
    
    test_data = {
        'left_camera': {
            'K': [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            'distortion': [-0.1, 0.05, 0, 0, 0]
        },
        'right_camera': {
            'K': [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            'distortion': [-0.1, 0.05, 0, 0, 0],
            'R': [[0.98, -0.1, 0], [0.1, 0.98, 0], [0, 0, 1]],
            'T': [50, 5, 0]
        }
    }
    
    # Test JSON serialization round-trip
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = f.name
    
    try:
        # Reload from file
        with open(temp_path, 'r') as f:
            loaded = json.load(f)
        
        # Verify data integrity
        K_loaded = np.array(loaded['left_camera']['K'])
        assert np.allclose(K_loaded, test_data['left_camera']['K']), "K mismatch after load"
        
        d_loaded = np.array(loaded['left_camera']['distortion'])
        assert np.allclose(d_loaded, test_data['left_camera']['distortion']), "distortion mismatch after load"
        
        R_loaded = np.array(loaded['right_camera']['R'])
        assert np.allclose(R_loaded, test_data['right_camera']['R']), "R mismatch after load"
        
        T_loaded = np.array(loaded['right_camera']['T'])
        assert np.allclose(T_loaded, test_data['right_camera']['T']), "T mismatch after load"
        
        print("  ✓ Calibration save/load test passed")
        return True
        
    finally:
        os.unlink(temp_path)


def test_image_formats():
    """Test loading different image formats."""
    print("Testing image format support...")
    
    from core.rectifier import StereoRectifier
    
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    
    test_formats = ['png']
    
    for fmt in test_formats:
        left_path = os.path.join(examples_dir, f'sphere_left.{fmt}')
        right_path = os.path.join(examples_dir, f'sphere_right.{fmt}')
        
        if os.path.exists(left_path) and os.path.exists(right_path):
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            rectifier = StereoRectifier()
            rectifier.set_left_image(left_img)
            rectifier.set_right_image(right_img)
            
            assert rectifier.left_image is not None, f"Failed to load {fmt} left image"
            assert rectifier.right_image is not None, f"Failed to load {fmt} right image"
            
            print(f"  ✓ {fmt.upper()} format supported")
    
    print("  ✓ Image format test passed")
    return True


def test_rectification_quality():
    """Test rectification produces valid results."""
    print("Testing rectification quality...")
    
    from core.rectifier import StereoRectifier
    import json
    
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    
    left_img = cv2.imread(os.path.join(examples_dir, 'circles_left.png'))
    right_img = cv2.imread(os.path.join(examples_dir, 'circles_right.png'))
    
    with open(os.path.join(examples_dir, 'circles_calibration.json'), 'r') as f:
        calib = json.load(f)
    
    rectifier = StereoRectifier()
    rectifier.set_left_image(left_img)
    rectifier.set_right_image(right_img)
    rectifier.set_left_intrinsics(
        np.array(calib['left_camera']['K']),
        np.array(calib['left_camera']['distortion'])
    )
    rectifier.set_right_intrinsics(
        np.array(calib['right_camera']['K']),
        np.array(calib['right_camera']['distortion'])
    )
    rectifier.set_extrinsics(
        np.array(calib['right_camera']['R']),
        np.array(calib['right_camera']['T'])
    )
    
    rect_left, rect_right = rectifier.rectify()
    
    assert rect_left.shape == rect_right.shape, "Rectified images should have same shape"
    
    assert rect_left.dtype == left_img.dtype, "Dtype should be preserved"
    
    with_epipolar = rectifier.draw_epipolar_lines(rect_left, num_lines=10)
    assert with_epipolar is not None, "Epipolar lines drawing failed"
    assert with_epipolar.shape == rect_left.shape, "Shape should be unchanged"
    
    print("  ✓ Rectification quality test passed")
    return True


def test_depth_range():
    """Test that depth values are in reasonable range."""
    print("Testing depth value range...")
    
    from core.depth import DepthEstimator
    from core.rectifier import StereoRectifier
    import json
    
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    
    left_img = cv2.imread(os.path.join(examples_dir, 'circles_left.png'))
    right_img = cv2.imread(os.path.join(examples_dir, 'circles_right.png'))
    
    with open(os.path.join(examples_dir, 'circles_calibration.json'), 'r') as f:
        calib = json.load(f)
    
    rectifier = StereoRectifier()
    rectifier.set_left_image(left_img)
    rectifier.set_right_image(right_img)
    rectifier.set_left_intrinsics(
        np.array(calib['left_camera']['K']),
        np.array(calib['left_camera']['distortion'])
    )
    rectifier.set_right_intrinsics(
        np.array(calib['right_camera']['K']),
        np.array(calib['right_camera']['distortion'])
    )
    rectifier.set_extrinsics(
        np.array(calib['right_camera']['R']),
        np.array(calib['right_camera']['T'])
    )
    
    rect_left, rect_right = rectifier.rectify()
    
    estimator = DepthEstimator()
    estimator.set_bm_params(numDisparities=32, blockSize=15)
    estimator.set_camera_params(baseline=0.03, focal_length=600)
    
    disparity = estimator.compute_disparity(rect_left, rect_right)
    depth = estimator.compute_depth()
    
    assert depth is not None, "Depth computation failed"
    
    valid_depth = depth[depth > 0]
    assert len(valid_depth) > 0, "No valid depth values"
    
    assert np.min(valid_depth) > 0, "Min depth should be positive"
    assert np.max(valid_depth) < 10000, "Max depth should be reasonable"
    
    print(f"  ✓ Depth range test passed (range: {np.min(valid_depth):.2f} - {np.max(valid_depth):.2f}m)")
    return True


def run_all_integration_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Integration Test Suite")
    print("=" * 60)
    print()
    
    from core.rectifier import StereoRectifier
    
    tests = [
        test_complete_workflow_sphere,
        test_complete_workflow_all_examples,
        test_bm_parameters_sensitivity,
        test_calibration_save_load,
        test_image_formats,
        test_rectification_quality,
        test_depth_range,
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
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)
