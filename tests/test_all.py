#!/usr/bin/env python3
"""
Comprehensive test suite for Stereo Camera Calibration & Depth Toolbox.
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rectifier import StereoRectifier
from core.depth import DepthEstimator, StereoBMParams


def test_stereo_rectifier_initialization():
    """Test StereoRectifier initialization."""
    print("Testing StereoRectifier initialization...")
    
    rectifier = StereoRectifier()
    
    assert rectifier.left_image is None, "Left image should be None initially"
    assert rectifier.right_image is None, "Right image should be None initially"
    assert np.allclose(rectifier.left_K, np.eye(3)), "Left K should be identity"
    assert np.allclose(rectifier.right_K, np.eye(3)), "Right K should be identity"
    assert np.allclose(rectifier.left_dist, np.zeros(5)), "Left dist should be zeros"
    assert np.allclose(rectifier.right_dist, np.zeros(5)), "Right dist should be zeros"
    assert np.allclose(rectifier.R, np.eye(3)), "R should be identity"
    assert np.allclose(rectifier.T, np.zeros(3)), "T should be zeros"
    
    print("  ✓ StereoRectifier initialization passed")


def test_stereo_rectifier_set_images():
    """Test setting images on StereoRectifier."""
    print("Testing StereoRectifier set images...")
    
    rectifier = StereoRectifier()
    left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    rectifier.set_left_image(left_img)
    rectifier.set_right_image(right_img)
    
    assert rectifier.left_image is not None, "Left image should be set"
    assert rectifier.right_image is not None, "Right image should be set"
    assert rectifier.left_image.shape == (480, 640, 3), "Left image shape mismatch"
    assert rectifier.right_image.shape == (480, 640, 3), "Right image shape mismatch"
    
    print("  ✓ Set images passed")


def test_stereo_rectifier_set_intrinsics():
    """Test setting intrinsics on StereoRectifier."""
    print("Testing StereoRectifier set intrinsics...")
    
    rectifier = StereoRectifier()
    
    K_left = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist_left = np.array([-0.1, 0.05, 0, 0, 0], dtype=np.float64)
    
    K_right = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist_right = np.array([-0.1, 0.05, 0, 0, 0], dtype=np.float64)
    
    rectifier.set_left_intrinsics(K_left, dist_left)
    rectifier.set_right_intrinsics(K_right, dist_right)
    
    assert np.allclose(rectifier.left_K, K_left), "Left K not set correctly"
    assert np.allclose(rectifier.left_dist, dist_left), "Left dist not set correctly"
    assert np.allclose(rectifier.right_K, K_right), "Right K not set correctly"
    assert np.allclose(rectifier.right_dist, dist_right), "Right dist not set correctly"
    
    print("  ✓ Set intrinsics passed")


def test_stereo_rectifier_set_extrinsics():
    """Test setting extrinsics on StereoRectifier."""
    print("Testing StereoRectifier set extrinsics...")
    
    rectifier = StereoRectifier()
    
    R = np.array([[0.98, -0.1, 0], [0.1, 0.98, 0], [0, 0, 1]], dtype=np.float64)
    T = np.array([50, 5, 0], dtype=np.float64)
    
    rectifier.set_extrinsics(R, T)
    
    assert np.allclose(rectifier.R, R), "R not set correctly"
    assert np.allclose(rectifier.T, T), "T not set correctly"
    
    print("  ✓ Set extrinsics passed")


def test_stereo_rectifier_get_image_size():
    """Test getting image size from StereoRectifier."""
    print("Testing StereoRectifier get image size...")
    
    rectifier = StereoRectifier()
    
    size = rectifier.get_image_size()
    assert size is None, "Size should be None when no image loaded"
    
    left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    rectifier.set_left_image(left_img)
    
    size = rectifier.get_image_size()
    assert size == (640, 480), f"Size should be (640, 480), got {size}"
    
    print("  ✓ Get image size passed")


def test_stereo_rectifier_rectify():
    """Test rectification with StereoRectifier."""
    print("Testing StereoRectifier rectify...")
    
    rectifier = StereoRectifier()
    
    result = rectifier.rectify()
    assert result == (None, None), "Rectify should return None when no images loaded"
    
    left_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    right_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    rectifier.set_left_image(left_img)
    rectifier.set_right_image(right_img)
    
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    rectifier.set_left_intrinsics(K, np.zeros(5))
    rectifier.set_right_intrinsics(K, np.zeros(5))
    rectifier.set_extrinsics(np.eye(3), np.array([50, 0, 0]))
    
    rect_left, rect_right = rectifier.rectify()
    
    assert rect_left is not None, "Rectified left should not be None"
    assert rect_right is not None, "Rectified right should not be None"
    assert rect_left.shape == (480, 640, 3), f"Rectified left shape mismatch: {rect_left.shape}"
    assert rect_right.shape == (480, 640, 3), f"Rectified right shape mismatch: {rect_right.shape}"
    
    print("  ✓ Rectify passed")


def test_stereo_rectifier_draw_epipolar_lines():
    """Test drawing epipolar lines."""
    print("Testing StereoRectifier draw epipolar lines...")
    
    rectifier = StereoRectifier()
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = rectifier.draw_epipolar_lines(img, num_lines=10)
    
    assert result is not None, "Result should not be None"
    assert result.shape == img.shape, "Shape should be unchanged"
    
    result_none = rectifier.draw_epipolar_lines(None)
    assert result_none is None, "Should return None for None input"
    
    print("  ✓ Draw epipolar lines passed")


def test_depth_estimator_initialization():
    """Test DepthEstimator initialization."""
    print("Testing DepthEstimator initialization...")
    
    estimator = DepthEstimator()
    
    assert estimator.bm_params.numDisparities == 16, "Default BM numDisparities should be 16"
    assert estimator.bm_params.blockSize == 9, "Default BM blockSize should be 9"
    assert estimator.sgbm_params.numDisparities == 16, "Default SGBM numDisparities should be 16"
    assert estimator.sgbm_params.blockSize == 9, "Default SGBM blockSize should be 9"
    assert estimator.algorithm == 'BM', "Default algorithm should be BM"
    assert estimator.disparity is None, "Disparity should be None initially"
    assert estimator.depth_map is None, "Depth map should be None initially"
    
    print("  ✓ DepthEstimator initialization passed")


def test_depth_estimator_set_bm_params():
    """Test setting BM parameters."""
    print("Testing DepthEstimator set BM params...")
    
    estimator = DepthEstimator()
    
    estimator.set_bm_params(numDisparities=32, blockSize=15, minDisparity=5)
    
    assert estimator.bm_params.numDisparities == 32, "numDisparities not set"
    assert estimator.bm_params.blockSize == 15, "blockSize not set"
    assert estimator.bm_params.minDisparity == 5, "minDisparity not set"
    
    print("  ✓ Set BM params passed")


def test_depth_estimator_set_sgbm_params():
    """Test setting SGBM parameters."""
    print("Testing DepthEstimator set SGBM params...")
    
    estimator = DepthEstimator()
    
    estimator.set_sgbm_params(numDisparities=32, blockSize=15, P1=300, P2=500)
    
    assert estimator.sgbm_params.numDisparities == 32, "numDisparities not set"
    assert estimator.sgbm_params.blockSize == 15, "blockSize not set"
    assert estimator.sgbm_params.P1 == 300, "P1 not set"
    assert estimator.sgbm_params.P2 == 500, "P2 not set"
    
    print("  ✓ Set SGBM params passed")


def test_depth_estimator_set_algorithm():
    """Test setting algorithm."""
    print("Testing DepthEstimator set algorithm...")
    
    estimator = DepthEstimator()
    
    assert estimator.algorithm == 'BM', "Default should be BM"
    
    estimator.set_algorithm('SGBM')
    assert estimator.algorithm == 'SGBM', "Failed to set SGBM"
    
    estimator.set_algorithm('bm')
    assert estimator.algorithm == 'BM', "Should uppercase BM"
    
    print("  ✓ Set algorithm passed")


def test_depth_estimator_set_camera_params():
    """Test setting camera parameters."""
    print("Testing DepthEstimator set camera params...")
    
    estimator = DepthEstimator()
    
    estimator.set_camera_params(baseline=0.1, focal_length=500)
    
    assert estimator.baseline == 0.1, "Baseline not set"
    assert estimator.focal_length == 500, "Focal length not set"
    
    print("  ✓ Set camera params passed")


def test_depth_estimator_compute_disparity():
    """Test disparity computation."""
    print("Testing DepthEstimator compute disparity...")
    
    estimator = DepthEstimator()
    
    result = estimator.compute_disparity(None, None)
    assert result is None, "Should return None for None inputs"
    
    left_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    right_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    disparity = estimator.compute_disparity(left_img, right_img)
    
    assert disparity is not None, "Disparity should not be None"
    assert disparity.shape == (480, 640), f"Disparity shape mismatch: {disparity.shape}"
    assert disparity.dtype == np.float32, f"Disparity dtype should be float32, got {disparity.dtype}"
    
    print("  ✓ Compute disparity passed")


def test_depth_estimator_compute_disparity_sgbm():
    """Test SGBM disparity computation."""
    print("Testing DepthEstimator compute disparity SGBM...")
    
    estimator = DepthEstimator()
    estimator.set_algorithm('SGBM')
    
    result = estimator.compute_disparity_sgbm(None, None)
    assert result is None, "Should return None for None inputs"
    
    left_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    right_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    disparity = estimator.compute_disparity_sgbm(left_img, right_img)
    
    assert disparity is not None, "Disparity should not be None"
    assert disparity.shape == (480, 640), f"Disparity shape mismatch: {disparity.shape}"
    assert disparity.dtype == np.float32, f"Disparity dtype should be float32, got {disparity.dtype}"
    
    print("  ✓ Compute disparity SGBM passed")


def test_depth_estimator_compute_depth():
    """Test depth computation."""
    print("Testing DepthEstimator compute depth...")
    
    estimator = DepthEstimator()
    
    result = estimator.compute_depth(None)
    assert result is None, "Should return None for None disparity"
    
    disparity = np.ones((100, 100), dtype=np.float32) * 10
    estimator.disparity = disparity
    
    depth = estimator.compute_depth()
    
    assert depth is not None, "Depth should not be None"
    assert depth.shape == (100, 100), f"Depth shape mismatch: {depth.shape}"
    
    print("  ✓ Compute depth passed")


def test_depth_estimator_apply_colormap():
    """Test applying colormap to disparity."""
    print("Testing DepthEstimator apply colormap...")
    
    estimator = DepthEstimator()
    
    result = estimator.apply_colormap(None)
    assert result is None, "Should return None for None disparity"
    
    disparity = np.random.uniform(0, 50, (100, 100)).astype(np.float32)
    
    colored = estimator.apply_colormap(disparity, cv2.COLORMAP_JET)
    
    assert colored is not None, "Colored disparity should not be None"
    assert colored.shape == (100, 100, 3), f"Colored shape mismatch: {colored.shape}"
    assert colored.dtype == np.uint8, f"Colored dtype should be uint8, got {colored.dtype}"
    
    print("  ✓ Apply colormap passed")


def test_depth_estimator_get_stats():
    """Test getting disparity and depth stats."""
    print("Testing DepthEstimator get stats...")
    
    estimator = DepthEstimator()
    
    stats = estimator.get_disparity_stats()
    assert 'min' in stats, "Stats should have 'min'"
    assert 'max' in stats, "Stats should have 'max'"
    assert 'mean' in stats, "Stats should have 'mean'"
    assert 'std' in stats, "Stats should have 'std'"
    
    disparity = np.ones((100, 100), dtype=np.float32) * 10
    disparity[50:60, 50:60] = 20
    estimator.disparity = disparity
    
    stats = estimator.get_disparity_stats()
    assert stats['min'] == 10.0, f"Min should be 10, got {stats['min']}"
    assert stats['max'] == 20.0, f"Max should be 20, got {stats['max']}"
    
    depth_stats = estimator.get_depth_stats()
    assert 'min' in depth_stats, "Depth stats should have 'min'"
    
    print("  ✓ Get stats passed")


def test_stereobm_params():
    """Test StereoBMParams dataclass."""
    print("Testing StereoBMParams...")
    
    params = StereoBMParams()
    
    assert params.numDisparities == 16, "Default numDisparities"
    assert params.blockSize == 9, "Default blockSize"
    assert params.minDisparity == 0, "Default minDisparity"
    assert params.uniquenessRatio == 10, "Default uniquenessRatio"
    assert params.speckleWindowSize == 100, "Default speckleWindowSize"
    assert params.speckleRange == 1, "Default speckleRange"
    
    params2 = StereoBMParams(numDisparities=32, blockSize=15)
    assert params2.numDisparities == 32, "Custom numDisparities"
    assert params2.blockSize == 15, "Custom blockSize"
    
    print("  ✓ StereoBMParams passed")


def test_with_example_images():
    """Test with actual example images."""
    print("Testing with example images...")
    
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')
    
    if not os.path.exists(examples_dir):
        print("  ⚠ Examples directory not found, skipping")
        return
    
    test_pairs = ['sphere', 'circles', 'dots', 'checkerboard', 'indoor']
    
    for pair_name in test_pairs:
        left_path = os.path.join(examples_dir, f"{pair_name}_left.png")
        right_path = os.path.join(examples_dir, f"{pair_name}_right.png")
        calib_path = os.path.join(examples_dir, f"{pair_name}_calibration.json")
        
        if not all(os.path.exists(p) for p in [left_path, right_path, calib_path]):
            print(f"  ⚠ {pair_name} files not found, skipping")
            continue
        
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        assert left_img is not None, f"Failed to load {left_path}"
        assert right_img is not None, f"Failed to load {right_path}"
        
        import json
        with open(calib_path, 'r') as f:
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
        assert rect_left is not None, f"{pair_name}: Rectification failed"
        assert rect_right is not None, f"{pair_name}: Rectification failed"
        
        depth_estimator = DepthEstimator()
        disparity = depth_estimator.compute_disparity(rect_left, rect_right)
        assert disparity is not None, f"{pair_name}: Disparity computation failed"
        
        colored = depth_estimator.apply_colormap(disparity)
        assert colored is not None, f"{pair_name}: Colormap application failed"
        
        print(f"  ✓ {pair_name} example passed")
    
    print("  ✓ All example images passed")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    rectifier = StereoRectifier()
    
    result = rectifier.rectify()
    assert result == (None, None), "Should handle None images"
    
    estimator = DepthEstimator()
    
    result = estimator.compute_disparity(None, None)
    assert result is None, "Should handle None images"
    
    result = estimator.compute_depth(None)
    assert result is None, "Should handle None disparity"
    
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    rectifier.set_left_image(img)
    rectifier.set_right_image(img)
    
    K = np.eye(3, dtype=np.float64)
    K[0, 0] = 0
    K[1, 1] = 0
    
    rectifier.set_left_intrinsics(K, np.zeros(5))
    rectifier.set_right_intrinsics(K, np.zeros(5))
    
    result = rectifier.rectify()
    assert result[0] is None, "Should return None for invalid intrinsics"
    
    print("  ✓ Edge cases passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Stereo Camera Calibration & Depth Toolbox - Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_stereo_rectifier_initialization,
        test_stereo_rectifier_set_images,
        test_stereo_rectifier_set_intrinsics,
        test_stereo_rectifier_set_extrinsics,
        test_stereo_rectifier_get_image_size,
        test_stereo_rectifier_rectify,
        test_stereo_rectifier_draw_epipolar_lines,
        test_depth_estimator_initialization,
        test_depth_estimator_set_bm_params,
        test_depth_estimator_set_sgbm_params,
        test_depth_estimator_set_algorithm,
        test_depth_estimator_set_camera_params,
        test_depth_estimator_compute_disparity,
        test_depth_estimator_compute_disparity_sgbm,
        test_depth_estimator_compute_depth,
        test_depth_estimator_apply_colormap,
        test_depth_estimator_get_stats,
        test_stereobm_params,
        test_with_example_images,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
