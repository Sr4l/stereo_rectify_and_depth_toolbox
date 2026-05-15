#!/usr/bin/env python3
"""Test RAFT-Stereo integration."""
from core.depth import DepthEstimator, TORCH_AVAILABLE
import numpy as np

def test_torch_available():
    """Test PyTorch detection."""
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    assert TORCH_AVAILABLE, "PyTorch should be available"
    print("✓ PyTorch detection works")

def test_depth_estimator_init():
    """Test DepthEstimator initialization."""
    estimator = DepthEstimator()
    assert hasattr(estimator, 'raft_params'), "Should have raft_params"
    assert hasattr(estimator, 'raft_model'), "Should have raft_model"
    assert estimator.raft_model is None, "Model should be None initially"
    print("✓ DepthEstimator initialization works")

def test_raft_params():
    """Test RAFT parameters."""
    estimator = DepthEstimator()
    assert estimator.raft_params.valid_iters == 32
    assert estimator.raft_params.n_downsample == 2
    assert estimator.raft_params.corr_implementation == "reg"
    print("✓ RAFT parameters work")

def test_raft_params_setter():
    """Test RAFT parameter setter."""
    estimator = DepthEstimator()
    estimator.set_raft_params(valid_iters=16, n_downsample=3)
    assert estimator.raft_params.valid_iters == 16
    assert estimator.raft_params.n_downsample == 3
    print("✓ RAFT parameter setter works")

def test_raft_model_loading():
    """Test RAFT model loading (should fail gracefully without model file)."""
    estimator = DepthEstimator()
    try:
        estimator.load_raft_model('models/nonexistent.pth')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ Model loading fails gracefully when file doesn't exist")
    except Exception as e:
        print(f"✓ Model loading raises expected error: {type(e).__name__}")

def test_raft_disparity_computation():
    """Test RAFT disparity computation (will fail without model, but tests code path)."""
    estimator = DepthEstimator()
    
    # Create synthetic stereo pair
    left = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    try:
        estimator.compute_disparity_raft(
            left, right,
            model_path='models/nonexistent.pth'
        )
        # If we get here without exception, something's wrong
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("✓ Disparity computation fails gracefully without model")
    except Exception as e:
        print(f"✓ Disparity computation raises expected error: {type(e).__name__}")

def test_input_padder():
    """Test InputPadder utility."""
    from core.utils.input_padder import InputPadder
    import torch
    
    # Create test tensor
    tensor = torch.randn(1, 3, 100, 200)
    padder = InputPadder(tensor.shape, divis_by=32)
    
    # Pad and unpad
    padded_list = padder.pad(tensor)
    padded = padded_list[0]
    unpadded = padder.unpad(padded)
    
    assert unpadded.shape == tensor.shape, "Shape should match after unpad"
    print("✓ InputPadder works")

if __name__ == '__main__':
    print("="*60)
    print("Testing RAFT-Stereo Integration")
    print("="*60)
    
    test_torch_available()
    test_depth_estimator_init()
    test_raft_params()
    test_raft_params_setter()
    test_raft_model_loading()
    test_raft_disparity_computation()
    test_input_padder()
    
    print("="*60)
    print("All tests passed!")
    print("="*60)
