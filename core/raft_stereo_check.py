"""Check if RAFT-Stereo is available."""

def check_raft_available():
    """
    Check if RAFT-Stereo can be used.
    
    Returns:
        bool: True if PyTorch is installed and RAFT-Stereo can be used
    """
    try:
        import torch
        from core.depth import TORCH_AVAILABLE
        return TORCH_AVAILABLE
    except ImportError:
        return False
