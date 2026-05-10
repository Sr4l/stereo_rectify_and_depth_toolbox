"""Check if RAFT-Stereo is available."""
import os

def check_raft_available():
    """
    Check if RAFT-Stereo can be used.
    
    Returns:
        tuple: (is_available: bool, reason: str)
    """
    try:
        import torch
        from core.depth import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            return False, "PyTorch not installed"
        
        # Check if submodule directory exists
        raft_path = os.path.join(os.path.dirname(__file__), '..', 'core', 'RAFT-Stereo')
        if not os.path.exists(raft_path):
            return False, "RAFT-Stereo submodule not found"
        
        # Check if submodule is properly initialized (has core/raft_stereo.py)
        raft_core_file = os.path.join(raft_path, 'core', 'raft_stereo.py')
        if not os.path.exists(raft_core_file):
            return False, "RAFT-Stereo submodule not initialized"
        
        return True, "Available"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, str(e)


def get_raft_fix_command():
    """Get the git command to fix RAFT-Stereo submodule issues."""
    return "git submodule update --init --recursive"
