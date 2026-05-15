import torch.nn.functional as F


class InputPadder:
    """Pads images to be divisible by divis_by."""
    
    def __init__(self, shape, divis_by=32):
        """
        Initialize padder.
        
        Args:
            shape: Tensor shape (B, C, H, W)
            divis_by: Divisibility requirement (default: 32)
        """
        self.batch_size, _, ht, wd = shape
        self.ht = ht
        self.wd = wd
        
        pad_ht = ((ht // divis_by) + 1) * divis_by - ht
        pad_wd = ((wd // divis_by) + 1) * divis_by - wd
        
        self.padding = [
            pad_wd // 2, pad_wd - pad_wd // 2,
            pad_ht // 2, pad_ht - pad_ht // 2
        ]
    
    def pad(self, *inputs):
        """Pad input tensors."""
        return [F.pad(x, self.padding, mode='replicate') for x in inputs]
    
    def unpad(self, x):
        """Remove padding from tensor."""
        ht, wd = self.ht, self.wd
        return x[..., :ht, :wd]
