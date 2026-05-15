# Stereo Rectification & Depth Toolbox

A Python GUI toolbox for testing and tuning stereo rectification parameters and evaluating the resulting disparity and depth calculations using OpenCV's **StereoBM**, **StereoSGBM**, and **RAFT-Stereo** (deep learning) algorithms.

![GUI Interface](asstes/GUI.png)

*Figure 1: Graphical user interface of the Stereo Rectification & Depth Toolbox showing camera parameter controls (left), rectified image display (center), and depth/disparity visualization with algorithm parameters (right).*

> ⚠️ **Warning**: The depth calculation functionality has **not been validated** for accuracy. The results produced by this toolbox, especially depth values in millimeters, should be considered **experimental and unverified**. Do not use this toolbox for applications requiring accurate depth measurements without independent validation.

## Purpose

This toolbox helps you:
- **Test stereo rectification parameters** (intrinsics, extrinsics, distortion)
- **Compare disparity/depth results** between StereoBM and StereoSGBM algorithms
- **Tune algorithm parameters** interactively with real-time feedback
- **Validate camera calibration** by visualizing rectified images and epipolar lines

## Features

- **Rectification Testing**: Load and adjust camera intrinsics (K matrix), distortion coefficients, and extrinsics (R, T) to test rectification quality
- **Algorithm Comparison**: Switch between three stereo matching algorithms:
  - **StereoBM** (Block Matching): Fast, suitable for texture-rich scenes
  - **StereoSGBM** (Semi-Global Block Matching): Slower but higher quality with better edge preservation
  - **RAFT-Stereo** (Deep Learning): State-of-the-art accuracy using neural networks (requires PyTorch)
- **Interactive Parameter Tuning**: Adjust algorithm parameters with real-time disparity/depth preview
- **Depth/Disparity Visualization**: Toggle between disparity (pixels) and depth (mm) views
- **Multiple Colormaps**: JET, VIRIDIS, MAGMA, INFERNO, PLASMA, CIVIDIS
- **Epipolar Line Visualization**: Verify rectification accuracy by checking epipolar line alignment
- **Import/Export**: Save and load calibration files in JSON format

## Requirements

- Python 3.11 or higher
- OpenCV 4.10.0 or higher (required for NumPy 2.x compatibility)
- NumPy 2.0.0 or higher
- Pillow 8.0.0 or higher
- tkinter (standard library, may need separate installation on Linux)

### Optional (for RAFT-Stereo deep learning support)

- PyTorch 1.7.0 or higher
- Torchvision 0.8.1 or higher
- gdown (for model download)
- Pretrained RAFT-Stereo model (~45MB)
- GPU recommended for faster inference (CUDA support)

## Installation

### Linux (Ubuntu/Debian)

1. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-venv python3-tk
   ```

2. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd stereo_rectify_and_depth_toolbox
   ```

3. **Initialize RAFT-Stereo submodule:**
   ```bash
   git submodule update --init --recursive
   ```

4. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

4. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

5. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6. **Optional: Install PyTorch for RAFT-Stereo support**
    ```bash
    # CPU-only version (slower inference, smaller in size, no CUDA or NVIDIA GPU needed)
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # CPU & GPU version (recommended, bigger download, uses CUDA if possible)
    # Visit https://pytorch.org for CUDA-specific installation commands
    pip install torch torchvision
    ```

7. **Optional: Download pretrained RAFT-Stereo model**
    ```bash
    # Download recommended Middlebury model
    python scripts/download_raft_models.py --model middlebury
    
    # Available models: middlebury (recommended), eth3d, sceneflow, realtime
    ```

8. **Run the application:**
    ```bash
    python main.py
    ```

### Windows

1. **Install Python:**
   - Download Python 3.11+ from https://www.python.org/downloads/
   - During installation, check "Add Python to PATH"

2. **Clone the repository:**
   ```cmd
   git clone <repository-url>
   cd stereo_rectify_and_depth_toolbox
   ```

3. **Initialize RAFT-Stereo submodule:**
   ```cmd
   git submodule update --init --recursive
   ```

4. **Create virtual environment:**
   ```cmd
   py -3.11 -m venv venv
   ```

4. **Activate virtual environment:**
   ```cmd
   venv\Scripts\activate
   ```

5. **Install Python dependencies:**
    ```cmd
    pip install -r requirements.txt
    ```

6. **Optional: Install PyTorch for RAFT-Stereo support**
    ```cmd
    # CPU-only version (slower inference, smaller in size, no CUDA or NVIDIA GPU needed)
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # CPU & GPU version (recommended, bigger download, uses CUDA if possible)
    # Visit https://pytorch.org for CUDA-specific installation commands
    pip install torch torchvision
    ```

7. **Optional: Download pretrained RAFT-Stereo model**
    ```cmd
    python scripts\download_raft_models.py --model middlebury
    ```

8. **Run the application:**
    ```cmd
    python main.py
    ```

### macOS

1. **Install Python 3.11 and tkinter:**
   ```bash
   brew install python@3.11
   ```

2. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd stereo_rectify_and_depth_toolbox
   ```

3. **Initialize RAFT-Stereo submodule:**
   ```bash
   git submodule update --init --recursive
   ```

4. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Optional: Install PyTorch for RAFT-Stereo support**
    ```bash
    # CPU-only version (slower inference, smaller in size, no CUDA or NVIDIA GPU needed)
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # CPU & GPU version (recommended, bigger download, uses CUDA if possible)
    # Visit https://pytorch.org for CUDA-specific installation commands
    pip install torch torchvision
    ```

6. **Optional: Download pretrained RAFT-Stereo model**
    ```bash
    python scripts/download_raft_models.py --model middlebury
    ```

7. **Run the application:**
    ```bash
    python main.py
    ```

## Usage

### Quick Start

```bash
# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Run the toolbox
python main.py
```

### Workflow

#### Quick Test with Example Images

The toolbox includes example datasets in the `examples/` folder. To quickly test with the Tsukuba dataset:

1. **Start the toolbox:**
   ```bash
   python main.py
   ```

2. **Load Tsukuba example images:**
   - Click **Load Left Image** → navigate to `examples/tsukuba_left.png`
   - Click **Load Right Image** → navigate to `examples/tsukuba_right.png`

3. **Load Tsukuba calibration:**
   - Click **Load Calibration** → select `examples/tsukuba_calibration.json`
   - The rectified images should appear immediately

4. **Test depth estimation:**
   - Select **SGBM** algorithm for best quality
   - Adjust parameters if needed (default values work well for Tsukuba)
   - Toggle between **disparity** and **depth (mm)** views
   - Try different colormaps (VIRIDIS or JET work well)

#### Using RAFT-Stereo (Deep Learning)

If you have PyTorch installed and downloaded a pretrained model:

1. **Select RAFT algorithm:**
   - Choose **RAFT** from the algorithm dropdown
   - The RAFT-Stereo Parameters panel will appear

2. **Configure RAFT parameters:**
   - **Iters**: Number of refinement iterations (default: 32, higher = more accurate but slower)
   - **Downsample**: Resolution factor (default: 2, lower = higher resolution but more memory)

3. **Run inference:**
   - RAFT-Stereo will automatically compute disparity when you select it
   - First run may take time as the model loads
   - GPU acceleration is automatically used if CUDA is available

4. **Compare results:**
   - Switch between BM, SGBM, and RAFT to compare quality
   - RAFT typically produces the best results on challenging scenes

#### Full Workflow with Your Images

1. **Load stereo image pair:**
   - Click **Load Left Image** to load the left camera image
   - Click **Load Right Image** to load the right camera image

2. **Test rectification parameters:**
   - **Option A**: Load existing calibration via **Load Calibration** (JSON file)
   - **Option B**: Manually enter intrinsics, distortion, and extrinsics
   - Verify rectification by checking that corresponding points align horizontally
   - Enable **Show Epipolar Lines** to verify alignment (lines should be horizontal)

3. **Compare stereo algorithms:**
   - Select **BM** or **SGBM** from the algorithm dropdown
   - Adjust algorithm-specific parameters using the sliders
   - Observe real-time changes in the disparity/depth visualization

4. **Evaluate depth results:**
   - Toggle between **disparity** (pixel shift) and **depth (mm)** views
   - Hover over the depth map to see values at specific pixels
   - Try different colormaps for better visualization
   - Compare BM vs SGBM results by switching algorithms

### Key Differences: BM vs SGBM vs RAFT-Stereo

| Aspect | StereoBM | StereoSGBM | RAFT-Stereo |
|--------|----------|------------|-------------|
| **Speed** | Fast (real-time) | Slower (2-3x BM) | Slow on CPU, Fast on GPU |
| **Quality** | Good for textured scenes | Better edge preservation | State-of-the-art accuracy |
| **Parameters** | 6 tunable parameters | 9 tunable parameters | 2 parameters (iters, downsample) |
| **Requirements** | OpenCV only | OpenCV only | PyTorch + pretrained model |
| **Best for** | Quick testing, texture-rich scenes | High-quality depth, smooth surfaces | Challenging scenes, best accuracy |
| **Hardware** | CPU | CPU | GPU recommended |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Load left image |
| `Ctrl+R` | Load right image |
| `Ctrl+S` | Save rectified images |
| `F5` | Refresh rectification |

## Calibration File Format

Calibration files are saved as JSON:

```json
{
  "left_camera": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion": [k1, k2, p1, p2, k3]
  },
  "right_camera": {
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "distortion": [k1, k2, p1, p2, k3],
    "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
    "T": [tx, ty, tz]
  }
}
```

## Algorithm Parameters

### StereoBM Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Num Disp | Number of disparities (must be multiple of 16) | 16-256 |
| Block | Block size for matching (must be odd) | 5-255 |
| Min Disp | Minimum disparity | -100 to 100 |
| Unique | Uniqueness ratio | 1-100 |
| Speckle W | Speckle window size | 0-200 |
| Speckle R | Speckle range | 0-50 |

### StereoSGBM Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Num Disp | Number of disparities (must be multiple of 16) | 16-256 |
| Block | Block size for matching (must be odd) | 5-255 |
| Min Disp | Minimum disparity | -100 to 100 |
| Unique | Uniqueness ratio | 1-100 |
| P1 | Smoothness constraint for similar colors | 0-1000 |
| P2 | Smoothness constraint for edges | 0-2000 |
| PreFilter | Pre-filter cap | 0-100 |
| Speckle W | Speckle window size | 0-200 |
| Speckle R | Speckle range | 0-50 |

### RAFT-Stereo Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| Iters | Number of refinement iterations | 1-64 | 32 |
| Downsample | Resolution factor (1/2^K) | 1-3 | 2 |

**Notes:**
- Higher iterations improve accuracy but increase computation time
- Lower downsample values give higher resolution but require more GPU memory
- Default settings work well for most applications
- Model selection affects performance (see model descriptions below)

### RAFT-Stereo Models

| Model | Best For | Description |
|-------|----------|-------------|
| **middlebury** | General purpose | Trained on Middlebury dataset, best for in-the-wild images (RECOMMENDED) |
| **eth3d** | High resolution | Trained on ETH3D dataset, optimized for high-resolution stereo |
| **sceneflow** | Synthetic scenes | Trained on SceneFlow (FlyingThings3D, Driving, Monkaa) |
| **realtime** | Speed | Optimized for real-time applications, lower accuracy |

Download models using:
```bash
python scripts/download_raft_models.py --model <model_name>
```

## Testing

Run the test suite to verify the toolbox functionality:

```bash
# Activate virtual environment first
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Run all tests (core functionality)
python tests/test_all.py

# Run GUI tests (requires X11 display)
python tests/test_gui.py

# Run integration tests
python tests/test_integration.py
```

## Architecture

```
stereo_rectify_and_depth_toolbox/
├── main.py              # Entry point
├── core/
│   ├── rectifier.py     # Stereo rectification (OpenCV)
│   ├── depth.py         # Depth estimation (BM, SGBM & RAFT-Stereo)
│   └── RAFT-Stereo/     # RAFT-Stereo deep learning model
├── gui/
│   ├── main_window.py   # Main application window
│   ├── param_panel.py   # Camera parameter inputs
│   └── image_panel.py   # Image display with zoom/pan
├── scripts/
│   └── download_raft_models.py  # Model download utility
├── tests/
│   ├── test_all.py      # Core tests
│   ├── test_gui.py      # GUI tests
│   └── test_integration.py
├── examples/            # Sample images and calibrations
├── models/              # Pretrained RAFT-Stereo models
└── requirements.txt     # Python dependencies
```

## Example Data

The included example datasets use images from the **Tsukuba Stereo Dataset** provided by CVLab, University of Tsukuba.

- **Source**: [CVLab Tsukuba Stereo Dataset](https://home.cvlab.cs.tsukuba.ac.jp/dataset)
- **License**: The dataset is provided for research and educational purposes
- **Usage**: The examples in this toolbox use a subset of the Tsukuba dataset for demonstration purposes

For more datasets and benchmark evaluations, visit the CVLab Tsukuba website.

## Troubleshooting

### Linux: tkinter not found
```bash
sudo apt install python3-tk
```

### Windows: tkinter not found
Reinstall Python and ensure "tcl/tk and IDLE" is checked during installation.

### macOS: tkinter not found
```bash
brew install python-tk
```

### RAFT-Stereo: PyTorch not installed
If you select RAFT algorithm and see an error about PyTorch:

```bash
# Install PyTorch (CPU version, smaller, slower, no NVidia GPU needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch (GPU version, recommended)
pip install torch torchvision
```

After installation, restart the application and RAFT will be available.

### RAFT-Stereo: Model not found
If you see "Model Not Found" error:

```bash
# Download the recommended Middlebury model
python scripts/download_raft_models.py --model middlebury

# Or download manually from Google Drive and place in models/ directory
```

### RAFT-Stereo: Out of memory (GPU)
If you run out of GPU memory:

- Reduce **Downsample** parameter (try 3 instead of 2)
- Reduce image resolution before loading
- Use CPU mode (slower but works with large images)
- Close other GPU applications

### RAFT-Stereo: Slow inference on CPU
RAFT-Stereo is designed for GPU acceleration. For better performance:

- Install CUDA-enabled PyTorch (requires NVIDIA GPU)
- Use the **realtime** model for faster inference
- Increase **Downsample** parameter to reduce resolution
- Consider using BM or SGBM for CPU-only systems

### "Invalid camera parameters" error
- Ensure focal lengths (fx, fy) are positive
- Principal point (cx, cy) should be within image bounds
- Check that rotation matrix is valid (orthogonal)

### Poor rectification (misaligned images)
- Verify calibration parameters are accurate
- Check that rotation matrix R is orthogonal
- Ensure translation vector T represents the baseline correctly
- Use epipolar lines to diagnose: they should be horizontal and aligned

### Poor depth quality
- **Try SGBM** instead of BM for better edge preservation
- **Increase P1 and P2** (SGBM) for smoother results on uniform surfaces
- **Adjust Block Size**: larger = smoother, smaller = more detail
- **Increase Uniqueness Ratio** to filter ambiguous matches
- **Enable speckle filtering** (Speckle W > 0) to remove noise
- Ensure images are **well-textured** (textureless regions fail)

### Disparity map is mostly black
- Check that **Num Disparities** is appropriate for your baseline
- Adjust **Min Disparity** if objects are far away
- Verify that rectification is working (images should be horizontally aligned)

### RAFT-Stereo produces all zeros or constant values
This was a bug in earlier versions. Make sure you have the latest version where the disparity sign is correctly handled. The RAFT model outputs optical flow (negative values), which must be negated to get positive disparity values.

## License

This toolbox is released under the BSD 3-Clause License.

### RAFT-Stereo License
This toolbox incorporates RAFT-Stereo code and pretrained models from the Princeton Vision & Learning Lab. We thank the authors for making their excellent work available to the research community.

RAFT-Stereo is released under the MIT License.

**Source:** Princeton Vision & Learning Lab  
**Repository:** https://github.com/princeton-vl/RAFT-Stereo.git  
**Paper:** "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching" (3DV 2021, Best Student Paper Award)  
**ArXiv:** https://arxiv.org/pdf/2109.07547.pdf

### Pretrained Models
The pretrained RAFT-Stereo models are provided by Princeton Vision & Learning Lab for research and educational purposes.

| Model | File | Description | Best Use Case |
|-------|------|-------------|---------------|
| **Middlebury** | `raftstereo-middlebury.pth` | Trained on Middlebury dataset | General purpose, in-the-wild images (RECOMMENDED) |
| **ETH3D** | `raftstereo-eth3d.pth` | Trained on ETH3D dataset | High-resolution stereo imagery |
| **SceneFlow** | `raftstereo-sceneflow.pth` | Trained on SceneFlow (FlyingThings3D, Driving, Monkaa) | Synthetic scenes, research |
| **Realtime** | `raftstereo-realtime.pth` | Optimized for speed | Real-time applications, lower latency requirements |

**Download:** Models can be downloaded using the included script:
```bash
python scripts/download_raft_models.py --model <model_name>
```

Or manually from the official Google Drive folder:  
https://drive.google.com/drive/folders/1booUFYEXmsdombVuglatP0nZXb5qI89J

**Model Usage:**
- All models are provided under the MIT License
- For best accuracy, use the **middlebury** model

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting:

```bash
python tests/test_all.py
```
