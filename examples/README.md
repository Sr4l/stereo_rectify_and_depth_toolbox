# Example Stereo Image Pairs

This folder contains synthetic stereo image pairs for testing the Stereo Camera Calibration & Depth Toolbox.

## Example Sets

### 1. Sphere (`sphere_*.png`)
- **Description**: A 3D sphere with smooth depth variation
- **Use case**: Testing depth estimation with continuous depth changes
- **Calibration**: Frontal parallel cameras, baseline=50px
- **Expected result**: Disparity increases toward sphere center

### 2. Concentric Circles (`circles_*.png`)
- **Description**: Pattern of concentric circles with horizontal lines
- **Use case**: Verifying rectification accuracy (epipolar lines should align)
- **Calibration**: Frontal parallel cameras, baseline=30px
- **Expected result**: Perfect horizontal epipolar lines after rectification

### 3. Random Dots (`dots_*.png`)
- **Description**: Random dot stereogram with varying disparities
- **Use case**: Testing stereo matching algorithms (StereoBM)
- **Calibration**: Frontal parallel cameras, baseline=40px
- **Expected result**: Discrete disparity values at dot locations

### 4. Checkerboard (`checkerboard_*.png`)
- **Description**: Chessboard pattern viewed from an angle
- **Use case**: Testing rectification with perspective distortion
- **Calibration**: Rotated cameras (10° yaw), baseline=35px
- **Expected result**: Rectified images should align the checkerboard squares

### 5. Indoor Scene (`indoor_*.png`)
- **Description**: Simple synthetic room with floor, wall, table, and object
- **Use case**: Testing with multiple depth planes and occlusions
- **Calibration**: Frontal parallel cameras, baseline=45px
- **Expected result**: Distinct depth layers visible in depth map

## Calibration Files

Each example has a corresponding `*_calibration.json` file containing:
- Intrinsic matrix (K) for both cameras
- Distortion coefficients (all zeros for synthetic images)
- Rotation matrix (R) between cameras
- Translation vector (T) between cameras

## Usage

### Load images in the GUI:
1. Start the application: `python main.py`
2. Click "Load Left Image" and select `*_left.png`
3. Click "Load Right Image" and select `*_right.png`
4. Click "Load Calibration" and select `*_calibration.json`
5. The rectified images and depth map will update automatically

### Recommended parameters for StereoBM:
- **sphere**: numDisparities=16, blockSize=9
- **circles**: numDisparities=32, blockSize=15
- **dots**: numDisparities=16, blockSize=5
- **checkerboard**: numDisparities=32, blockSize=11
- **indoor**: numDisparities=48, blockSize=11

## Creating Your Own Examples

You can modify `generate_examples.py` to create custom test patterns. The script uses:
- OpenCV for image generation
- Known camera geometry for ground truth
- Various depth configurations for comprehensive testing

## Notes

- All images are 640x480 pixels
- Synthetic images have no noise or lens distortion
- Real-world images will require proper camera calibration
- Use these examples to verify the toolbox is working correctly before testing with real data
