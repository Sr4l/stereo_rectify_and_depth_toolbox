#!/usr/bin/env python3
"""
Generate example stereo image pairs for testing the calibration toolbox.

This script creates several synthetic stereo image pairs with known
camera parameters for testing rectification and depth estimation.
"""

import numpy as np
import cv2
import os


def create_checkerboard_pattern(width=640, height=480, square_size=40):
    """Create a checkerboard pattern image."""
    img = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                img[y:y+square_size, x:x+square_size] = 255
    
    return img


def create_stereo_pair_with_disparity(
    width=640,
    height=480,
    baseline=50,
    focal_length=500,
    object_distance=1000
):
    """
    Create a stereo pair with a simple geometric shape at known depth.
    
    Args:
        width: Image width
        height: Image height
        baseline: Distance between cameras in pixels
        focal_length: Focal length in pixels
        object_distance: Distance to object in pixels
    
    Returns:
        Tuple of (left_image, right_image, true_disparity)
    """
    left_img = np.ones((height, width), dtype=np.uint8) * 200
    right_img = np.ones((height, width), dtype=np.uint8) * 200
    
    true_disparity = np.zeros((height, width), dtype=np.float32)
    
    cx, cy = width // 2, height // 2
    radius = 100
    
    for y in range(height):
        for x in range(width):
            dx = x - cx
            dy = y - cy
            if dx*dx + dy*dy <= radius*radius:
                depth = object_distance - int(np.sqrt(radius*radius - dx*dx - dy*dy))
                disparity = (baseline * focal_length) / depth
                
                left_x = int(x + disparity / 2)
                right_x = int(x - disparity / 2)
                
                if 0 <= left_x < width:
                    left_img[y, left_x] = 100
                    true_disparity[y, left_x] = disparity
                
                if 0 <= right_x < width:
                    right_img[y, right_x] = 100
    
    left_img = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    
    return left_img, right_img, true_disparity


def create_concentric_circles_pair(width=640, height=480):
    """
    Create stereo pair with concentric circles pattern.
    Good for testing rectification accuracy.
    """
    def draw_circles(img, center_offset=0):
        img[:] = 255
        cx, cy = width // 2 + center_offset, height // 2
        
        for radius in range(20, min(width, height) // 2, 30):
            thickness = 10
            cv2.circle(img, (cx, cy), radius, (0, 0, 0), thickness)
        
        cv2.circle(img, (cx, cy), 15, (0, 0, 0), -1)
        
        for i in range(-200, 201, 50):
            cv2.line(img, (0, cy + i), (width, cy + i), (0, 0, 0), 2)
        
        return img
    
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    right_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    left_img = draw_circles(left_img, center_offset=-15)
    right_img = draw_circles(right_img, center_offset=15)
    
    return left_img, right_img


def create_random_dots_pair(width=640, height=480, num_dots=500):
    """
    Create stereo pair with random dot pattern (random dot stereogram).
    Excellent for testing stereo matching algorithms.
    """
    np.random.seed(42)
    
    left_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    right_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    true_disparity = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(num_dots):
        x = np.random.randint(50, width - 50)
        y = np.random.randint(50, height - 50)
        radius = np.random.randint(3, 8)
        color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
        
        depth_factor = np.random.uniform(0.5, 2.0)
        disparity = int(30 * depth_factor)
        
        cv2.circle(left_img, (x, y), radius, color, -1)
        cv2.circle(right_img, (x - disparity, y), radius, color, -1)
        
        if 0 <= x < width and 0 <= y < height:
            true_disparity[y, x] = disparity
    
    return left_img, right_img, true_disparity


def create_calibration_board_pair(width=640, height=480):
    """
    Create stereo pair simulating a calibration chessboard at an angle.
    """
    def draw_board(img, perspective_shift=0):
        img[:] = 255
        square_size = 40
        rows, cols = 9, 12
        
        start_x = 50 + perspective_shift
        start_y = 80
        
        for row in range(rows):
            for col in range(cols):
                x1 = start_x + col * square_size + int(row * perspective_shift * 0.3)
                y1 = start_y + row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                if (row + col) % 2 == 0:
                    pts = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2 + int(perspective_shift * 0.1), y2],
                        [x1 + int(perspective_shift * 0.1), y2]
                    ], np.int32)
                    cv2.fillPoly(img, [pts], (0, 0, 0))
        
        return img
    
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    right_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    left_img = draw_board(left_img, perspective_shift=-20)
    right_img = draw_board(right_img, perspective_shift=20)
    
    return left_img, right_img


def create_indoor_scene_pair(width=640, height=480):
    """
    Create a simple synthetic indoor scene with multiple depth planes.
    """
    left_img = np.zeros((height, width, 3), dtype=np.uint8)
    right_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    sky_color = (135, 206, 235)
    wall_color = (210, 180, 140)
    floor_color = (139, 90, 43)
    table_color = (101, 67, 33)
    object_color = (220, 20, 60)
    
    left_img[:height//3, :] = sky_color
    left_img[height//3:height//2, :] = wall_color
    left_img[height//2:, :] = floor_color
    
    right_img[:height//3, :] = sky_color
    right_img[height//3:height//2, :] = wall_color
    right_img[height//2:, :] = floor_color
    
    for y in range(height//2, height):
        shift = (y - height//2) // 20
        cv2.line(left_img, (0, y), (width, y), (120, 80, 30), 1)
        cv2.line(right_img, (0 + shift, y), (width + shift, y), (120, 80, 30), 1)
    
    table_y = height//2 + 50
    table_h = 150
    
    cv2.rectangle(left_img, (150, table_y), (490, table_y + table_h), table_color, -1)
    cv2.rectangle(right_img, (130, table_y), (470, table_y + table_h), table_color, -1)
    
    box_size = 60
    cv2.rectangle(left_img, (280, table_y - box_size), (360, table_y), object_color, -1)
    cv2.rectangle(right_img, (250, table_y - box_size), (330, table_y), object_color, -1)
    
    return left_img, right_img


def save_pair(output_dir, name, left_img, right_img):
    """Save a stereo pair to files."""
    left_path = os.path.join(output_dir, f"{name}_left.png")
    right_path = os.path.join(output_dir, f"{name}_right.png")
    
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)
    
    print(f"  Created: {name}_left.png and {name}_right.png")
    
    return left_path, right_path


def create_calibration_files(output_dir):
    """Create calibration parameter files for each example."""
    
    examples = {
        'sphere': {
            'left_K': [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            'right_K': [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            'left_dist': [0, 0, 0, 0, 0],
            'right_dist': [0, 0, 0, 0, 0],
            'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'T': [50, 0, 0]
        },
        'circles': {
            'left_K': [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            'right_K': [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            'left_dist': [0, 0, 0, 0, 0],
            'right_dist': [0, 0, 0, 0, 0],
            'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'T': [30, 0, 0]
        },
        'dots': {
            'left_K': [[550, 0, 320], [0, 550, 240], [0, 0, 1]],
            'right_K': [[550, 0, 320], [0, 550, 240], [0, 0, 1]],
            'left_dist': [0, 0, 0, 0, 0],
            'right_dist': [0, 0, 0, 0, 0],
            'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'T': [40, 0, 0]
        },
        'checkerboard': {
            'left_K': [[700, 0, 320], [0, 700, 240], [0, 0, 1]],
            'right_K': [[700, 0, 320], [0, 700, 240], [0, 0, 1]],
            'left_dist': [0, 0, 0, 0, 0],
            'right_dist': [0, 0, 0, 0, 0],
            'R': [[0.98, -0.1, 0], [0.1, 0.98, 0], [0, 0, 1]],
            'T': [35, 5, 0]
        },
        'indoor': {
            'left_K': [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            'right_K': [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            'left_dist': [0, 0, 0, 0, 0],
            'right_dist': [0, 0, 0, 0, 0],
            'R': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'T': [45, 0, 0]
        }
    }
    
    for name, params in examples.items():
        calib_file = os.path.join(output_dir, f"{name}_calibration.json")
        
        import json
        calibration = {
            'left_camera': {
                'K': params['left_K'],
                'distortion': params['left_dist']
            },
            'right_camera': {
                'K': params['right_K'],
                'distortion': params['right_dist'],
                'R': params['R'],
                'T': params['T']
            }
        }
        
        with open(calib_file, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"  Created: {name}_calibration.json")


def main():
    """Generate all example stereo pairs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    
    print("Generating example stereo image pairs...")
    print()
    
    pairs = [
        ('sphere', *create_stereo_pair_with_disparity()),
        ('circles', *create_concentric_circles_pair()),
        ('dots', *create_random_dots_pair()),
        ('checkerboard', *create_calibration_board_pair()),
        ('indoor', *create_indoor_scene_pair()),
    ]
    
    for name, left_img, right_img, *_ in pairs:
        print(f"Creating {name} stereo pair...")
        save_pair(output_dir, name, left_img, right_img)
    
    print()
    print("Creating calibration files...")
    create_calibration_files(output_dir)
    
    print()
    print("Example generation complete!")
    print(f"All files saved to: {output_dir}")
    print()
    print("Example files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith(('.png', '.json')):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
