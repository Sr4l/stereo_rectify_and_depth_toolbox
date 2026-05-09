#!/usr/bin/env python3
"""
Stereo Camera Calibration & Depth Toolbox

A GUI application for interactive stereo camera calibration,
rectification, and depth estimation using OpenCV.

Usage:
    python main.py

Requirements:
    opencv-python>=4.5.0
    numpy>=1.19.0
    Pillow>=8.0.0
"""

from gui.main_window import StereoCalibrationGUI


def main():
    """Main entry point for the application."""
    app = StereoCalibrationGUI()
    app.run()


if __name__ == '__main__':
    main()
