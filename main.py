#!/usr/bin/env python3
"""Stereo Camera Calibration & Depth Toolbox - Main Entry Point

This application provides stereo camera rectification and depth estimation
using three algorithms: StereoBM, StereoSGBM, and RAFT-Stereo.

Usage:
    python main.py
"""

import sys


def main():
    """Main entry point."""
    from gui.qt_main_window import StereoCalibrationGUI
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for consistent dark appearance
    app.setApplicationName("Stereo Camera Calibration & Depth Toolbox")

    window = StereoCalibrationGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()