#!/usr/bin/env python3
"""
Crypto AI Predictor - PyQt6 Entry Point

Launch the PyQt6-based GUI application.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui_pyqt.app import main

if __name__ == "__main__":
    main()
