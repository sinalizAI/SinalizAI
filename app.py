#!/usr/bin/env python3
"""
TCC - SinalizAI Application
Main application entry point following MVC architecture
"""

import sys
import os

# Add app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import main application
from main import main

if __name__ == '__main__':
    main()