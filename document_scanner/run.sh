#!/bin/bash
# Document Scanner GUI Launcher

cd "$(dirname "$0")"
source venv/bin/activate
python scanner_gui.py
