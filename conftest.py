"""
conftest.py — pytest configuration.
Adds the project root to sys.path so `from src.xxx import ...` resolves
regardless of where pytest is invoked from.
"""
import sys
from pathlib import Path

# Project root = directory containing this conftest.py
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
