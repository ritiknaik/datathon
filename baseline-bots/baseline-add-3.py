"""baseline-add-3.py - stronger practice opponent 3 (see opponents.py). Port 5009 by default."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from addbase import serve

if __name__ == "__main__":
    serve("add3")
