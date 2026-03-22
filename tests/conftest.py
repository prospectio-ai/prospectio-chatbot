"""
Root conftest: import shared fixtures so they are available to all tests.
"""
import sys
import os

# Ensure tests package is importable
sys.path.insert(0, os.path.dirname(__file__))

from fixtures.external import *  # noqa: F401, F403
