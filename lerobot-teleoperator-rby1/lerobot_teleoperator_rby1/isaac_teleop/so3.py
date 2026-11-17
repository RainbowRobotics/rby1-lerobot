"""Moved to ``rby1_xr_teleop.so3``; this path is an alias of that module."""

import sys

import rby1_xr_teleop.so3 as _module

sys.modules[__name__] = _module
