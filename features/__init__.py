# ============================================================== #
#  Module:      features/__init__.py
#  Description: Features package initialiser
#  Author:      Siya Jethliya
#  Copyright (c) 2026 SciVizAI — All rights reserved.
# ============================================================== #

"""
Feature computation module for MD trajectories.
"""
from .compute_md_features import compute_features

__all__ = ['compute_features']
