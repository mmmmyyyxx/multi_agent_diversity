"""Compatibility names; new execution imports current_layer1 directly."""
from .legacy.pattern_layer1 import *  # noqa: F401,F403
from .current_layer1 import (GradientPatternLayer1Config, GradientPatternLocalTask, gradient_pattern_input, GradientPatternMemoryOptimizer, GradientPatternMemoryEngine)
