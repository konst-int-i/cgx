"""
Main package for column generation solver. Note that the contents
of the beam_search and binarizer files are based on the IBM AIX360 implementation
https://github.com/Trusted-AI/AIX360
We only use and modify some elements of the original repository, which is why we
separated it out into this implementation
"""
from cgx.explain.binarizer import FeatureBinarizer
from cgx.explain.boolean_rule_cg import BooleanRuleCG
from cgx.explain.column_generation import column_generation_rules
from cgx.explain.decompositional import extract_cg_decompositional

__all__ = [
    "FeatureBinarizer",
    "BooleanRuleCG",
    "column_generation_rules",
    "extract_cg_decompositional"
]