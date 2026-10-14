"""Tier-1 deterministic detection."""

from .rules import RULES, RuleContext, RuleDetector, RuleMatch

__all__ = ["RuleDetector", "RuleMatch", "RuleContext", "RULES"]