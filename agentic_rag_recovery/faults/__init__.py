"""Fault-injection framework."""

from .definitions import ALL_FAULTS, FaultDefinition, FaultRegistry
from .injector import FaultInjector

__all__ = ["FaultInjector", "FaultRegistry", "FaultDefinition", "ALL_FAULTS"]