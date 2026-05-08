"""Telemetry side-channels for carl-studio.

This subpackage hosts opt-in telemetry forwarders that emit observability
events to carl.camp. Privacy-first: every forwarder honors
``consent.telemetry`` and respects ``AXON_FORWARD_DISABLED`` env opt-out.

The default-OFF doctrine matches the rest of the consent-gated subsystem
in :mod:`carl_studio.consent` — nothing leaves the local process unless
the user has explicitly opted in AND the user has not set the kill-switch.
"""

from __future__ import annotations

# Re-export the public surface for convenience.
from carl_studio.telemetry.axon import (
    AxonEvent,
    AxonForwarder,
    install_default_forwarder,
    reset_default_forwarder,
)

__all__ = [
    "AxonEvent",
    "AxonForwarder",
    "install_default_forwarder",
    "reset_default_forwarder",
]
