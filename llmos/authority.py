"""Authority — who may grant a capability a process asks for.

The human ask-channel: when a process needs a capability it does not have (a
privileged or irreversible action), it emits a REQUEST instruction. The kernel
routes that to an Authority. In an interactive session the Authority is the human
(a decision box); headless, it is a policy. Approval *is* the capability grant.
"""
from __future__ import annotations


class Authority:
    def request(self, pcb, capability: str, reason: str) -> bool:
        raise NotImplementedError


class DenyAuthority(Authority):
    """Safe default when no human or policy is attached: grant nothing."""

    def request(self, pcb, capability, reason) -> bool:
        return False


class PolicyAuthority(Authority):
    """Headless / testing authority: grant a fixed allow-set, minus an explicit deny-set."""

    def __init__(self, grant=None, deny=None):
        self.grant = set(grant or [])
        self.deny = set(deny or [])

    def request(self, pcb, capability, reason) -> bool:
        if capability in self.deny:
            return False
        return capability in self.grant


class HumanAuthority(Authority):
    """The interactive binding: in a Cowork session this is where the kernel escalates
    to the human — it raises a decision box ('Process N requests <cap> because
    <reason> — grant?') and the human's approval is the grant. Wire the box via
    on_request(pcb, capability, reason) -> bool. With no human attached it prints the
    request and denies by default (fail safe)."""

    def __init__(self, on_request=None):
        self.on_request = on_request

    def request(self, pcb, capability, reason) -> bool:
        if self.on_request is not None:
            return bool(self.on_request(pcb, capability, reason))
        print(f"[human] process {pcb.pid} requests '{capability}' — {reason} "
              f"(no human attached; denying)")
        return False
