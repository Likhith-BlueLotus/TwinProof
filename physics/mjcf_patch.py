"""Whitelisted MJCF patcher (master doc §2.6). Anything off the whitelist is refused; G7 relies on this."""
from __future__ import annotations

import xml.etree.ElementTree as ET

WHITELIST: dict[str, frozenset[str]] = {
    "joint": frozenset({"pos", "damping", "frictionloss", "armature"}),
    "geom": frozenset({"friction", "solref", "solimp", "mass"}),
    "inertial": frozenset({"mass", "diaginertia"}),
}  # no actuator fields: gear or kp would change the authority G3 bounds against ctrlrange


class PatchRejected(ValueError):
    pass


def apply_patch(base_xml: str, patch: dict[str, str]) -> str:
    """Patch keys are "<element path>@<attr>", e.g. ".//joint[@name='hinge']@damping"."""
    root = ET.fromstring(base_xml)
    for key, value in patch.items():
        path, sep, attr = key.rpartition("@")
        if not sep or not path:
            raise PatchRejected(f"malformed patch key {key!r}")
        nodes = root.findall(path)
        if len(nodes) != 1:
            raise PatchRejected(f"{path!r} matches {len(nodes)} elements, need exactly 1")
        if attr not in WHITELIST.get(nodes[0].tag, frozenset()):
            raise PatchRejected(f"<{nodes[0].tag} {attr}> is not a whitelisted field")
        nodes[0].set(attr, str(value))
    return ET.tostring(root, encoding="unicode")
