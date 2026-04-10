import re
from typing import Tuple

_CELL_RE = re.compile(r'^([A-Z][A-Z0-9]*)D(\d+)BWP16P90$')

TYPE_PREFIX_MAP = {
    "BUFF": 0,
    "AN2": 1, "AN3": 2, "AN4": 3,
    "OR2": 4, "OR3": 5, "OR4": 6,
    "AOI21": 7, "AOI22": 8,
    "AOI31": 9, "AOI32": 10, "AOI33": 11,
    "XOR2": 12, "XOR3": 13, "XOR4": 14,
    "INV": 15,
    "ND2": 16, "ND3": 17, "ND4": 18,
    "NR2": 19, "NR3": 20, "NR4": 21,
    "OAI21": 22, "OAI22": 23,
    "OAI31": 24, "OAI32": 25, "OAI33": 26,
    "XNR2": 27, "XNR3": 28, "XNR4": 29,
    "MUX2": 30, "MUX3": 31, "MUX4": 32,
}

_CANONICAL_DRIVES = [0, 1, 2, 3, 4, 6, 8]
_DRIVE_TO_SIZE_ID = {d: i for i, d in enumerate(_CANONICAL_DRIVES)}


def parse_cell_type(cell_type: str) -> Tuple[int, int]:
    """Parse a full library cell name into (type_id, size_id).

    Expects names of the form <PREFIX>D<DRIVE>BWP16P90, e.g.
    BUFFD0BWP16P90, ND2D4BWP16P90, AOI21D2BWP16P90.

    Returns (-1, -1) for any name that does not match the pattern or whose
    prefix / drive strength is not in the known set (e.g. flip-flops).
    """
    m = _CELL_RE.match(cell_type)
    if not m:
        return -1, -1
    prefix = m.group(1)
    drive = int(m.group(2))
    type_id = TYPE_PREFIX_MAP.get(prefix, -1)
    size_id = _DRIVE_TO_SIZE_ID.get(drive, -1)
    return type_id, size_id


def max_type_max_size() -> Tuple[int, int]:
    """Return (max_type_id, max_size_id) for embedding table sizing."""
    return max(TYPE_PREFIX_MAP.values()), len(_CANONICAL_DRIVES) - 1
