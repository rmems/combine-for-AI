"""Git blob object ids. SHA-1 is git's format, not a security MAC."""

from __future__ import annotations

import hashlib


def git_blob_sha1(content: bytes) -> str:
    """Return ``sha1(\"blob \" + len + NUL + content)`` (git hash-object)."""
    header = b"blob " + str(len(content)).encode("ascii") + b"\0"
    return hashlib.sha1(header + content, usedforsecurity=False).hexdigest()
