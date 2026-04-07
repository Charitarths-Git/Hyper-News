"""
Password hashing helpers for backend-backed credentials auth.
"""

from __future__ import annotations

import hashlib
import secrets


_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        str(password or "").encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, raw_n, raw_r, raw_p, salt_hex, digest_hex = str(stored_hash or "").split("$", 5)
        if scheme != "scrypt":
            return False
        derived = hashlib.scrypt(
            str(password or "").encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(raw_n),
            r=int(raw_r),
            p=int(raw_p),
        ).hex()
        return secrets.compare_digest(derived, digest_hex)
    except Exception:
        return False
