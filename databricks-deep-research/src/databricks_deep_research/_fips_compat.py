"""FIPS compatibility: patch hashlib.md5 to default usedforsecurity=False.

On FIPS-enabled kernels, hashlib.md5() raises ValueError because MD5 is
banned for security purposes. Third-party libraries (e.g. dateparser via
trafilatura) call md5() for non-security purposes like cache keys. This
patch makes those calls safe by defaulting usedforsecurity=False.

Import this module as a side-effect before any library that uses md5().
"""
import hashlib
from typing import Any

_original_md5 = hashlib.md5


def _fips_md5(*args: Any, **kwargs: Any) -> "hashlib._Hash":
    if "usedforsecurity" not in kwargs:
        kwargs["usedforsecurity"] = False
    return _original_md5(*args, **kwargs)


hashlib.md5 = _fips_md5
