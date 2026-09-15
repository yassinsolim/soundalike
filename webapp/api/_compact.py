"""Compact, row-aligned UTF-8 metadata without changing the release index."""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Union

import numpy as np


class CompactStrings(Sequence):
    """String sequence backed by one UTF-8 buffer and byte offsets."""

    __slots__ = ("_buf", "_off", "_n")

    def __init__(self, buf: np.ndarray, off: np.ndarray):
        self._buf = np.asarray(buf, dtype=np.uint8)
        self._off = np.asarray(off)
        if self._off.ndim != 1 or self._off.size < 1:
            raise ValueError("offset array must be 1-D with at least one entry")
        self._n = int(self._off.size) - 1

    @classmethod
    def from_strings(cls, values: Iterable[str]) -> "CompactStrings":
        encoded = [str(value).encode("utf-8") for value in values]
        total = sum(len(value) for value in encoded)
        dtype = np.int32 if total < 2 ** 31 - 1 else np.int64
        off = np.zeros(len(encoded) + 1, dtype=dtype)
        buf = np.empty(total, dtype=np.uint8)
        pos = 0
        for i, value in enumerate(encoded):
            size = len(value)
            if size:
                buf[pos:pos + size] = np.frombuffer(value, dtype=np.uint8)
            pos += size
            off[i + 1] = pos
        return cls(buf, off)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key: Union[int, np.integer, slice]):
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += self._n
            if not 0 <= i < self._n:
                raise IndexError(f"index {key} out of range for {self._n} rows")
            start, end = int(self._off[i]), int(self._off[i + 1])
            return self._buf[start:end].tobytes().decode("utf-8")
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(self._n))]
        raise TypeError(f"unsupported index type {type(key)!r}")

    def __iter__(self) -> Iterator[str]:
        buf, off = self._buf, self._off
        for i in range(self._n):
            yield buf[int(off[i]):int(off[i + 1])].tobytes().decode("utf-8")

    @property
    def nbytes(self) -> int:
        return int(self._buf.nbytes + self._off.nbytes)

    def tolist(self) -> List[str]:
        return list(self)

    def __repr__(self) -> str:
        return f"CompactStrings(rows={self._n}, bytes={self.nbytes})"
