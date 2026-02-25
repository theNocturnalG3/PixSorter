from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpan:
    start: int
    end: int


def _linmap(span: StageSpan, done: int, total: int) -> int:
    total = max(1, total)
    done = max(0, min(done, total))
    width = span.end - span.start
    return span.start + int(width * (done / total))


class ProgressPlan:
    """
    Centralized % mapping so worker code stays readable.
    Adjust spans here to tune perceived progress.
    """
    EMBED = StageSpan(1, 35)
    VERIFY = StageSpan(35, 75)
    WRITE = StageSpan(75, 82)
    BEST = StageSpan(82, 99)

    def embed_pct(self, done: int, total: int) -> int:
        return _linmap(self.EMBED, done, total)

    def verify_pct(self, done: int, total: int) -> int:
        return _linmap(self.VERIFY, done, total)

    def write_pct(self) -> int:
        return self.WRITE.end

    def best_pct(self, done: int, total: int) -> int:
        return _linmap(self.BEST, done, total)