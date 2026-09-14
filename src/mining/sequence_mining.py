"""
Sequential pattern mining for discovering high-success action patterns.
"""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

try:
    from prefixspan import PrefixSpan
except ImportError:
    PrefixSpan = None  # type: ignore[assignment]

from .features import build_action_sequences

logger = logging.getLogger(__name__)


class SequenceMiner:
    """Discover frequent sequential patterns and correlate them with success.

    Parameters
    ----------
    min_support : float
        Minimum fraction of sequences a pattern must appear in.
    min_length : int
        Minimum number of actions in a discovered pattern.
    max_length : int
        Maximum number of actions in a discovered pattern.
    """

    def __init__(
        self,
        min_support: float = 0.3,
        min_length: int = 2,
        max_length: int = 10,
    ) -> None:
        self.min_support = min_support
        self.min_length = min_length
        self.max_length = max_length
        self.patterns_: list[tuple[list[str], float]] = []
        self.success_patterns_: list[tuple[list[str], float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: list[list[str]],
        successes: Optional[list[bool]] = None,
    ) -> SequenceMiner:
        """Find frequent patterns and optionally rank by success correlation.

        Parameters
        ----------
        sequences : list of list of str
            Action sequences from different runs.
        successes : list of bool, optional
            Whether each corresponding sequence was successful.

        Returns
        -------
        SequenceMiner
            Self, for method chaining.
        """
        self.patterns_ = self._find_frequent_patterns(sequences)

        if successes is not None:
            if len(successes) != len(sequences):
                raise ValueError(
                    f"Length mismatch: {len(sequences)} sequences vs "
                    f"{len(successes)} success labels"
                )
            self.success_patterns_ = self._find_success_correlated(
                sequences, successes
            )
        else:
            self.success_patterns_ = []

        return self

    def get_top_patterns(self, n: int = 10) -> list[tuple[list[str], float]]:
        """Return top *n* patterns ranked by support (descending).

        Returns list of ``(pattern, support)`` tuples.
        """
        return sorted(
            self.patterns_, key=lambda p: p[1], reverse=True
        )[:n]

    def get_success_correlated(self, n: int = 10) -> list[tuple[list[str], float]]:
        """Return top *n* patterns ranked by success-lift (descending).

        Returns list of ``(pattern, lift)`` tuples.
        """
        return sorted(
            self.success_patterns_, key=lambda p: p[1], reverse=True
        )[:n]

    def predict_success(self, sequence: list[str]) -> float:
        """Estimate success probability for a new action sequence.

        The estimate is the fraction of success-correlated patterns that
        appear as subsequences of the given *sequence*.  Returns ``0.0``
        when no patterns have been learned.

        Parameters
        ----------
        sequence : list of str
            A single action sequence to evaluate.

        Returns
        -------
        float
            Estimated success probability in ``[0, 1]``.
        """
        if not self.success_patterns_:
            return 0.0

        hits = sum(
            1
            for pattern, _ in self.success_patterns_
            if _is_subsequence(pattern, sequence)
        )
        return hits / len(self.success_patterns_)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_frequent_patterns(
        self, sequences: list[list[str]]
    ) -> list[tuple[list[str], float]]:
        """Dispatch to PrefixSpan or n-gram fallback."""
        if not sequences:
            return []

        n_sequences = len(sequences)

        if PrefixSpan is not None:
            ps = PrefixSpan(sequences)
            min_count = max(1, int(np.ceil(self.min_support * n_sequences)))
            raw = ps.frequent(min_count)

            # prefixspan returns list of (count, pattern)
            patterns: list[tuple[list[str], float]] = []
            for count, pattern in raw:
                support = count / n_sequences
                length = len(pattern)
                if self.min_length <= length <= self.max_length:
                    patterns.append((list(pattern), support))
            return patterns

        # Fallback: simple n-gram counting
        return self._ngram_fallback(sequences)

    def _ngram_fallback(
        self, sequences: list[list[str]]
    ) -> list[tuple[list[str], float]]:
        """Find frequent contiguous subsequences via n-gram counting."""
        n_sequences = len(sequences)
        ngram_counts: Counter[tuple[str, ...]] = Counter()

        for seq in sequences:
            # Use a set per sequence so each pattern is counted at most once
            seen: set[tuple[str, ...]] = set()
            for length in range(self.min_length, min(self.max_length, len(seq)) + 1):
                for start in range(len(seq) - length + 1):
                    ngram = tuple(seq[start : start + length])
                    seen.add(ngram)
            for ngram in seen:
                ngram_counts[ngram] += 1

        min_count = max(1, int(np.ceil(self.min_support * n_sequences)))
        return [
            (list(ngram), count / n_sequences)
            for ngram, count in ngram_counts.items()
            if count >= min_count
        ]

    def _find_success_correlated(
        self,
        sequences: list[list[str]],
        successes: list[bool],
    ) -> list[tuple[list[str], float]]:
        """Compute lift for each pattern: P(pattern|success) / P(pattern)."""
        n_total = len(sequences)
        n_success = sum(successes)
        n_fail = n_total - n_success

        if n_success == 0 or n_fail == 0:
            logger.warning(
                "All runs share the same outcome; lift is undefined. "
                "Returning empty success patterns."
            )
            return []

        global_support: dict[tuple[str, ...], float] = {
            tuple(p): s for p, s in self.patterns_
        }

        # Count pattern occurrences per outcome group
        success_counts: Counter[tuple[str, ...]] = Counter()
        fail_counts: Counter[tuple[str, ...]] = Counter()

        for seq, is_success in zip(sequences, successes):
            seen: set[tuple[str, ...]] = set()
            for pattern_tuple in global_support:
                if _is_subsequence(list(pattern_tuple), seq):
                    seen.add(pattern_tuple)
            for p in seen:
                if is_success:
                    success_counts[p] += 1
                else:
                    fail_counts[p] += 1

        results: list[tuple[list[str], float]] = []
        for pattern_tuple, p_global in global_support.items():
            p_given_success = success_counts[pattern_tuple] / n_success
            if p_global > 0 and p_given_success > 0:
                lift = p_given_success / p_global
                # Only keep patterns that are more likely in successful runs
                if lift > 1.0:
                    results.append((list(pattern_tuple), float(lift)))

        return results


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _is_subsequence(pattern: list[str], sequence: list[str]) -> bool:
    """Check whether *pattern* appears as a (not necessarily contiguous)
    subsequence of *sequence*.
    """
    it = iter(sequence)
    return all(item in it for item in pattern)


def mine_frequent_sequences(
    sequences: list[list[str]],
    min_support: float = 0.3,
    min_length: int = 2,
) -> list[tuple[list[str], float]]:
    """Standalone convenience wrapper around :class:`SequenceMiner`.

    Parameters
    ----------
    sequences : list of list of str
        Action sequences.
    min_support : float
        Minimum support threshold.
    min_length : int
        Minimum pattern length.

    Returns
    -------
    list of (pattern, support)
        Discovered frequent patterns sorted by support descending.
    """
    miner = SequenceMiner(
        min_support=min_support,
        min_length=min_length,
    )
    miner.fit(sequences)
    return miner.get_top_patterns(n=len(miner.patterns_))
