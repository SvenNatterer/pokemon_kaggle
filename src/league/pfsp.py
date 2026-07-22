"""Small, training-only helpers for PFSP-lite opponent sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


def _bounded_distribution(
    values: Sequence[float],
    *,
    lower: float,
    upper: float,
) -> list[float]:
    """Project positive scores onto a probability simplex with soft bounds."""
    count = len(values)
    if count == 0:
        raise ValueError("At least one opponent is required")

    lower = max(0.0, min(float(lower), 1.0 / count))
    upper = max(1.0 / count, min(float(upper), 1.0))
    scores = [max(0.0, float(value)) for value in values]
    result = [lower] * count
    remaining = 1.0 - count * lower
    active = set(range(count))
    while active and remaining > 1e-12:
        active_total = sum(scores[index] for index in active)
        if active_total <= 0.0:
            additions = {index: remaining / len(active) for index in active}
        else:
            additions = {
                index: remaining * scores[index] / active_total
                for index in active
            }
        capped = [
            index
            for index, addition in additions.items()
            if result[index] + addition > upper + 1e-12
        ]
        if not capped:
            for index, addition in additions.items():
                result[index] += addition
            remaining = 0.0
            break
        for index in capped:
            capacity = upper - result[index]
            result[index] = upper
            remaining -= capacity
            active.remove(index)

    correction = 1.0 - sum(result)
    if abs(correction) > 1e-12:
        adjustable = [
            index
            for index, value in enumerate(result)
            if lower - 1e-12 <= value + correction <= upper + 1e-12
        ]
        result[adjustable[0] if adjustable else 0] += correction
    return result


@dataclass
class OpponentRecord:
    games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def effective_wins(self) -> float:
        return self.wins + 0.5 * self.draws

    def posterior(self, prior_games: float) -> tuple[float, float]:
        alpha = 0.5 * prior_games + self.effective_wins
        beta = 0.5 * prior_games + self.losses + 0.5 * self.draws
        total = alpha + beta
        mean = alpha / total
        variance = alpha * beta / (total * total * (total + 1.0))
        return mean, math.sqrt(max(0.0, variance))

    def add(self, outcome: int) -> None:
        if outcome > 0:
            self.wins += 1
        elif outcome < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.games += 1


@dataclass
class PFSPLite:
    labels: Sequence[str]
    initial_weights: Sequence[float]
    prior_games: float = 4.0
    random_fraction: float = 0.20
    max_probability: float = 0.35
    records: dict[str, OpponentRecord] = field(init=False)
    current_probabilities: list[float] = field(init=False)
    segment_records: dict[str, OpponentRecord] = field(init=False)
    completed_segments: int = 0

    def __post_init__(self) -> None:
        self.labels = [str(label) for label in self.labels]
        if not self.labels or len(self.labels) != len(self.initial_weights):
            raise ValueError("Labels and initial weights must have the same non-zero length")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("PFSP-lite requires unique opponent labels")
        if self.prior_games <= 0.0:
            raise ValueError("prior_games must be positive")
        if not 0.0 <= self.random_fraction <= 1.0:
            raise ValueError("random_fraction must be between 0 and 1")
        if not 0.0 < self.max_probability <= 1.0:
            raise ValueError("max_probability must be between 0 and 1")

        self.records = {label: OpponentRecord() for label in self.labels}
        self.segment_records = {label: OpponentRecord() for label in self.labels}
        self.current_probabilities = _bounded_distribution(
            self.initial_weights,
            lower=0.0,
            upper=1.0,
        )

    @property
    def segment_games(self) -> int:
        return sum(record.games for record in self.segment_records.values())

    def observe(self, label: str, outcome: int) -> bool:
        if label not in self.records or outcome not in {-1, 0, 1}:
            return False
        self.records[label].add(outcome)
        self.segment_records[label].add(outcome)
        return True

    def restore(self, summary: dict) -> None:
        """Restore cumulative PFSP state while starting a fresh pending segment."""
        probabilities = summary.get("probabilities") or {}
        opponents = summary.get("opponents") or {}
        expected = set(self.labels)
        if set(probabilities) != expected or set(opponents) != expected:
            raise ValueError("PFSP state labels do not match the configured opponent pool")

        restored_probabilities = [float(probabilities[label]) for label in self.labels]
        if (
            any(not math.isfinite(value) or value < 0.0 for value in restored_probabilities)
            or sum(restored_probabilities) <= 0.0
        ):
            raise ValueError("PFSP state contains invalid probabilities")
        self.current_probabilities = _bounded_distribution(
            restored_probabilities,
            lower=0.0,
            upper=1.0,
        )

        restored_records = {}
        for label in self.labels:
            payload = opponents[label]
            record = OpponentRecord(
                games=int(payload.get("games", 0)),
                wins=int(payload.get("wins", 0)),
                losses=int(payload.get("losses", 0)),
                draws=int(payload.get("draws", 0)),
            )
            if min(record.games, record.wins, record.losses, record.draws) < 0:
                raise ValueError("PFSP state contains negative result counts")
            if record.wins + record.losses + record.draws != record.games:
                raise ValueError("PFSP state result counts do not add up to games")
            restored_records[label] = record
        self.records = restored_records
        self.segment_records = {label: OpponentRecord() for label in self.labels}
        self.completed_segments = max(0, int(summary.get("completed_segments", 0)))

    def finish_segment(self) -> tuple[list[float], dict]:
        if self.segment_games <= 0:
            raise ValueError("Cannot finish an empty PFSP segment")

        raw_scores = []
        for label in self.labels:
            mean, uncertainty = self.records[label].posterior(self.prior_games)
            informativeness = 4.0 * mean * (1.0 - mean)
            raw_scores.append(max(1e-12, informativeness * uncertainty))

        count = len(self.labels)
        score_total = sum(raw_scores)
        pfsp = [score / score_total for score in raw_scores]
        mixed = [
            (1.0 - self.random_fraction) * probability
            + self.random_fraction / count
            for probability in pfsp
        ]
        self.current_probabilities = _bounded_distribution(
            mixed,
            lower=self.random_fraction / count,
            upper=self.max_probability,
        )

        segment = {
            "segment": self.completed_segments + 1,
            "games": self.segment_games,
            "opponents": {
                label: self._record_dict(self.segment_records[label])
                for label in self.labels
            },
            "probabilities": dict(zip(self.labels, self.current_probabilities)),
        }
        self.completed_segments += 1
        self.segment_records = {label: OpponentRecord() for label in self.labels}
        return list(self.current_probabilities), segment

    def _record_dict(self, record: OpponentRecord) -> dict:
        mean, uncertainty = record.posterior(self.prior_games)
        effective_win_rate = (
            record.effective_wins / record.games if record.games > 0 else None
        )
        return {
            "games": record.games,
            "wins": record.wins,
            "losses": record.losses,
            "draws": record.draws,
            "effective_win_rate": effective_win_rate,
            "posterior_mean": mean,
            "posterior_uncertainty": uncertainty,
        }

    def summary(self) -> dict:
        return {
            "completed_segments": self.completed_segments,
            "probabilities": dict(zip(self.labels, self.current_probabilities)),
            "opponents": {
                label: self._record_dict(self.records[label])
                for label in self.labels
            },
        }


def labels_and_weights(pool: Iterable[dict]) -> tuple[list[str], list[float]]:
    entries = list(pool)
    return (
        [str(entry["label"]) for entry in entries],
        [max(0.0, float(entry.get("weight", 1.0))) for entry in entries],
    )
