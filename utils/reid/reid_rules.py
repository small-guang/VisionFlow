"""Small composable acceptance rules for person ReID classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from core.algorithm_config import algorithm_config

from .reid_models import ReidCandidate


@dataclass(frozen=True)
class ReidClassFacts:
    reid: str
    query_face: str
    query_device: str
    canonical_face: str
    selected: tuple[ReidCandidate, ...]
    best_score: float
    mean_score: float
    raw_runner_score: float
    raw_margin: float

    @property
    def face_anchored(self) -> bool:
        return bool(self.query_face and self.canonical_face == self.query_face)

    @property
    def same_device_support(self) -> bool:
        return any(
            self.query_device and str(hit.device_id or "") == self.query_device
            for hit in self.selected
        )


@dataclass(frozen=True)
class RuleDecision:
    accepted: bool
    reason: str = "accepted"


class ReidClassRule(Protocol):
    def evaluate(self, facts: ReidClassFacts) -> RuleDecision: ...


class BodySupportRule:
    """Accept supported classes and unique face-anchor singletons."""

    def evaluate(self, facts: ReidClassFacts) -> RuleDecision:
        if len(facts.selected) >= 2:
            return RuleDecision(True)
        if self._is_unique_anchor(facts):
            return RuleDecision(True, "unique_face_anchor")
        return RuleDecision(False)

    @staticmethod
    def _is_unique_anchor(facts: ReidClassFacts) -> bool:
        hit = facts.selected[0]
        same_device = bool(
            facts.query_device
            and str(hit.device_id or "") == facts.query_device
        )
        unique_floor = (
            algorithm_config.reid.classification.body_local_top1
            if same_device
            else algorithm_config.reid.classification.unique_anchor_cross_device_top1
        )
        return bool(
            facts.canonical_face
            and facts.best_score > unique_floor
            and facts.raw_margin
            >= algorithm_config.reid.classification.unique_anchor_raw_margin
        )


class FormalMeanRule:
    def evaluate(self, facts: ReidClassFacts) -> RuleDecision:
        mean_floor = (
            algorithm_config.reid.classification.body_local_mean
            if facts.same_device_support
            else algorithm_config.reid.classification.body_cross_device_mean
        )
        return RuleDecision(facts.mean_score > mean_floor)


class ReidClassPolicy(Protocol):
    def evaluate(self, facts: ReidClassFacts) -> RuleDecision: ...


class _ComposedPolicy:
    rules: Sequence[ReidClassRule] = ()
    final_reason: str | None = None

    def evaluate(self, facts: ReidClassFacts) -> RuleDecision:
        reason = "accepted"
        for rule in self.rules:
            decision = rule.evaluate(facts)
            if not decision.accepted:
                return decision
            if decision.reason != "accepted":
                reason = decision.reason
        return RuleDecision(True, self.final_reason or reason)


class SameFacePolicy(_ComposedPolicy):
    pass


class AnonymousBodyPolicy(_ComposedPolicy):
    rules = (BodySupportRule(), FormalMeanRule())
