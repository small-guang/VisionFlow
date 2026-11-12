"""Composable class-resolution policies for person ReID candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from core.algorithm_config import algorithm_config

from .reid_models import (
    REID_SOURCE_BUFFER,
    REID_SOURCE_MILVUS,
    ReidCandidate,
    ReidClassResolution,
    finite_float,
    normalize_face_id,
)
from .reid_rules import (
    AnonymousBodyPolicy,
    ReidClassFacts,
    ReidClassPolicy,
    SameFacePolicy,
)
from .reid_state import (
    REID_STATE_ACTIVE,
    REID_STATE_UNASSIGNED,
)


@dataclass(frozen=True)
class _ClassScore:
    reid: str
    score: float
    best_score: float
    mean_score: float
    support_count: int
    canonical_face_id: str
    sources: tuple[str, ...]
    reason: str
    raw_runner_up_score: float
    raw_margin: float


def _candidate_minimum(
    candidate: ReidCandidate,
    *,
    query_face_id: str,
    canonical_face_id: str,
    query_device_id: str,
) -> float:
    base = max(0.0, finite_float(candidate.threshold))
    if query_face_id and canonical_face_id == query_face_id:
        return base
    same_device = bool(
        query_device_id and str(candidate.device_id or "") == query_device_id
    )
    return max(
        base,
        algorithm_config.reid.classification.body_local_top1
        if same_device
        else algorithm_config.reid.classification.body_cross_device_top1,
    )


def _group_candidates(
    candidates: Iterable[ReidCandidate],
) -> tuple[dict[str, list[ReidCandidate]], dict[str, list[ReidCandidate]]]:
    raw_grouped: dict[str, list[ReidCandidate]] = {}
    grouped: dict[str, list[ReidCandidate]] = {}
    for candidate in candidates:
        reid = str(candidate.reid or "").strip()
        score = finite_float(candidate.score, default=float("-inf"))
        threshold = finite_float(candidate.threshold, default=float("inf"))
        if not reid or not math.isfinite(score) or not math.isfinite(threshold):
            continue
        raw_grouped.setdefault(reid, []).append(candidate)
        if score > threshold:
            grouped.setdefault(reid, []).append(candidate)
    return raw_grouped, grouped


def _canonical_face(all_hits: Sequence[ReidCandidate]) -> str | None:
    face_ids = {
        face_id
        for face_id in (normalize_face_id(hit.face_id) for hit in all_hits)
        if face_id
    }
    if len(face_ids) > 1:
        return None
    return next(iter(face_ids), "")


def _identity_key(reid: str, canonical_face: str | None) -> str:
    face_id = normalize_face_id(canonical_face)
    if face_id:
        return f"face:{face_id}"
    return f"reid:{str(reid or '').strip()}"


def _selected_hits(
    hits: Sequence[ReidCandidate],
    *,
    query_face: str,
    canonical_face: str,
    query_device: str,
    top_k: int,
) -> tuple[ReidCandidate, ...]:
    accepted = [
        hit
        for hit in hits
        if float(hit.score)
        > _candidate_minimum(
            hit,
            query_face_id=query_face,
            canonical_face_id=canonical_face,
            query_device_id=query_device,
        )
    ]
    source_order = {REID_SOURCE_MILVUS: 0, REID_SOURCE_BUFFER: 1}
    accepted.sort(
        key=lambda hit: (
            -float(hit.score),
            source_order.get(str(hit.source or ""), 99),
            str(hit.prototype_id or ""),
        )
    )
    return tuple(accepted[:top_k])


class ReidClassResolver:
    def __init__(self) -> None:
        self.same_face_policy = SameFacePolicy()
        self.anonymous_body_policy = AnonymousBodyPolicy()

    def resolve(
        self,
        candidates: Iterable[ReidCandidate],
        *,
        query_face_id: Any = "",
        query_device_id: Any = "",
        top_k: int | None = None,
    ) -> ReidClassResolution:
        if top_k is None:
            top_k = algorithm_config.reid.classification.class_top_k
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_face = normalize_face_id(query_face_id)
        query_device = str(query_device_id or "").strip()
        required_margin = (
            algorithm_config.reid.classification.face_class_margin
            if query_face
            else algorithm_config.reid.classification.body_class_margin
        )
        raw_grouped, grouped = _group_candidates(candidates)
        raw_best = {
            reid: max(float(hit.score) for hit in hits)
            for reid, hits in raw_grouped.items()
        }
        raw_identity_keys = {
            reid: _identity_key(reid, _canonical_face(hits))
            for reid, hits in raw_grouped.items()
        }
        scores = self._evaluate_classes(
            raw_grouped,
            grouped,
            raw_best,
            raw_identity_keys,
            query_face=query_face,
            query_device=query_device,
            top_k=top_k,
        )
        return self._select_winner(
            scores,
            raw_best,
            raw_identity_keys,
            required_margin,
            collapse_identities=not bool(query_face),
        )

    def _evaluate_classes(
        self,
        raw_grouped: dict[str, list[ReidCandidate]],
        grouped: dict[str, list[ReidCandidate]],
        raw_best: dict[str, float],
        raw_identity_keys: dict[str, str],
        *,
        query_face: str,
        query_device: str,
        top_k: int,
    ) -> list[_ClassScore]:
        scores: list[_ClassScore] = []
        for reid, hits in grouped.items():
            facts = self._build_facts(
                reid,
                hits,
                raw_grouped.get(reid, hits),
                raw_best,
                raw_identity_keys,
                query_face=query_face,
                query_device=query_device,
                top_k=top_k,
            )
            if facts is None:
                continue
            policy = self._policy_for(facts)
            decision = policy.evaluate(facts)
            if decision.accepted:
                scores.append(self._score(facts, decision.reason))
        return scores

    def _build_facts(
        self,
        reid: str,
        hits: list[ReidCandidate],
        all_hits: list[ReidCandidate],
        raw_best: dict[str, float],
        raw_identity_keys: dict[str, str],
        *,
        query_face: str,
        query_device: str,
        top_k: int,
    ) -> ReidClassFacts | None:
        canonical_face = _canonical_face(all_hits)
        if canonical_face is None:
            return None
        if query_face and canonical_face != query_face:
            return None
        selected = _selected_hits(
            hits,
            query_face=query_face,
            canonical_face=canonical_face,
            query_device=query_device,
            top_k=top_k,
        )
        if not selected:
            return None
        best_score = float(selected[0].score)
        raw_runner = max(
            (
                score
                for other, score in raw_best.items()
                if raw_identity_keys.get(other) != raw_identity_keys.get(reid)
            ),
            default=0.0,
        )
        return ReidClassFacts(
            reid=reid,
            query_face=query_face,
            query_device=query_device,
            canonical_face=canonical_face,
            selected=selected,
            best_score=best_score,
            mean_score=sum(float(hit.score) for hit in selected) / len(selected),
            raw_runner_score=raw_runner,
            raw_margin=best_score - raw_runner,
        )

    def _policy_for(self, facts: ReidClassFacts) -> ReidClassPolicy:
        if facts.face_anchored:
            return self.same_face_policy
        return self.anonymous_body_policy

    @staticmethod
    def _score(facts: ReidClassFacts, reason: str) -> _ClassScore:
        source_order = {REID_SOURCE_MILVUS: 0, REID_SOURCE_BUFFER: 1}
        sources = tuple(
            sorted(
                {str(hit.source or "") for hit in facts.selected if hit.source},
                key=lambda source: source_order.get(source, 99),
            )
        )
        return _ClassScore(
            reid=facts.reid,
            score=(
                algorithm_config.reid.classification.best_score_weight
                * facts.best_score
                + algorithm_config.reid.classification.mean_score_weight
                * facts.mean_score
            ),
            best_score=facts.best_score,
            mean_score=facts.mean_score,
            support_count=len(facts.selected),
            canonical_face_id=facts.canonical_face,
            sources=sources,
            reason=reason,
            raw_runner_up_score=facts.raw_runner_score,
            raw_margin=facts.raw_margin,
        )

    @staticmethod
    def _select_winner(
        scores: list[_ClassScore],
        raw_best: dict[str, float],
        raw_identity_keys: dict[str, str],
        required_margin: float,
        *,
        collapse_identities: bool,
    ) -> ReidClassResolution:
        if not scores:
            if collapse_identities:
                raw_by_identity: dict[str, float] = {}
                for reid, score in raw_best.items():
                    identity_key = raw_identity_keys[reid]
                    raw_by_identity[identity_key] = max(
                        score,
                        raw_by_identity.get(identity_key, float("-inf")),
                    )
                raw_scores = sorted(raw_by_identity.values(), reverse=True)
            else:
                raw_scores = sorted(raw_best.values(), reverse=True)
            raw_top = raw_scores[0] if raw_scores else 0.0
            raw_runner = raw_scores[1] if len(raw_scores) > 1 else 0.0
            return ReidClassResolution(
                reid=None,
                required_margin=required_margin,
                raw_runner_up_score=raw_runner,
                raw_margin=raw_top - raw_runner,
                reason="no_compatible_class",
            )
        scores.sort(key=lambda item: (-item.score, -item.best_score, item.reid))
        if collapse_identities:
            representatives: list[_ClassScore] = []
            identity_keys: set[str] = set()
            for score in scores:
                identity_key = _identity_key(score.reid, score.canonical_face_id)
                if identity_key not in identity_keys:
                    representatives.append(score)
                    identity_keys.add(identity_key)
            scores = representatives
        winner = scores[0]
        runner_score = scores[1].score if len(scores) > 1 else 0.0
        margin = winner.score - runner_score
        if len(scores) > 1 and margin < required_margin:
            return _resolution_from_score(
                winner,
                reid=None,
                runner_score=runner_score,
                margin=margin,
                required_margin=required_margin,
                reason="ambiguous_margin",
                source="",
            )
        return _resolution_from_score(
            winner,
            reid=winner.reid,
            runner_score=runner_score,
            margin=margin,
            required_margin=required_margin,
            reason=winner.reason,
            source="+".join(winner.sources),
        )


def _resolution_from_score(
    score: _ClassScore,
    *,
    reid: str | None,
    runner_score: float,
    margin: float,
    required_margin: float,
    reason: str,
    source: str,
) -> ReidClassResolution:
    return ReidClassResolution(
        reid=reid,
        score=score.score,
        best_score=score.best_score,
        mean_score=score.mean_score,
        runner_up_score=runner_score,
        margin=margin,
        required_margin=required_margin,
        raw_runner_up_score=score.raw_runner_up_score,
        raw_margin=score.raw_margin,
        source=source,
        support_count=score.support_count,
        canonical_face_id=score.canonical_face_id,
        state=REID_STATE_ACTIVE if reid is not None else REID_STATE_UNASSIGNED,
        reason=reason,
    )


_DEFAULT_RESOLVER = ReidClassResolver()


def resolve_reid_classes(
    candidates: Iterable[ReidCandidate],
    *,
    query_face_id: Any = "",
    query_device_id: Any = "",
    top_k: int | None = None,
) -> ReidClassResolution:
    return _DEFAULT_RESOLVER.resolve(
        candidates,
        query_face_id=query_face_id,
        query_device_id=query_device_id,
        top_k=top_k,
    )
