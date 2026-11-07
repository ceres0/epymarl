# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import json
import os
import random
from dataclasses import dataclass

from .opponent_features import FEATURE_KEYS


STYLE_TO_VECTOR = {
    "unknown": [0.0, 0.0, 0.0, 0.0],
    "aggressive_focus_fire": [1.0, 0.0, 0.0, 0.0],
    "evasive_kiting": [0.0, 1.0, 0.0, 0.0],
    "scattered_target_switching": [0.0, 0.0, 1.0, 0.0],
    "balanced": [0.0, 0.0, 0.0, 1.0],
}

STYLE_TO_LABEL = {
    "unknown": 0,
    "aggressive_focus_fire": 1,
    "evasive_kiting": 2,
    "scattered_target_switching": 3,
    "balanced": 0,
}

STYLE_CATALOG = set(STYLE_TO_VECTOR)

STYLE_ALIASES = {
    "aggressive": "aggressive_focus_fire",
    "aggressive_focus": "aggressive_focus_fire",
    "focus_fire": "aggressive_focus_fire",
    "focus_fire_aggressive": "aggressive_focus_fire",
    "target_focus": "aggressive_focus_fire",
    "passive": "unknown",
    "passive_unclear": "unknown",
    "passive_undecided": "unknown",
    "passive_random": "unknown",
    "low_evidence": "unknown",
    "low_evidence_unknown": "unknown",
    "unclear": "unknown",
    "undecided": "unknown",
    "group_focused": "balanced",
    "group-focused": "balanced",
    "coordinated": "balanced",
    "coordinated_aggressive": "balanced",
    "cohesive": "balanced",
    "cohesive_focus": "balanced",
    "deathball": "balanced",
    "balanced_aggressive": "balanced",
    "balanced_defensive": "balanced",
    "evasive": "evasive_kiting",
    "kiting": "evasive_kiting",
    "defensive_retreat": "evasive_kiting",
    "retreat": "evasive_kiting",
    "target_switching": "scattered_target_switching",
    "high_target_switching": "scattered_target_switching",
    "switching": "scattered_target_switching",
}


@dataclass
class OpponentProfile:
    opponent_style: str
    behavior_vector: list
    counter_strategy: str
    confidence: float
    feature_summary: dict
    source_episodes: list
    profile_version: str = "magpie-profile-v1"
    generation_mode: str = "heuristic"
    llm_call_count: int = 0
    token_estimate: int = 0
    estimated_cost: float = 0.0
    failure_reason: str = ""

    def to_dict(self):
        return {
            "opponent_style": self.opponent_style,
            "behavior_vector": list(self.behavior_vector),
            "counter_strategy": self.counter_strategy,
            "confidence": float(self.confidence),
            "feature_summary": self.feature_summary,
            "source_episodes": list(self.source_episodes),
            "profile_version": self.profile_version,
            "generation_mode": self.generation_mode,
            "llm_call_count": int(self.llm_call_count),
            "token_estimate": int(self.token_estimate),
            "estimated_cost": float(self.estimated_cost),
            "failure_reason": self.failure_reason,
        }


class OpponentProfileStore:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.profile_path = os.path.join(output_dir, "opponent_profile.json")
        self.history_path = os.path.join(output_dir, "opponent_profile_history.jsonl")

    def save(self, profile):
        payload = profile.to_dict()
        with open(self.profile_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return self.profile_path

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_profile_payload(payload)
        return payload


class OpponentProfileGenerator:
    def __init__(self, vector_dim=8, mode="heuristic", profile_path=""):
        self.vector_dim = int(vector_dim or 8)
        self.mode = mode or "heuristic"
        self.profile_path = profile_path or ""

    def build(self, feature_summary, source_episodes=None):
        if self.mode == "file" and self.profile_path:
            payload = OpponentProfileStore.load(self.profile_path)
            return self._profile_from_payload(payload, feature_summary, source_episodes)
        if self.mode == "random":
            style = random.choice(list(STYLE_TO_VECTOR.keys()))
            return self._make_profile(style, feature_summary, source_episodes, "random")
        if self.mode == "zero":
            return OpponentProfile(
                opponent_style="zero_profile",
                behavior_vector=[0.0] * self.vector_dim,
                counter_strategy="no opponent profile signal is provided",
                confidence=0.0,
                feature_summary=feature_summary,
                source_episodes=source_episodes or [],
                generation_mode="zero",
            )
        style, confidence = self._classify(feature_summary)
        return self._make_profile(style, feature_summary, source_episodes, self.mode, confidence)

    def _profile_from_payload(self, payload, feature_summary, source_episodes):
        behavior_vector = self._fit_vector(payload.get("behavior_vector", []))
        opponent_style = normalize_opponent_style(
            payload["opponent_style"], behavior_vector
        )
        return OpponentProfile(
            opponent_style=opponent_style,
            behavior_vector=behavior_vector,
            counter_strategy=payload.get("counter_strategy", ""),
            confidence=payload.get("confidence", 0.0),
            feature_summary=payload.get("feature_summary") or feature_summary,
            source_episodes=payload.get("source_episodes") or source_episodes or [],
            profile_version=payload.get("profile_version", "magpie-profile-v1"),
            generation_mode=payload.get("generation_mode", "file"),
            llm_call_count=payload.get("llm_call_count", 0),
            token_estimate=payload.get("token_estimate", 0),
            estimated_cost=payload.get("estimated_cost", 0.0),
            failure_reason=payload.get("failure_reason", ""),
        )

    def _make_profile(self, style, feature_summary, source_episodes, mode, confidence=None):
        style = normalize_opponent_style(style)
        if confidence is None:
            confidence = 0.25 if style == "unknown" else 0.6
        return OpponentProfile(
            opponent_style=style,
            behavior_vector=self._fit_vector(STYLE_TO_VECTOR.get(style, STYLE_TO_VECTOR["unknown"])),
            counter_strategy=counter_strategy_for(style),
            confidence=confidence,
            feature_summary=feature_summary,
            source_episodes=source_episodes or [],
            generation_mode=mode,
        )

    def _classify(self, feature_summary):
        means = feature_summary.get("means", {}) if feature_summary else {}
        focus = float(means.get("focus_fire_rate", 0.0) or 0.0)
        chase = float(means.get("low_hp_chase_ratio", 0.0) or 0.0)
        switch = float(means.get("target_switch_rate", 0.0) or 0.0)
        distance = float(means.get("engage_distance_proxy", 0.0) or 0.0)
        if focus >= 0.6 or chase >= 0.5:
            return "aggressive_focus_fire", min(0.95, 0.5 + max(focus, chase) / 2)
        if switch >= 0.45:
            return "scattered_target_switching", min(0.9, 0.5 + switch / 2)
        if distance >= 0.45:
            return "evasive_kiting", min(0.85, 0.45 + distance / 2)
        if any(key in means for key in FEATURE_KEYS):
            return "balanced", 0.5
        return "unknown", 0.2

    def _fit_vector(self, vector):
        values = [float(item) for item in vector]
        if len(values) >= self.vector_dim:
            return values[: self.vector_dim]
        return values + [0.0] * (self.vector_dim - len(values))


def counter_strategy_for(style):
    style = normalize_opponent_style(style)
    return {
        "aggressive_focus_fire": "protect low-HP allies, counter-focus exposed enemies, avoid isolated deaths",
        "evasive_kiting": "control engage distance, avoid over-chasing, preserve formation while closing",
        "scattered_target_switching": "maintain formation cohesion and stable target focus",
        "balanced": "balance survival, focus fire, and objective progress",
        "unknown": "use outcome-return baseline until behavior evidence is sufficient",
    }.get(style, "use outcome-return baseline until behavior evidence is sufficient")


def style_label_for(style):
    style = normalize_opponent_style(style)
    return int(STYLE_TO_LABEL.get(style, STYLE_TO_LABEL["unknown"]))


def normalize_opponent_style(style, behavior_vector=None):
    if style == "zero_profile":
        return "zero_profile"
    key = _style_key(style)
    if key in STYLE_CATALOG:
        return key
    if key in STYLE_ALIASES:
        alias = STYLE_ALIASES[key]
        if key in {"aggressive", "balanced_aggressive"} and behavior_vector is not None:
            return _style_from_behavior_vector(behavior_vector)
        return alias
    if behavior_vector is not None:
        return _style_from_behavior_vector(behavior_vector)
    return "unknown"


def _style_key(style):
    return str(style or "").strip().lower().replace(" ", "_").replace("-", "_")


def _style_from_behavior_vector(vector):
    try:
        values = [float(item) for item in vector]
    except (TypeError, ValueError):
        return "unknown"
    if not values or max(abs(item) for item in values) <= 0.2:
        return "unknown"
    max_index = max(range(len(values)), key=lambda index: values[index])
    if max_index == 2:
        return "scattered_target_switching"
    if max_index in {3, 7}:
        return "evasive_kiting"
    if max_index == 6:
        return "balanced"
    return "aggressive_focus_fire"


def validate_profile_payload(payload):
    required = {
        "opponent_style",
        "behavior_vector",
        "counter_strategy",
        "confidence",
        "feature_summary",
        "source_episodes",
        "profile_version",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Opponent profile missing required keys: {missing}")
    if not isinstance(payload["behavior_vector"], list):
        raise ValueError("Opponent profile behavior_vector must be a list.")
