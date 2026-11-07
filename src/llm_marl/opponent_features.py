# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import json
import os
from collections import Counter


FEATURE_KEYS = [
    "focus_fire_rate",
    "low_hp_chase_ratio",
    "target_switch_rate",
    "engage_distance_proxy",
    "formation_spread_proxy",
    "damage_burst_window_proxy",
]


def _as_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _distance_proxy(context):
    for key in ("engage_distance", "enemy_distance", "mean_enemy_distance"):
        if key in context:
            return _as_float(context.get(key), None)
    ally = _as_float(context.get("ally_health_fraction"), None)
    enemy = _as_float(context.get("enemy_health_fraction"), None)
    if ally is None or enemy is None:
        return None
    return abs(ally - enemy)


def _spread_proxy(context):
    for key in ("formation_spread", "ally_spread", "mean_ally_distance"):
        if key in context:
            return _as_float(context.get(key), None)
    alive = _as_float(context.get("ally_alive"), None)
    if alive is None:
        return None
    return 1.0 / max(1.0, alive)


def _target_id(step):
    env_info = step.get("env_info", {}) or {}
    for key in ("enemy_focus_target", "target_enemy", "focus_target"):
        if key in env_info:
            return env_info.get(key)
    actions = step.get("actions", [])
    if hasattr(actions, "tolist"):
        actions = actions.tolist()
    if isinstance(actions, (list, tuple)) and actions:
        counts = Counter(str(item) for item in actions)
        return counts.most_common(1)[0][0]
    return None


class OpponentFeatureExtractor:
    def extract_episode(self, episode):
        steps = episode.get("steps", [])
        diagnostics = []
        if not steps:
            return self._empty_features(["no_steps"])

        focus_events = 0
        target_events = 0
        low_hp_chase_events = 0
        low_hp_windows = 0
        target_switches = 0
        previous_target = None
        distances = []
        spreads = []
        damage_deltas = []

        for step in steps:
            prev_context = step.get("prev_context", {}) or {}
            curr_context = step.get("curr_context", {}) or {}

            target = _target_id(step)
            if target is not None:
                target_events += 1
                actions = step.get("actions", [])
                if hasattr(actions, "tolist"):
                    actions = actions.tolist()
                if isinstance(actions, (list, tuple)) and actions:
                    top_count = Counter(str(item) for item in actions).most_common(1)[0][1]
                    if top_count / max(1, len(actions)) >= 0.5:
                        focus_events += 1
                if previous_target is not None and target != previous_target:
                    target_switches += 1
                previous_target = target

            ally_hp = _as_float(curr_context.get("ally_health_fraction"), None)
            enemy_hp = _as_float(curr_context.get("enemy_health_fraction"), None)
            if ally_hp is None:
                diagnostics.append("missing_ally_health_fraction")
            if enemy_hp is None:
                diagnostics.append("missing_enemy_health_fraction")
            if ally_hp is not None and ally_hp < 0.35:
                low_hp_windows += 1
                enemy_delta = _as_float(prev_context.get("enemy_health_total")) - _as_float(
                    curr_context.get("enemy_health_total")
                )
                if enemy_delta > 0.0:
                    low_hp_chase_events += 1

            distance = _distance_proxy(curr_context)
            if distance is None:
                diagnostics.append("missing_engage_distance_proxy")
            else:
                distances.append(distance)

            spread = _spread_proxy(curr_context)
            if spread is None:
                diagnostics.append("missing_formation_spread_proxy")
            else:
                spreads.append(spread)

            damage_delta = _as_float(prev_context.get("enemy_health_total")) - _as_float(
                curr_context.get("enemy_health_total")
            )
            if damage_delta > 0:
                damage_deltas.append(damage_delta)

        features = {
            "focus_fire_rate": focus_events / max(1, target_events),
            "low_hp_chase_ratio": low_hp_chase_events / max(1, low_hp_windows),
            "target_switch_rate": target_switches / max(1, target_events - 1),
            "engage_distance_proxy": sum(distances) / len(distances) if distances else 0.0,
            "formation_spread_proxy": sum(spreads) / len(spreads) if spreads else 0.0,
            "damage_burst_window_proxy": max(damage_deltas) if damage_deltas else 0.0,
            "diagnostics": sorted(set(diagnostics)),
        }
        return features

    def aggregate(self, summaries):
        rows = [row.get("opponent_features") or {} for row in summaries]
        rows = [row for row in rows if row]
        if not rows:
            return {"episodes": 0, "means": {}, "diagnostics": ["no_opponent_features"]}
        means = {}
        for key in FEATURE_KEYS:
            values = [_as_float(row.get(key)) for row in rows]
            means[key] = sum(values) / len(values)
        diagnostics = sorted(
            set(item for row in rows for item in row.get("diagnostics", []))
        )
        return {"episodes": len(rows), "means": means, "diagnostics": diagnostics}

    def _empty_features(self, diagnostics):
        return {key: 0.0 for key in FEATURE_KEYS} | {"diagnostics": diagnostics}


def load_episode_file(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_episode_file(path):
    episode = load_episode_file(path)
    features = OpponentFeatureExtractor().extract_episode(episode)
    summary = dict(episode.get("summary", {}))
    summary["opponent_features"] = features
    summary["episode_path"] = path
    return summary


def summarize_episode_tree(root):
    summaries = []
    if not os.path.isdir(root):
        return summaries
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.endswith(".json") and filename.startswith("episode_"):
                summaries.append(summarize_episode_file(os.path.join(dirpath, filename)))
    return summaries
