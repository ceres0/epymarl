# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from dataclasses import dataclass


OUTCOME_ORDER = {"timeout": 0, "loss": 1, "win": 2}


@dataclass
class TPEResult:
    passed: bool
    threshold: float
    accuracy: float
    total_pairs: int
    correct_pairs: int
    hacking_flags: list
    examples: list
    behavior_accuracy: float = 1.0
    combined_accuracy: float = 1.0
    behavior_counter_examples: list = None
    opponent_style: str = "unknown"

    def to_dict(self):
        return {
            "passed": self.passed,
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "outcome_accuracy": self.accuracy,
            "behavior_accuracy": self.behavior_accuracy,
            "combined_accuracy": self.combined_accuracy,
            "total_pairs": self.total_pairs,
            "correct_pairs": self.correct_pairs,
            "hacking_flags": list(self.hacking_flags),
            "examples": list(self.examples),
            "behavior_counter_examples": list(self.behavior_counter_examples or []),
            "opponent_style": self.opponent_style,
            "method_name": "MAGPIE-PE",
        }


class TPEEvaluator:
    def __init__(self, threshold, behavior_weight=0.5, behavior_enabled=False, hacking_check=True):
        self.threshold = float(threshold)
        self.behavior_weight = float(behavior_weight)
        self.behavior_enabled = bool(behavior_enabled)
        self.hacking_check = bool(hacking_check)

    def evaluate(self, trajectories, opponent_profile=None):
        ranked = [trajectory for trajectory in trajectories if trajectory["episode_length"] > 0]
        opponent_style = "unknown"
        if opponent_profile:
            opponent_style = opponent_profile.get("opponent_style", "unknown")

        total_pairs = 0
        correct_pairs = 0
        counter_examples = []
        for better in ranked:
            for worse in ranked:
                if better is worse:
                    continue
                if OUTCOME_ORDER[better["outcome"]] <= OUTCOME_ORDER[worse["outcome"]]:
                    continue
                total_pairs += 1
                if better["normalized_team_return"] > worse["normalized_team_return"]:
                    correct_pairs += 1
                elif len(counter_examples) < 3:
                    counter_examples.append(
                        {
                            "better_outcome": better["outcome"],
                            "worse_outcome": worse["outcome"],
                            "better_return": better["normalized_team_return"],
                            "worse_return": worse["normalized_team_return"],
                            "better_episode_path": better.get("episode_path", ""),
                            "worse_episode_path": worse.get("episode_path", ""),
                        }
                    )

        accuracy = 1.0 if total_pairs == 0 else correct_pairs / total_pairs
        behavior_accuracy, behavior_examples = self._evaluate_behavior(
            ranked, opponent_style
        )
        if not self.behavior_enabled:
            behavior_accuracy = 1.0
            behavior_examples = []
        combined_accuracy = (
            (1.0 - self.behavior_weight) * accuracy
            + self.behavior_weight * behavior_accuracy
        )
        hacking_flags = self._detect_hacking(ranked, opponent_style) if self.hacking_check else []
        passed = combined_accuracy >= self.threshold and not hacking_flags

        return TPEResult(
            passed=passed,
            threshold=self.threshold,
            accuracy=accuracy,
            total_pairs=total_pairs,
            correct_pairs=correct_pairs,
            hacking_flags=hacking_flags,
            examples=counter_examples,
            behavior_accuracy=behavior_accuracy,
            combined_accuracy=combined_accuracy,
            behavior_counter_examples=behavior_examples,
            opponent_style=opponent_style,
        )

    def _evaluate_behavior(self, trajectories, opponent_style):
        checks = []
        examples = []
        for item in trajectories:
            features = item.get("opponent_features") or {}
            combat = item.get("combat_summary") or {}
            normalized_return = item.get("normalized_team_return", 0.0)
            ok = True
            reason = ""
            if opponent_style == "aggressive_focus_fire":
                ally_alive = combat.get("ally_alive")
                if item.get("outcome") != "win" and normalized_return > 0 and ally_alive == 0:
                    ok = False
                    reason = "positive reward despite low-survival focus-fire loss"
            elif opponent_style == "scattered_target_switching":
                if features.get("target_switch_rate", 0.0) > 0.5 and normalized_return > 0:
                    ok = False
                    reason = "positive reward on high target-switching trajectory"
            elif opponent_style == "evasive_kiting":
                if features.get("engage_distance_proxy", 0.0) > 0.5 and normalized_return > 0:
                    ok = False
                    reason = "positive reward on over-chase distance pattern"
            checks.append(ok)
            if not ok and len(examples) < 3:
                examples.append(
                    {
                        "episode_path": item.get("episode_path", ""),
                        "opponent_style": opponent_style,
                        "reason": reason,
                        "normalized_team_return": normalized_return,
                    }
                )
        return (sum(1 for item in checks if item) / len(checks), examples) if checks else (1.0, [])

    def _detect_hacking(self, trajectories, opponent_style="unknown"):
        flags = []
        wins = [item for item in trajectories if item["outcome"] == "win"]
        losses = [item for item in trajectories if item["outcome"] == "loss"]
        timeouts = [item for item in trajectories if item["outcome"] == "timeout"]

        mean = lambda rows, key: (
            sum(item[key] for item in rows) / len(rows) if rows else None
        )

        win_return = mean(wins, "normalized_team_return")
        loss_return = mean(losses, "normalized_team_return")
        timeout_return = mean(timeouts, "normalized_team_return")

        if win_return is not None and loss_return is not None and loss_return >= win_return:
            flags.append("loss_return_not_below_win_return")
        if win_return is not None and timeout_return is not None and timeout_return >= win_return:
            flags.append("timeout_return_not_below_win_return")

        low_win_rate = len(wins) / max(1, len(trajectories)) < 0.2
        if low_win_rate:
            avg_enemy_damage = mean(
                [
                    {
                        "enemy_damage": item["subreward_totals"].get("enemy_damage", 0.0)
                        / max(1, item["episode_length"])
                    }
                    for item in trajectories
                ],
                "enemy_damage",
            )
            if avg_enemy_damage is not None and avg_enemy_damage > 0.5:
                flags.append("high_damage_reward_with_low_win_rate")

        avg_timeout_length = mean(timeouts, "episode_length")
        avg_win_length = mean(wins, "episode_length")
        if (
            avg_timeout_length is not None
            and avg_win_length is not None
            and avg_timeout_length > avg_win_length
            and timeout_return is not None
            and timeout_return > 0.0
        ):
            flags.append("positive_timeout_returns")

        if opponent_style == "aggressive_focus_fire":
            suspicious = [
                item
                for item in trajectories
                if item.get("normalized_team_return", 0.0) > 0.0
                and item.get("combat_summary", {}).get("ally_alive") == 0
            ]
            if suspicious:
                flags.append("positive_reward_with_focus_fire_victim_death")

        for item in trajectories:
            subrewards = item.get("subreward_totals", {}) or {}
            total_abs = sum(abs(value) for value in subrewards.values())
            if total_abs <= 0:
                continue
            largest = max(abs(value) for value in subrewards.values())
            if largest / total_abs > 0.9 and item.get("outcome") != "win":
                flags.append("single_subreward_dominates_non_win_return")
                break

        return flags
