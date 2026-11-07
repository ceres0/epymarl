# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import json
import os

from .opponent_features import OpponentFeatureExtractor


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class TrajectoryStore:
    def __init__(self, output_dir, save_episodes=True):
        self.output_dir = output_dir
        self.save_episodes = bool(save_episodes)
        self.episodes_dir = os.path.join(output_dir, "episodes")
        if self.save_episodes:
            os.makedirs(self.episodes_dir, exist_ok=True)
        self._active = {}
        self._completed = []
        self._counter = self._initial_counter() if self.save_episodes else 0
        self._log_buffer = {"train": [], "test": []}
        self._feature_extractor = OpponentFeatureExtractor()

    def _initial_counter(self):
        highest = -1
        if not os.path.isdir(self.episodes_dir):
            return 0
        for root, _, files in os.walk(self.episodes_dir):
            for filename in files:
                if not filename.startswith("episode_") or not filename.endswith(".json"):
                    continue
                stem = filename[len("episode_") : -len(".json")]
                try:
                    highest = max(highest, int(stem))
                except ValueError:
                    continue
        return highest + 1

    def start_episode(self, env_id, metadata, initial_context, test_mode):
        self._active[env_id] = {
            "env_id": env_id,
            "metadata": _to_jsonable(metadata),
            "initial_context": _to_jsonable(initial_context),
            "steps": [],
            "test_mode": bool(test_mode),
        }

    def record_step(
        self,
        env_id,
        step_index,
        actions,
        env_reward,
        shaped_reward,
        subrewards,
        aux_stats,
        prev_context,
        curr_context,
        terminated,
        truncated,
        env_info,
    ):
        episode = self._active[env_id]
        episode["steps"].append(
            {
                "step_index": int(step_index),
                "actions": _to_jsonable(actions),
                "env_reward": _to_jsonable(env_reward),
                "shaped_reward": _to_jsonable(shaped_reward),
                "subrewards": _to_jsonable(subrewards),
                "aux_stats": _to_jsonable(aux_stats),
                "prev_context": _to_jsonable(prev_context),
                "curr_context": _to_jsonable(curr_context),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "env_info": _to_jsonable(env_info),
            }
        )

    def finish_episode(self, env_id, outcome, env_info):
        episode = self._active.pop(env_id)
        steps = episode["steps"]

        team_return = 0.0
        subreward_totals = {}
        final_context = {}
        for step in steps:
            shaped = step["shaped_reward"]
            if isinstance(shaped, list):
                team_return += float(sum(shaped))
            else:
                team_return += float(shaped)
            for key, value in step["subrewards"].items():
                subreward_totals[key] = subreward_totals.get(key, 0.0) + float(value)
            final_context = step["curr_context"]

        episode_length = len(steps)
        opponent_features = self._feature_extractor.extract_episode(episode)
        summary = {
            "episode_id": self._counter,
            "env_id": env_id,
            "test_mode": episode["test_mode"],
            "metadata": episode["metadata"],
            "outcome": outcome,
            "episode_length": episode_length,
            "team_return": team_return,
            "normalized_team_return": team_return / max(1, episode_length),
            "subreward_totals": subreward_totals,
            "opponent_features": opponent_features,
            "combat_summary": {
                "ally_alive": final_context.get("ally_alive"),
                "enemy_alive": final_context.get("enemy_alive"),
                "ally_health_total": final_context.get("ally_health_total"),
                "enemy_health_total": final_context.get("enemy_health_total"),
                "battle_won": bool(env_info.get("battle_won", False)),
            },
            "env_info": _to_jsonable(env_info),
        }
        episode["summary"] = summary
        episode["opponent_features"] = opponent_features
        self._completed.append(episode)

        bucket = "test" if episode["test_mode"] else "train"
        self._log_buffer[bucket].append(summary)

        episode_path = ""
        if self.save_episodes:
            bucket_dir = os.path.join(self.episodes_dir, bucket)
            os.makedirs(bucket_dir, exist_ok=True)
            episode_path = os.path.join(bucket_dir, f"episode_{self._counter:06d}.json")
            with open(episode_path, "w", encoding="utf-8") as handle:
                json.dump(_to_jsonable(episode), handle, indent=2, sort_keys=True)
                handle.write("\n")

        summary["episode_path"] = episode_path
        self._counter += 1
        self._write_opponent_feature_summary()
        return summary

    def completed_summaries(self, test_mode=None):
        summaries = [episode["summary"] for episode in self._completed]
        if test_mode is None:
            return list(summaries)
        return [summary for summary in summaries if summary["test_mode"] == bool(test_mode)]

    def consume_log_summaries(self, test_mode):
        bucket = "test" if test_mode else "train"
        payload = list(self._log_buffer[bucket])
        self._log_buffer[bucket].clear()
        return payload

    def aggregate_opponent_features(self, test_mode=None):
        return self._feature_extractor.aggregate(self.completed_summaries(test_mode))

    def _write_opponent_feature_summary(self):
        path = os.path.join(self.output_dir, "opponent_feature_summary.json")
        payload = {
            "train": self.aggregate_opponent_features(test_mode=False),
            "test": self.aggregate_opponent_features(test_mode=True),
            "all": self.aggregate_opponent_features(test_mode=None),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
