# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import json
import os

from .reward_spec import RewardSpecStore


class CardOrchestrator:
    def __init__(self, output_dir, reward_spec):
        self.output_dir = output_dir
        self.reward_spec = reward_spec
        self.history_path = os.path.join(output_dir, "card_history.jsonl")
        self.spec_history_dir = os.path.join(output_dir, "spec_history")
        self.current_spec_path = os.path.join(output_dir, "current_reward_spec.json")
        os.makedirs(output_dir, exist_ok=True)
        RewardSpecStore.copy_into_run(reward_spec, output_dir)
        RewardSpecStore.save_version(reward_spec, self.spec_history_dir)
        RewardSpecStore.save_current(reward_spec, output_dir)

    def record_run(self, payload):
        with open(self.history_path, "a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")

    def create_candidate(self, metadata=None):
        next_version = self.reward_spec.version + 1
        candidate = self.reward_spec.clone_with_version(next_version)
        if metadata:
            candidate.metadata.update(metadata)
        RewardSpecStore.save_version(candidate, self.spec_history_dir)
        return candidate

    def activate_spec(self, spec):
        self.reward_spec = spec
        version_path = RewardSpecStore.save_version(spec, self.spec_history_dir)
        current_path = RewardSpecStore.save_current(spec, self.output_dir)
        self.record_run(
            {
                "event": "reward_spec_activated",
                "spec_id": spec.spec_id,
                "version": spec.version,
                "name": spec.name,
                "version_path": version_path,
                "current_path": current_path,
            }
        )
        return {"version_path": version_path, "current_path": current_path}
