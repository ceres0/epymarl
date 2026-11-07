# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import json
import os
import shutil
from dataclasses import dataclass, field


DEFAULT_SUBREWARD_KEYS = [
    "win_bonus",
    "enemy_damage",
    "ally_damage_penalty",
    "enemy_kill_bonus",
    "ally_loss_penalty",
    "time_penalty",
]

DEFAULT_ALLOWED_CONTEXT_KEYS = [
    "actions",
    "agent_count",
    "base_reward",
    "env_info",
    "episode_limit",
    "step_index",
    "terminated",
    "truncated",
    "prev",
    "curr",
]


@dataclass
class RewardSpec:
    format_version: int
    spec_id: str
    version: int
    name: str
    reward_mode: str = "team"
    description: str = ""
    subreward_keys: list = field(default_factory=lambda: list(DEFAULT_SUBREWARD_KEYS))
    allowed_context_keys: list = field(
        default_factory=lambda: list(DEFAULT_ALLOWED_CONTEXT_KEYS)
    )
    strategy: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload):
        strategy = payload.get("strategy", {})
        if "type" not in strategy:
            raise ValueError("Reward spec strategy must define a type.")
        if payload.get("reward_mode", "team") not in {"team", "individual"}:
            raise ValueError("reward_mode must be either 'team' or 'individual'.")
        return cls(
            format_version=int(payload.get("format_version", 1)),
            spec_id=str(payload["spec_id"]),
            version=int(payload.get("version", 1)),
            name=str(payload["name"]),
            reward_mode=payload.get("reward_mode", "team"),
            description=payload.get("description", ""),
            subreward_keys=list(payload.get("subreward_keys", DEFAULT_SUBREWARD_KEYS)),
            allowed_context_keys=list(
                payload.get("allowed_context_keys", DEFAULT_ALLOWED_CONTEXT_KEYS)
            ),
            strategy=strategy,
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self):
        return {
            "format_version": self.format_version,
            "spec_id": self.spec_id,
            "version": self.version,
            "name": self.name,
            "reward_mode": self.reward_mode,
            "description": self.description,
            "subreward_keys": list(self.subreward_keys),
            "allowed_context_keys": list(self.allowed_context_keys),
            "strategy": dict(self.strategy),
            "metadata": dict(self.metadata),
        }

    def clone_with_version(self, version):
        payload = self.to_dict()
        payload["version"] = int(version)
        return RewardSpec.from_dict(payload)


class RewardSpecStore:
    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return RewardSpec.from_dict(payload)

    @staticmethod
    def save(spec, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(spec.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    @staticmethod
    def save_version(spec, root_dir):
        spec_dir = os.path.join(root_dir, spec.spec_id)
        os.makedirs(spec_dir, exist_ok=True)
        version_path = os.path.join(spec_dir, f"v{spec.version:03d}.json")
        RewardSpecStore.save(spec, version_path)

        manifest_path = os.path.join(spec_dir, "manifest.json")
        manifest = {"current_version": spec.version, "spec_id": spec.spec_id}
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return version_path

    @staticmethod
    def save_current(spec, root_dir, filename="current_reward_spec.json"):
        os.makedirs(root_dir, exist_ok=True)
        current_path = os.path.join(root_dir, filename)
        RewardSpecStore.save(spec, current_path)
        return current_path

    @staticmethod
    def load_current(root_dir, filename="current_reward_spec.json"):
        current_path = os.path.join(root_dir, filename)
        if not os.path.exists(current_path):
            raise FileNotFoundError(current_path)
        return RewardSpecStore.load(current_path)

    @staticmethod
    def rollback(root_dir, spec_id, version):
        spec_dir = os.path.join(root_dir, spec_id)
        target_path = os.path.join(spec_dir, f"v{int(version):03d}.json")
        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Cannot rollback reward spec '{spec_id}' to version {version}."
            )

        manifest_path = os.path.join(spec_dir, "manifest.json")
        manifest = {"current_version": int(version), "spec_id": spec_id}
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return RewardSpecStore.load(target_path)

    @staticmethod
    def copy_into_run(spec, run_dir):
        dst_dir = os.path.join(run_dir, "reward_versions")
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, f"{spec.spec_id}_v{spec.version:03d}.json")
        RewardSpecStore.save(spec, dst_path)
        return dst_path

    @staticmethod
    def seed_history(root_dir, spec_path):
        spec = RewardSpecStore.load(spec_path)
        RewardSpecStore.save_version(spec, root_dir)
        return spec

    @staticmethod
    def duplicate_version(root_dir, spec_id, src_version, dst_version):
        spec_dir = os.path.join(root_dir, spec_id)
        src = os.path.join(spec_dir, f"v{int(src_version):03d}.json")
        dst = os.path.join(spec_dir, f"v{int(dst_version):03d}.json")
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        shutil.copyfile(src, dst)
        return RewardSpecStore.load(dst)
