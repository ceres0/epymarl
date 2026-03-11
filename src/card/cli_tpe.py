from __future__ import annotations

import argparse
import json

from card.tpe import eval_tpe_from_npz_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="CARD TPE evaluator (offline)")
    ap.add_argument("--traj-dir", required=True, help="Directory containing *.npz trajectories")
    ap.add_argument(
        "--reward-plugin",
        required=True,
        help="Python module path, e.g. card.rewards.smac_dense_v1",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Preference accuracy threshold (paper uses 0.8)",
    )
    ap.add_argument("--plugin-kwargs", default="{}", help="JSON dict passed to reward plugin")
    args = ap.parse_args()

    plugin_kwargs = json.loads(args.plugin_kwargs)
    res = eval_tpe_from_npz_dir(
        traj_dir=args.traj_dir,
        reward_plugin=args.reward_plugin,
        threshold=args.threshold,
        plugin_kwargs=plugin_kwargs,
    )

    print(
        json.dumps(
            {
                "passed": res.passed,
                "accuracy": res.accuracy,
                "n_success": res.n_success,
                "n_fail": res.n_fail,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
