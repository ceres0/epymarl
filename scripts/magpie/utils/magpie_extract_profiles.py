import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llm_marl.opponent_features import OpponentFeatureExtractor, summarize_episode_tree
from llm_marl.opponent_profile import OpponentProfileGenerator, OpponentProfileStore


def main():
    parser = argparse.ArgumentParser(description="Extract MAGPIE opponent profiles from trajectory episodes.")
    parser.add_argument("--input", required=True, help="Directory containing episode_*.json files or episodes/")
    parser.add_argument("--output", required=True, help="Directory for summary/profile artifacts")
    parser.add_argument("--profile-mode", default="heuristic", choices=["heuristic", "random", "file"])
    parser.add_argument("--profile-path", default="")
    parser.add_argument("--vector-dim", type=int, default=8)
    args = parser.parse_args()

    input_root = os.path.abspath(args.input)
    episode_root = os.path.join(input_root, "episodes") if os.path.isdir(os.path.join(input_root, "episodes")) else input_root
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    summaries = summarize_episode_tree(episode_root)
    feature_summary = OpponentFeatureExtractor().aggregate(summaries)
    source_episodes = [row.get("episode_path", "") for row in summaries]
    profile = OpponentProfileGenerator(
        vector_dim=args.vector_dim,
        mode=args.profile_mode,
        profile_path=args.profile_path,
    ).build(feature_summary, source_episodes=source_episodes)
    OpponentProfileStore(output_dir).save(profile)

    summary_path = os.path.join(output_dir, "opponent_feature_stability_report.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input": input_root,
                "episodes": len(summaries),
                "feature_summary": feature_summary,
                "profile": profile.to_dict(),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(summary_path)


if __name__ == "__main__":
    main()
