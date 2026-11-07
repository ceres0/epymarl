# Modified from EPyMARL for OA-QMIX and opponent-profile research.
class FeedbackBuilder:
    def build_process_feedback(self, trajectories):
        if not trajectories:
            return {
                "metrics": {},
                "text": "No trajectories were collected for process feedback.",
            }

        total = len(trajectories)
        wins = [row for row in trajectories if row["outcome"] == "win"]
        losses = [row for row in trajectories if row["outcome"] == "loss"]
        timeouts = [row for row in trajectories if row["outcome"] == "timeout"]

        metrics = {
            "episodes": total,
            "win_rate": len(wins) / total,
            "loss_rate": len(losses) / total,
            "timeout_rate": len(timeouts) / total,
            "avg_episode_length": sum(row["episode_length"] for row in trajectories) / total,
            "avg_team_return": sum(row["team_return"] for row in trajectories) / total,
            "avg_normalized_team_return": sum(
                row["normalized_team_return"] for row in trajectories
            )
            / total,
        }

        aggregate_subrewards = {}
        for row in trajectories:
            for key, value in row["subreward_totals"].items():
                aggregate_subrewards[key] = aggregate_subrewards.get(key, 0.0) + value
        for key, value in aggregate_subrewards.items():
            metrics[f"avg_{key}"] = value / total

        text = (
            "Process Feedback\n"
            f"- Episodes: {metrics['episodes']}\n"
            f"- Win rate: {metrics['win_rate']:.3f}\n"
            f"- Loss rate: {metrics['loss_rate']:.3f}\n"
            f"- Timeout rate: {metrics['timeout_rate']:.3f}\n"
            f"- Avg episode length: {metrics['avg_episode_length']:.2f}\n"
            f"- Avg team return: {metrics['avg_team_return']:.3f}\n"
            f"- Avg normalized team return: {metrics['avg_normalized_team_return']:.3f}\n"
        )
        if aggregate_subrewards:
            text += "- Avg subrewards:\n"
            for key in sorted(aggregate_subrewards):
                text += f"  - {key}: {aggregate_subrewards[key] / total:.3f}\n"

        return {"metrics": metrics, "text": text}

    def build_trajectory_feedback(self, trajectories):
        if not trajectories:
            return {
                "best": None,
                "worst": None,
                "text": "No trajectories were collected for trajectory feedback.",
            }

        ranked = sorted(
            trajectories,
            key=lambda item: item["normalized_team_return"],
            reverse=True,
        )
        best = ranked[0]
        worst = ranked[-1]
        text = (
            "Trajectory Feedback\n"
            f"- Best episode: outcome={best['outcome']}, normalized_return={best['normalized_team_return']:.3f}, "
            f"length={best['episode_length']}, path={best.get('episode_path', '')}\n"
            f"- Worst episode: outcome={worst['outcome']}, normalized_return={worst['normalized_team_return']:.3f}, "
            f"length={worst['episode_length']}, path={worst.get('episode_path', '')}\n"
        )
        return {"best": best, "worst": worst, "text": text}

    def build_preference_feedback(self, tpe_result):
        payload = tpe_result.to_dict()
        text = (
            "Preference Feedback\n"
            f"- TPE passed: {tpe_result.passed}\n"
            f"- Preference accuracy: {tpe_result.accuracy:.3f} "
            f"({tpe_result.correct_pairs}/{tpe_result.total_pairs})\n"
            f"- MAGPIE-PE behavior accuracy: {payload.get('behavior_accuracy', 1.0):.3f}\n"
            f"- MAGPIE-PE combined accuracy: {payload.get('combined_accuracy', tpe_result.accuracy):.3f}\n"
        )
        if tpe_result.hacking_flags:
            text += "- Hacking flags:\n"
            for flag in tpe_result.hacking_flags:
                text += f"  - {flag}\n"
        if tpe_result.examples:
            text += "- Counter examples:\n"
            for example in tpe_result.examples:
                text += (
                    "  - "
                    f"{example['better_outcome']} should outrank {example['worse_outcome']}, "
                    f"but normalized returns were {example['better_return']:.3f} <= {example['worse_return']:.3f}\n"
                )
        behavior_examples = payload.get("behavior_counter_examples", [])
        if behavior_examples:
            text += "- MAGPIE-PE behavior counter examples:\n"
            for example in behavior_examples:
                text += (
                    "  - "
                    f"{example.get('opponent_style', 'unknown')}: {example.get('reason', '')} "
                    f"path={example.get('episode_path', '')}\n"
                )
        return {"text": text, "tpe": tpe_result.to_dict()}
