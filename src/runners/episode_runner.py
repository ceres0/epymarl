from functools import partial
import os
from pathlib import Path

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2

from card.trajectory import NpzTrajectoryWriter


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        self.env = env_REGISTRY[self.args.env](
            **self.args.env_args,
            common_reward=self.args.common_reward,
            reward_scalarisation=self.args.reward_scalarisation,
        )
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # CARD optional trajectory recording
        self._card_traj_enabled = bool(getattr(self.args, "card_collect_trajectories", False))
        self._card_train_collect = bool(getattr(self.args, "card_collect_train_episodes", False))
        self._card_test_collect = bool(getattr(self.args, "card_collect_test_episodes", True))
        self._card_writers = None

        if self._card_traj_enabled:
            base_dir = getattr(self.args, "card_trajectory_dir", None)
            if not base_dir:
                run_id = getattr(self.args, "card_run_id", None) or getattr(
                    self.args, "unique_token", "run"
                )
                base_dir = os.path.join(
                    self.args.local_results_path, "card", "trajectories", str(run_id)
                )

            max_eps = int(getattr(self.args, "card_trajectory_max_episodes", 100))
            sample_interval = int(getattr(self.args, "card_trajectory_sample_interval", 1))
            state_dim_cap = getattr(self.args, "card_trajectory_state_dim_cap", None)

            self._card_writers = {
                "train": NpzTrajectoryWriter(
                    out_dir=Path(base_dir) / "train",
                    max_episodes=max_eps,
                    sample_interval=sample_interval,
                    state_dim_cap=state_dim_cap,
                ),
                "test": NpzTrajectoryWriter(
                    out_dir=Path(base_dir) / "test",
                    max_episodes=max_eps,
                    sample_interval=sample_interval,
                    state_dim_cap=state_dim_cap,
                ),
            }

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        if self.args.common_reward:
            episode_return = 0
        else:
            episode_return = np.zeros(self.args.n_agents)
        self.mac.init_hidden(batch_size=self.batch_size)

        # CARD: per-episode component aggregation
        card_comp_sum = {}
        card_comp_steps = 0

        # CARD: optional trajectory buffer
        traj_buf = None
        mode = "test" if test_mode else "train"
        can_collect = (
            self._card_traj_enabled
            and self._card_writers is not None
            and ((test_mode and self._card_test_collect) or ((not test_mode) and self._card_train_collect))
            and self._card_writers[mode].can_write_next()
        )
        if can_collect:
            traj_buf = self._card_writers[mode].start_episode()

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )

            _, reward, terminated, truncated, env_info = self.env.step(actions[0])
            terminated = terminated or truncated

            if test_mode and self.args.render:
                self.env.render()

            episode_return += reward

            # CARD: accumulate sub-reward components if provided
            step_comps = None
            if isinstance(env_info, dict):
                step_comps = env_info.get("card_reward_components", None)
            if step_comps:
                card_comp_steps += 1
                for k, v in step_comps.items():
                    card_comp_sum[str(k)] = card_comp_sum.get(str(k), 0.0) + float(v)

            # CARD: record (state, action) -> reward sequence
            if traj_buf is not None:
                if self.args.common_reward:
                    r_scalar = float(reward)
                else:
                    r_scalar = float(np.asarray(reward).sum())

                self._card_writers[mode].add_step(
                    traj_buf,
                    state=pre_transition_data["state"][0],
                    actions=actions[0],
                    reward=r_scalar,
                    terminated=bool(terminated),
                    components=step_comps,
                )

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        safe_env_info = {}
        if isinstance(env_info, dict):
            for k, v in env_info.items():
                if k in ("card_reward_components", "card_reward_error"):
                    continue
                if isinstance(v, (bool, int, float, np.number)):
                    safe_env_info[k] = v

        cur_stats.update(
            {
                k: cur_stats.get(k, 0) + safe_env_info.get(k, 0)
                for k in set(cur_stats) | set(safe_env_info)
            }
        )
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # CARD: log mean component values per episode
        if card_comp_steps > 0:
            for k, s in card_comp_sum.items():
                cur_stats[f"card_comp_{k}"] = cur_stats.get(f"card_comp_{k}", 0.0) + (s / float(card_comp_steps))

        # CARD: flush trajectory to disk (one file per episode)
        if traj_buf is not None:
            if self.args.common_reward:
                ep_ret_scalar = float(episode_return)
            else:
                ep_ret_scalar = float(np.asarray(episode_return).sum())

            meta = {
                "t_env": int(self.t_env),
                "test_mode": bool(test_mode),
                "episode_length": int(self.t),
                "episode_return": ep_ret_scalar,
                "battle_won": bool(env_info.get("battle_won", False)) if isinstance(env_info, dict) else False,
                "map_name": self.args.env_args.get("map_name", self.args.env_args.get("key", None)),
                "card_reward_plugin": (self.args.env_args.get("card_reward", {}) or {}).get("plugin", None),
            }
            self._card_writers[mode].end_episode(traj_buf, meta=meta)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if self.args.common_reward:
            self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
            self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        else:
            for i in range(self.args.n_agents):
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_mean",
                    np.array(returns)[:, i].mean(),
                    self.t_env,
                )
                self.logger.log_stat(
                    prefix + f"agent_{i}_return_std",
                    np.array(returns)[:, i].std(),
                    self.t_env,
                )
            total_returns = np.array(returns).sum(axis=-1)
            self.logger.log_stat(
                prefix + "total_return_mean", total_returns.mean(), self.t_env
            )
            self.logger.log_stat(
                prefix + "total_return_std", total_returns.std(), self.t_env
            )
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

