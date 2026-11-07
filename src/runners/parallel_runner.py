# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from functools import partial
from multiprocessing import Pipe, Process
import time

import numpy as np

from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY
from envs import register_smac, register_smacv2


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.card_runtime = None
        self.card_contexts = {}
        self.batch_size = self.args.batch_size_run
        self.env_start_delay = getattr(self.args, "parallel_env_start_delay", 0.0)
        if self.env_start_delay <= 0 and self.args.env in ("sc2", "sc2v2") and self.batch_size >= 64:
            self.env_start_delay = 0.1
        self.env_reset_retries = int(getattr(self.args, "parallel_env_reset_retries", 0) or 0)
        if (
            self.env_reset_retries <= 0
            and self.args.env in ("sc2", "sc2v2")
            and self.batch_size >= 64
        ):
            self.env_reset_retries = 3
        self.env_reset_retry_delay = float(
            getattr(self.args, "parallel_env_reset_retry_delay", 5.0) or 5.0
        )

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )

        # registering both smac and smacv2 causes a pysc2 error
        # --> dynamically register the needed env
        if self.args.env == "sc2":
            register_smac()
        elif self.args.env == "sc2v2":
            register_smacv2()

        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i
            env_args[i]["common_reward"] = self.args.common_reward
            env_args[i]["reward_scalarisation"] = self.args.reward_scalarisation
            env_args[i]["card"] = getattr(self.args, "card", {})
        self.ps = [
            Process(
                target=env_worker,
                args=(
                    worker_conn,
                    CloudpickleWrapper(partial(env_fn, **env_arg)),
                    idx,
                    self.env_start_delay,
                    self.env_reset_retries,
                    self.env_reset_retry_delay,
                ),
            )
            for idx, (env_arg, worker_conn) in enumerate(zip(env_args, self.worker_conns))
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        self._closed = False

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
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def set_card_runtime(self, card_runtime):
        self.card_runtime = card_runtime

    def _opponent_batch_fields(self, batch_size):
        if "opp_profile" not in self.scheme:
            return {}
        profile = None
        if self.card_runtime is not None:
            profile = self.card_runtime.current_opponent_profile_vector()
        if profile is None:
            profile = [0.0] * int(getattr(self.args, "opp_profile_dim", 0) or 0)
        fields = {"opp_profile": [profile for _ in range(batch_size)]}
        if "opp_label_mask" in self.scheme:
            labels = None
            if self.card_runtime is not None:
                labels = self.card_runtime.current_opponent_aux_labels()
            if labels is None:
                labels = {
                    "opp_style_label": (0,),
                    "enemy_focus_target": (0,),
                    "danger_window": (0.0,),
                    "opp_label_mask": [0.0, 0.0, 0.0],
                }
            fields.update(
                {key: [value for _ in range(batch_size)] for key, value in labels.items()}
            )
        return fields

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        if self._closed:
            return
        self._closed = True

        for parent_conn in self.parent_conns:
            try:
                parent_conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass

        for parent_conn in self.parent_conns:
            try:
                parent_conn.close()
            except OSError:
                pass

        for worker_conn in self.worker_conns:
            try:
                worker_conn.close()
            except OSError:
                pass

        for process in self.ps:
            if process is None:
                continue
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            self._raise_worker_error(data)
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            if self.card_runtime is not None:
                env_idx = len(pre_transition_data["state"]) - 1
                self.card_contexts[env_idx] = data["card_context"]
                self.card_runtime.start_episode(
                    env_id=env_idx,
                    metadata=data["card_env_descriptor"],
                    initial_context=data["card_context"],
                    test_mode=test_mode,
                )

        pre_transition_data.update(self._opponent_batch_fields(self.batch_size))
        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset(test_mode=test_mode)

        all_terminated = False
        if self.args.common_reward:
            episode_returns = [0 for _ in range(self.batch_size)]
        else:
            episode_returns = [
                np.zeros(self.args.n_agents) for _ in range(self.batch_size)
            ]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            envs_not_terminated = _live_env_indices(terminated)
            if not envs_not_terminated:
                break

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode,
            )
            cpu_actions = actions.to("cpu").numpy()
            selected_avail_actions = (
                self.batch["avail_actions"][envs_not_terminated, self.t]
                .to("cpu")
                .numpy()
            )
            illegal_actions = _unavailable_action_report(
                cpu_actions, selected_avail_actions
            )
            if illegal_actions:
                raise RuntimeError(
                    "parallel runner selected unavailable actions before env.step "
                    f"at t_ep={self.t}, t_env={self.t_env}: {illegal_actions[:5]}"
                )

            # Update the actions taken
            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            action_lookup = {}
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    parent_conn.send(("step", cpu_actions[action_idx]))
                    action_lookup[idx] = cpu_actions[action_idx]
                    action_idx += 1  # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Post step data we will insert for the current timestep
            post_transition_data = {"reward": [], "terminated": []}
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            # Receive data back for each unterminated env
            next_env_count = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    data = parent_conn.recv()
                    self._raise_worker_error(data)
                    if self.card_runtime is not None:
                        reward = self.card_runtime.compute_reward(
                            env_id=idx,
                            step_index=self.t,
                            actions=action_lookup[idx],
                            env_reward=data["reward"],
                            pre_context=self.card_contexts[idx],
                            post_context=data["card_context"],
                            terminated=data["terminated"],
                            truncated=False,
                            env_info=data["info"],
                        )
                        self.card_contexts[idx] = data["card_context"]
                        data["reward"] = reward
                    # Remaining data for this current timestep
                    if self.args.common_reward:
                        post_transition_data["reward"].append((data["reward"],))
                    else:
                        post_transition_data["reward"].append(tuple(data["reward"]))

                    if self.args.common_reward:
                        episode_returns[idx] += data["reward"]
                    else:
                        episode_returns[idx] += np.array(data["reward"])
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                        if self.card_runtime is not None:
                            self.card_runtime.finish_episode(idx, data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    next_env_count += 1

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            pre_transition_data.update(self._opponent_batch_fields(next_env_count))
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            self._raise_worker_error(env_stat)
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
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
        if self.card_runtime is not None:
            self.card_runtime.log_runner_stats(
                self.logger,
                prefix,
                self.t_env,
                test_mode=(prefix == "test_"),
            )
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

    def _raise_worker_error(self, data):
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"])


def _close_env_quietly(env):
    if env is None:
        return
    try:
        env.close()
    except Exception as exc:
        print(f"[parallel_runner] ignoring env.close failure: {exc}", flush=True)


def _reset_env_with_retries(env, env_fn, worker_index, reset_retries, retry_delay):
    last_error = None
    max_attempts = reset_retries + 1
    for attempt in range(max_attempts):
        try:
            env.reset()
            return env, None
        except Exception as exc:
            last_error = exc
            print(
                "[parallel_runner] worker "
                f"{worker_index} env.reset failed "
                f"attempt {attempt + 1}/{max_attempts}: {exc}",
                flush=True,
            )
            _close_env_quietly(env)
            if attempt < reset_retries:
                time.sleep(retry_delay * (attempt + 1))
                env = env_fn.x()
    return env, last_error


def _unavailable_action_report(actions, avail_actions):
    actions = np.asarray(actions)
    avail_actions = np.asarray(avail_actions)
    if actions.ndim == 2:
        report = []
        for env_position, env_actions in enumerate(actions):
            env_report = _unavailable_action_report(
                env_actions, avail_actions[env_position]
            )
            for item in env_report:
                item = dict(item)
                item["env_position"] = env_position
                report.append(item)
        return report

    report = []
    for agent_id, action in enumerate(actions):
        action = int(action)
        if action < 0 or action >= len(avail_actions[agent_id]):
            report.append(
                {
                    "agent_id": agent_id,
                    "action": action,
                    "available": None,
                    "n_actions": int(len(avail_actions[agent_id])),
                }
            )
        elif int(avail_actions[agent_id][action]) == 0:
            report.append(
                {
                    "agent_id": agent_id,
                    "action": action,
                    "available": 0,
                    "avail_actions": [int(v) for v in avail_actions[agent_id]],
                }
            )
    return report


def _live_env_indices(terminated):
    return [b_idx for b_idx, termed in enumerate(terminated) if not termed]


def env_worker(
    remote,
    env_fn,
    worker_index=0,
    initial_reset_delay=0.0,
    reset_retries=0,
    reset_retry_delay=5.0,
):
    # Make environment
    env = env_fn.x()
    first_reset = True
    try:
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break
            if cmd == "step":
                actions = data
                # Take a step in the environment
                try:
                    _, reward, terminated, truncated, env_info = env.step(actions)
                except Exception as exc:
                    try:
                        action_availability = _unavailable_action_report(
                            actions, env.get_avail_actions()
                        )
                    except Exception as avail_exc:
                        action_availability = [
                            {"avail_actions_error": str(avail_exc)}
                        ]
                    remote.send(
                        {
                            "error": (
                                f"parallel env worker {worker_index} step failed: {exc}; "
                                f"current_action_availability={action_availability[:5]}"
                            )
                        }
                    )
                    break
                terminated = terminated or truncated
                # Return the observations, avail_actions and state to make the next action
                state = env.get_state()
                avail_actions = env.get_avail_actions()
                obs = env.get_obs()
                remote.send(
                    {
                        # Data for the next timestep needed to pick an action
                        "state": state,
                        "avail_actions": avail_actions,
                        "obs": obs,
                        # Rest of the data for the current timestep
                        "reward": reward,
                        "terminated": terminated,
                        "info": env_info,
                        "card_context": env.get_card_context(),
                    }
                )
            elif cmd == "reset":
                if first_reset and initial_reset_delay > 0:
                    time.sleep(worker_index * initial_reset_delay)
                first_reset = False
                env, reset_error = _reset_env_with_retries(
                    env, env_fn, worker_index, reset_retries, reset_retry_delay
                )
                if reset_error is not None:
                    remote.send(
                        {
                            "error": (
                                f"parallel env worker {worker_index} reset failed "
                                f"after {reset_retries + 1} attempts: {reset_error}"
                            )
                        }
                    )
                    break
                remote.send(
                    {
                        "state": env.get_state(),
                        "avail_actions": env.get_avail_actions(),
                        "obs": env.get_obs(),
                        "card_context": env.get_card_context(),
                        "card_env_descriptor": env.get_card_env_descriptor(),
                    }
                )
            elif cmd == "close":
                break
            elif cmd == "get_env_info":
                remote.send(env.get_env_info())
            elif cmd == "get_stats":
                remote.send(env.get_stats())
            elif cmd == "render":
                env.render()
            elif cmd == "save_replay":
                env.save_replay()
            else:
                raise NotImplementedError
    finally:
        try:
            env.close()
        finally:
            try:
                remote.close()
            except OSError:
                pass


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)
