from smac.env import StarCraft2Env

from .multiagentenv import MultiAgentEnv

try:
    from card.reward_shaper import CardRewardShaper
except Exception:
    CardRewardShaper = None


class SMACWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, card_reward=None, **kwargs):
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit

        self._card_cfg = card_reward or {}
        self._card_shaper = None
        if bool(self._card_cfg.get("enabled", False)):
            if CardRewardShaper is None:
                raise RuntimeError(
                    "CARD reward shaping requested, but card.reward_shaper could not be imported."
                )
            plugin = self._card_cfg.get("plugin", "card.rewards.smac_env_reward")
            plugin_kwargs = self._card_cfg.get("kwargs", {})
            if isinstance(plugin_kwargs, str):
                try:
                    plugin_kwargs = json.loads(plugin_kwargs)
                except Exception:
                    plugin_kwargs = {}
            self._card_shaper = CardRewardShaper(
                enabled=True, plugin=str(plugin), plugin_kwargs=plugin_kwargs
            )

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews_raw, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False

        # Optional CARD-style reward shaping
        if self._card_shaper is not None:
            info = dict(info or {})
            state = self.get_state()
            rews, comps = self._card_shaper.compute(
                state=state,
                obs=obss,
                actions=actions,
                raw_reward=rews_raw,
                terminated=terminated,
                info=info,
                env=self.env,
            )
            try:
                info["card_reward_raw"] = float(rews_raw)
            except Exception:
                info["card_reward_raw"] = float(sum(rews_raw))
            info["card_reward_total"] = float(rews)
            if comps:
                info["card_reward_components"] = comps
        else:
            rews = rews_raw

        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_stats(self):
        return self.env.get_stats()



