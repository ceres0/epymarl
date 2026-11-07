# Modified from EPyMARL for OA-QMIX and opponent-profile research.
from smac.env import StarCraft2Env

from .multiagentenv import MultiAgentEnv


def _extract_unit_list(env, attr):
    backend = env
    for _ in range(4):
        units = getattr(backend, attr, None)
        if units is not None:
            return list(units)
        backend = getattr(backend, "env", None)
        if backend is None:
            break
    return []


def _unit_health(unit):
    health = float(getattr(unit, "health", 0.0) or 0.0)
    shield = float(getattr(unit, "shield", 0.0) or 0.0)
    return max(0.0, health) + max(0.0, shield)


def _unit_max_health(unit):
    health_max = float(getattr(unit, "health_max", 0.0) or 0.0)
    shield_max = float(getattr(unit, "shield_max", 0.0) or 0.0)
    if shield_max == 0.0:
        shield_max = float(getattr(unit, "max_shield", 0.0) or 0.0)
    return max(0.0, health_max) + max(0.0, shield_max)


class SMACWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        self.map_name = map_name
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
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

    def get_card_context(self):
        allies = _extract_unit_list(self.env, "agents")
        enemies = _extract_unit_list(self.env, "enemies")

        ally_health_total = sum(_unit_health(unit) for unit in allies)
        enemy_health_total = sum(_unit_health(unit) for unit in enemies)
        ally_health_cap = max(1.0, sum(_unit_max_health(unit) for unit in allies))
        enemy_health_cap = max(1.0, sum(_unit_max_health(unit) for unit in enemies))
        ally_alive = sum(1 for unit in allies if _unit_health(unit) > 0.0)
        enemy_alive = sum(1 for unit in enemies if _unit_health(unit) > 0.0)

        return {
            "ally_alive": float(ally_alive),
            "enemy_alive": float(enemy_alive),
            "ally_health_total": float(ally_health_total),
            "enemy_health_total": float(enemy_health_total),
            "ally_health_fraction": float(ally_health_total / ally_health_cap),
            "enemy_health_fraction": float(enemy_health_total / enemy_health_cap),
        }

    def get_card_env_descriptor(self):
        env_info = self.get_env_info()
        return {
            "env_name": "sc2",
            "map_name": self.map_name,
            "episode_limit": self.episode_limit,
            "n_agents": env_info["n_agents"],
            "n_actions": env_info["n_actions"],
        }
