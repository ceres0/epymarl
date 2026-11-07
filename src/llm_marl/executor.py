# Modified from EPyMARL for OA-QMIX and opponent-profile research.
import ast
import math

import numpy as np


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class CardRewardError(RuntimeError):
    pass


class RewardCodeValidator(ast.NodeVisitor):
    BLOCKED_NODES = (
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.AsyncWith,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Try,
        ast.With,
    )

    def generic_visit(self, node):
        if isinstance(node, self.BLOCKED_NODES):
            raise CardRewardError(
                f"Unsupported syntax for CARD reward execution: {type(node).__name__}."
            )
        super().generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("_"):
            raise CardRewardError(
                "CARD reward code cannot access private or dunder attributes."
            )
        self.generic_visit(node)


class RestrictedContext(dict):
    def __init__(self, payload, path="context"):
        super().__init__(payload)
        self._path = path

    def __getitem__(self, key):
        if key not in self:
            allowed = ", ".join(sorted(self.keys()))
            raise CardRewardError(
                f"Illegal context access at {self._path}.{key}. Allowed keys: {allowed}"
            )
        value = super().__getitem__(key)
        return wrap_restricted(value, f"{self._path}.{key}")

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]


def wrap_restricted(value, path):
    if isinstance(value, dict):
        return RestrictedContext(value, path=path)
    if isinstance(value, list):
        return [wrap_restricted(item, f"{path}[{idx}]") for idx, item in enumerate(value)]
    return value


def sanitize_numeric_dict(payload):
    sanitized = {}
    for key, value in payload.items():
        try:
            sanitized[key] = float(value)
        except (TypeError, ValueError):
            continue
    return sanitized


class RewardExecutor:
    def __init__(self, spec):
        self.spec = spec
        self._entrypoint = None
        self.validate_spec()
        self._compile_strategy()

    def validate_spec(self):
        if self.spec.strategy.get("type") != "python_function":
            raise CardRewardError(
                "This implementation currently supports only 'python_function' reward specs."
            )
        if not self.spec.strategy.get("code"):
            raise CardRewardError("Reward spec must include non-empty python code.")

    def _compile_strategy(self):
        code = self.spec.strategy["code"]
        tree = ast.parse(code)
        RewardCodeValidator().visit(tree)

        compiled_globals = {
            "__builtins__": SAFE_BUILTINS,
            "math": math,
            "np": np,
        }
        compiled_locals = {}
        exec(compile(tree, "<card_reward_spec>", "exec"), compiled_globals, compiled_locals)

        entrypoint_name = self.spec.strategy.get("entrypoint", "compute_reward")
        entrypoint = compiled_locals.get(entrypoint_name) or compiled_globals.get(
            entrypoint_name
        )
        if entrypoint is None or not callable(entrypoint):
            raise CardRewardError(
                f"Reward code must define a callable entrypoint named '{entrypoint_name}'."
            )
        self._entrypoint = entrypoint

    def compute(self, context, expected_agents):
        filtered_context = {
            key: context[key] for key in self.spec.allowed_context_keys if key in context
        }
        result = self._entrypoint(RestrictedContext(filtered_context))
        return self._normalise_result(result, expected_agents)

    def _normalise_result(self, result, expected_agents):
        if isinstance(result, tuple):
            if self.spec.reward_mode == "team" and len(result) == 3:
                payload = {
                    "team_reward": result[0],
                    "subreward_dict": result[1],
                    "aux_stats": result[2],
                }
            else:
                raise CardRewardError(
                    "Tuple reward output must match the configured reward mode."
                )
        elif isinstance(result, dict):
            payload = result
        else:
            raise CardRewardError(
                "Reward output must be a dict or a (team_reward, subreward_dict, aux_stats) tuple."
            )

        subreward_dict = payload.get("subreward_dict", {})
        if not isinstance(subreward_dict, dict):
            raise CardRewardError("subreward_dict must be a dictionary.")

        subrewards = {key: 0.0 for key in self.spec.subreward_keys}
        for key, value in subreward_dict.items():
            if key not in subrewards:
                raise CardRewardError(
                    f"Unexpected subreward key '{key}'. Expected keys: {self.spec.subreward_keys}"
                )
            subrewards[key] = float(value)

        aux_stats = sanitize_numeric_dict(payload.get("aux_stats", {}))

        if self.spec.reward_mode == "team":
            if "team_reward" not in payload:
                raise CardRewardError("Team reward specs must return 'team_reward'.")
            return {
                "reward": float(payload["team_reward"]),
                "subrewards": subrewards,
                "aux_stats": aux_stats,
                "reward_mode": "team",
            }

        agent_rewards = payload.get("agent_rewards")
        if agent_rewards is None:
            raise CardRewardError("Individual reward specs must return 'agent_rewards'.")
        if len(agent_rewards) != expected_agents:
            raise CardRewardError(
                f"Expected {expected_agents} agent rewards, got {len(agent_rewards)}."
            )
        return {
            "reward": [float(value) for value in agent_rewards],
            "team_reward": float(payload.get("team_reward", sum(agent_rewards))),
            "subrewards": subrewards,
            "aux_stats": aux_stats,
            "reward_mode": "individual",
        }
