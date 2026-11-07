import argparse
import importlib.util
import os
import shutil
import subprocess
import sys


MODULES = [
    "gymnasium",
    "matrixgames",
    "smaclite",
    "smac",
    "smacv2",
    "pettingzoo",
    "vmas",
    "torch",
    "wandb",
]


def module_available(name):
    return importlib.util.find_spec(name) is not None


def command_available(name):
    return shutil.which(name) is not None


def docker_compose_config():
    if not command_available("docker"):
        return False, "docker command not found"
    try:
        completed = subprocess.run(
            ["docker", "compose", "config", "--quiet"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            text=True,
            capture_output=True,
            timeout=60,
        )
    except Exception as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout).strip()
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Check local/Docker readiness for MAGPIE experiments.")
    parser.add_argument("--docker", action="store_true", help="Also validate docker compose config.")
    args = parser.parse_args()

    print("Python:", sys.executable)
    print("Working directory:", os.getcwd())
    print("\nPython modules:")
    for name in MODULES:
        print(f"  {name}: {'ok' if module_available(name) else 'missing'}")

    sc2path = os.environ.get("SC2PATH", "")
    print("\nEnvironment:")
    print(f"  SC2PATH: {sc2path or 'unset'}")
    if sc2path:
        print(f"  SC2PATH exists: {os.path.isdir(sc2path)}")

    print("\nCommands:")
    for name in ("git", "docker"):
        print(f"  {name}: {'ok' if command_available(name) else 'missing'}")

    if args.docker:
        ok, detail = docker_compose_config()
        print("\nDocker compose:")
        print(f"  config: {'ok' if ok else 'failed'}")
        if detail != "ok":
            print(f"  detail: {detail}")
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
