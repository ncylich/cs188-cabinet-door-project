"""
Step 0: Verify Installation
============================
Checks that all required packages are installed and the OpenCabinet
environment can be created successfully.
"""

import os
import sys

# On Linux/WSL2, EGL offscreen rendering requires /dev/dri device access which
# is unavailable in most WSL environments.  Force osmesa (CPU-based offscreen
# renderer) so the environment-creation check doesn't require a GPU.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


def check_package(name, min_version=None):
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        status = "OK"
        if min_version and version != min_version:
            status = f"WARNING (expected {min_version}, got {version})"
        print(f"  {name:25s} {version:15s} [{status}]")
        return True
    except ImportError:
        print(f"  {name:25s} {'MISSING':15s} [FAIL]")
        return False


def main():
    print("=" * 60)
    print("RoboCasa Cabinet Door Project - Installation Check")
    print("=" * 60)
    print(f"\nPython: {sys.version}\n")

    print("Checking required packages:")
    all_ok = True
    all_ok &= check_package("mujoco", "3.3.1")
    all_ok &= check_package("numpy", "2.2.5")
    all_ok &= check_package("robosuite")
    all_ok &= check_package("robocasa")
    all_ok &= check_package("gymnasium")
    all_ok &= check_package("imageio")

    if not all_ok:
        print("\nSome packages are missing or have wrong versions.")
        print("Please fix the issues above before continuing.")
        sys.exit(1)

    print("\nChecking environment creation...")
    try:
        import robocasa  # noqa: F401 - registers environments
        from robocasa.utils.env_utils import create_env

        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=42,
            camera_widths=128,
            camera_heights=128,
        )
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Environment created: OpenCabinet")
        print(f"  Task description:    {lang}")
        print(f"  Layout ID:           {env.layout_id}")
        print(f"  Style ID:            {env.style_id}")
        print(f"  Robot:               {env.robots[0].name}")
        env.close()
        print("  Environment creation: [OK]")
    except Exception as e:
        print(f"  Environment creation: [FAIL] {e}")
        print("\n  If kitchen assets are missing, run:")
        print("  python -m robocasa.scripts.download_kitchen_assets")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All checks passed! You are ready to start the project.")
    print("=" * 60)


if __name__ == "__main__":
    main()
