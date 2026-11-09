"""``rby1-profile``: export, attach and compare RB-Y1 robot profiles.

Examples::

    # Read-only snapshot of a robot (no power / servo / motion)
    rby1-profile export --out robot_a.json --robot.address=192.168.30.1:50051

    # Put a profile into a dataset (local root or Hugging Face dataset repo)
    rby1-profile attach robot_a.json --dataset <hf_user>/<dataset>

    # Compare two robots / datasets (files, dataset roots or repo ids)
    rby1-profile diff <hf_user>/<dataset> robot_b.json

    # lerobot-policy-server config for a checkpoint (features from its dataset)
    rby1-profile server-config --checkpoint <hf_user>/<policy> --out server.yaml

``diff`` exits with 1 when an error-level difference exists (the policy's
observation/action layout or joint meaning would differ), else 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from . import profile as rprofile

logger = logging.getLogger(__name__)


def _export(args: argparse.Namespace, robot_args: list[str]) -> int:
    import draccus
    import rby1_sdk as rby

    from . import model_probe
    from .config_rby1 import Rby1Config

    cfg = draccus.parse(
        config_class=Rby1Config,
        args=[a.replace("--robot.", "--", 1) for a in robot_args],
    )
    model, version = model_probe.resolve_model_version(
        cfg.model, cfg.version, cfg.address, cfg.connect_timeout_sec
    )
    robot = rby.create_robot(cfg.address, model)
    if not robot.connect():
        print(f"Failed to connect to RB-Y1 at {cfg.address}.", file=sys.stderr)
        return 2
    try:
        profile = rprofile.build_profile(
            cfg,
            robot=rprofile.read_robot_section(robot, model, version),
            core_parameters=rprofile.read_core_parameters(robot),
            version=version,
        )
    finally:
        robot.disconnect()
    written = rprofile.write_profile(args.out, profile, force=args.force)
    print(f"Wrote {written}")
    return 0


def _attach(args: argparse.Namespace) -> int:
    profile = rprofile.load_profile(args.profile)
    target = Path(args.dataset).expanduser()
    if target.is_dir():
        try:
            written = rprofile.write_profile(
                target / "meta" / rprofile.PROFILE_FILENAME, profile, force=args.force
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Wrote {written}")
        return 0

    from huggingface_hub import HfApi

    path_in_repo = f"meta/{rprofile.PROFILE_FILENAME}"
    try:
        existing = rprofile.load_profile(rprofile.resolve_profile_source(args.dataset))
    except Exception:  # noqa: BLE001  (no profile on the Hub yet)
        existing = None
    if existing is not None:
        diffs = [d for d in rprofile.diff_profiles(existing, profile) if d.severity != rprofile.INFO]
        if diffs and not args.force:
            print(
                f"{args.dataset} already has a different profile:\n{rprofile.format_diffs(diffs)}\n"
                "Use --force to replace it.",
                file=sys.stderr,
            )
            return 1
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / rprofile.PROFILE_FILENAME
        local.write_text(json.dumps(profile, indent=2) + "\n")
        HfApi().upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=args.dataset,
            repo_type="dataset",
            commit_message="Add RB-Y1 robot profile",
        )
    print(f"Uploaded {path_in_repo} to {args.dataset}")
    return 0


def _diff(args: argparse.Namespace) -> int:
    a = rprofile.load_profile(rprofile.resolve_profile_source(args.reference))
    b = rprofile.load_profile(rprofile.resolve_profile_source(args.other))
    diffs = rprofile.diff_profiles(a, b)
    print(f"reference: {args.reference}\nother:     {args.other}")
    print(rprofile.format_diffs(diffs))
    return 1 if any(d.severity == rprofile.ERROR for d in diffs) else 0


def _server_config(args: argparse.Namespace) -> int:
    from .server_config import build_server_config, write_server_config

    config = build_server_config(
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        deployment=args.deployment,
        semantics=args.semantics,
        device=args.device,
        listen_endpoint=args.listen,
        fps=args.fps,
        n_action_steps=args.n_action_steps,
    )
    written = write_server_config(config, args.out)
    print(f"Wrote {written}")
    print(
        "Run:   lerobot-policy-server --config_path=" + str(written) + "\n"
        f"Then:  lerobot-rollout --robot.type=rby1 ... --inference.type=remote "
        f"--inference.endpoint=tcp/<server-host>:{args.listen.rsplit(':', 1)[-1]} "
        f"--inference.deployment={config['deployment']} --inference.semantics={config['semantics']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="rby1-profile", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="read-only profile of a connected robot")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="keep an existing different profile, write a time-stamped one")

    p = sub.add_parser("attach", help="put a profile into a dataset's meta/")
    p.add_argument("profile")
    p.add_argument("--dataset", required=True, help="dataset root or Hugging Face dataset repo id")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("diff", help="compare two profiles (file, dataset root or repo id)")
    p.add_argument("reference")
    p.add_argument("other")

    p = sub.add_parser("server-config", help="write a lerobot-policy-server YAML for a checkpoint")
    p.add_argument("--checkpoint", required=True, help="checkpoint directory or Hub model repo id")
    p.add_argument("--dataset", help="dataset root or repo id (default: from the checkpoint's train_config.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--deployment", default="rby1")
    p.add_argument("--semantics", help="default: the dataset profile's semantics")
    p.add_argument("--device", default="cuda")
    p.add_argument("--listen", default="tcp/0.0.0.0:7447")
    p.add_argument("--fps", type=float, help="action fps (default: the dataset fps)")
    p.add_argument("--n_action_steps", type=int)

    args, rest = parser.parse_known_args(argv)
    if args.command == "export":
        return _export(args, rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "attach":
        return _attach(args)
    if args.command == "server-config":
        return _server_config(args)
    return _diff(args)


if __name__ == "__main__":
    sys.exit(main())
