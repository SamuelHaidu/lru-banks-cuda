"""Command-line interface.

Five commands, and deliberately almost no flags:

    model create   --run-name X [--arch A] [--dataset D]
    model config   --run-name X [set <dotted.key> <value>]
    model train    --run-name X [--epochs N]
    model evaluate --run-name X [--checkpoint latest|best|<epoch>]
    model generate --run-name X --prompt "..." [sampling flags]

Hyperparameters are set exclusively through `model config set`, never through
a flag. A flag with a hyperparameter default silently overrides the persisted
config for anyone who does not pass it -- that exact pattern produced three
separate bugs in the previous codebase. The only flags accepted here either
select a run, select a checkpoint, or control one-off generation.

Heavy modules are imported inside each command so that `config.json`'s
`device` can be applied (via `KERAS_DEVICE`) *before* keras is imported.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from model import config as cfg_mod


def _apply_device(run_name: str) -> None:
    """Pins the compute device from the run's `config.json`.

    Runs before the command's heavy imports, so `KERAS_DEVICE` is already in
    place by the time keras is first touched. `setup_device` is then called
    explicitly rather than relying on that environment variable, so a command
    that never triggers a lazy resolution still ends up on the right device.
    """
    try:
        device = cfg_mod.load(run_name).get("device")
    except cfg_mod.ConfigError:
        return
    if device:
        os.environ.setdefault("KERAS_DEVICE", str(device))
    from model.device import setup_device

    setup_device(device)


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------

def cmd_create(args: argparse.Namespace) -> int:
    from model.archs import ARCHS, arch_defaults

    if args.arch is not None and args.arch not in ARCHS:
        print(f"Unknown arch {args.arch!r}. Available: {sorted(ARCHS)}", file=sys.stderr)
        return 1

    path = cfg_mod.config_path(args.run_name)
    if path.exists() and not args.force:
        print(f"Run '{args.run_name}' already exists at {path}. Use --force to overwrite.", file=sys.stderr)
        return 1

    if args.init_from:
        # Fine-tuning run: start from the source run's config and weights.
        if args.arch or args.dataset:
            print("--arch/--dataset are taken from --init-from's run; drop them.", file=sys.stderr)
            return 1
        config = cfg_mod.load(args.init_from)
        config["run_name"] = args.run_name
        config["init_from"] = {"run": args.init_from, "checkpoint": args.init_checkpoint}
        # Pin the kernel basis to the source's window, so changing window_size
        # evaluates the same basis functions instead of rebuilding them.
        params = config["arch_params"]
        if "basis_window" in arch_defaults(config["arch"]) and not params.get("basis_window"):
            params["basis_window"] = params["window_size"]
    else:
        arch = args.arch or "dual-sphere"
        dataset = args.dataset or "tinystories-10k"
        meta = cfg_mod.load_dataset_metadata(dataset)

        config = json.loads(json.dumps(cfg_mod.DEFAULT_CONFIG))  # deep copy
        config["run_name"] = args.run_name
        config["arch"] = arch
        config["data"]["dataset"] = dataset

        # Write the architecture's defaults out explicitly, so every value the run
        # depends on is visible in config.json rather than hidden in a constructor
        # default that may change later. `vocabulary_size` is overridden from the
        # dataset metadata so it cannot drift from the corpus it was built for.
        params = arch_defaults(arch)
        params.pop("embeddings_initializer", None)
        params["vocabulary_size"] = meta["vocabulary_size"]
        if "reset_token" in params:  # archs whose state restarts at the document marker
            params["reset_token"] = meta["doc_token_id"]
        config["arch_params"] = params
        # Settings an architecture is meant to train with (e.g. warmup), over the global defaults.
        for section, values in getattr(ARCHS[arch], "config_defaults", {}).items():
            config[section].update(values)
    if args.context_length:
        config["arch_params"]["window_size"] = int(args.context_length)

    cfg_mod.save(args.run_name, config)
    print(f"Created {path}")
    print(json.dumps(config, indent=2))
    return 0


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def cmd_config(args: argparse.Namespace) -> int:
    config = cfg_mod.load(args.run_name)
    if not args.action:
        print(json.dumps(config, indent=2))
        return 0
    if args.action != "set":
        print(f"Unknown config action {args.action!r}. Only 'set' is supported.", file=sys.stderr)
        return 1
    if not args.key or args.value is None:
        print("Usage: model config --run-name X set <dotted.key> <value>", file=sys.stderr)
        return 1

    value = cfg_mod.set_value(config, args.key, args.value)
    cfg_mod.save(args.run_name, config)
    print(f"{args.key} = {json.dumps(value)}")
    return 0


# --------------------------------------------------------------------------
# train / evaluate / generate
# --------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    _apply_device(args.run_name)
    from model.train import train

    train(args.run_name, epochs=args.epochs)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _apply_device(args.run_name)
    from model.evaluate import evaluate, format_results

    results = evaluate(args.run_name, checkpoint=args.checkpoint, batch_size=args.batch_size,
                       context_length=args.context_length)
    print(format_results(results))
    return 0


# --------------------------------------------------------------------------
# distill
# --------------------------------------------------------------------------

def cmd_distill_create(args: argparse.Namespace) -> int:
    from model.distill import create_distill_run, distill_config_path

    try:
        create_distill_run(
            run_name=args.run_name,
            teacher_run=args.teacher_run,
            student_arch=args.student_arch,
            dataset=args.dataset,
            teacher_checkpoint=args.teacher_checkpoint,
            force=args.force,
        )
    except cfg_mod.ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Created {distill_config_path(args.run_name)}")
    return 0


def cmd_distill_train(args: argparse.Namespace) -> int:
    from model.distill import distill

    distill(args.run_name, epochs=args.epochs)
    return 0


def cmd_scripts(args: argparse.Namespace) -> int:
    import importlib

    from model.scripts import SCRIPTS

    if args.script not in SCRIPTS:
        print(f"Unknown script {args.script!r}. Available: {sorted(SCRIPTS)}", file=sys.stderr)
        return 1

    module = importlib.import_module(SCRIPTS[args.script])
    script_args = module.build_parser().parse_args(args.script_args or [])
    if getattr(script_args, "run_name", None):  # cross-run scripts (e.g. compare-runs) take no run
        _apply_device(script_args.run_name)
    return module.main(args.script_args or [])


def cmd_generate(args: argparse.Namespace) -> int:
    _apply_device(args.run_name)
    from model.evaluate import load_checkpoint

    config = cfg_mod.load(args.run_name)
    model = load_checkpoint(args.run_name, args.checkpoint)
    if args.context_length and args.context_length != model.context_length:
        model = model.with_context_length(args.context_length)

    tokenizer = None
    tokenizer_file = cfg_mod.tokenizer_path(config["data"]["dataset"])
    if tokenizer_file.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_file))

    import time

    stamps: list = []
    new_ids: list = []
    shown = 0
    printer = None
    if args.pretty:
        from model.pretty import AgentPrinter

        printer = AgentPrinter()

    def emit(text: str) -> None:
        if printer is not None:
            printer.feed(text)
        else:
            print(text, end="", flush=True)

    def on_token(token_id: int) -> None:
        # Stream the text: decode all new ids and print what is new; hold back a trailing partial UTF-8 character.
        nonlocal shown
        stamps.append(time.perf_counter())
        new_ids.append(token_id)
        if tokenizer is None:
            print(token_id, end=" ", flush=True)
            return
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        if text.endswith("\ufffd"):
            return
        emit(text[shown:])
        shown = len(text)

    emit(args.prompt)
    start = time.perf_counter()
    model.generate(
        args.prompt,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_tokens=args.stop_tokens,
        on_token=on_token,
    )
    if tokenizer is not None and new_ids:
        emit(tokenizer.decode(new_ids, skip_special_tokens=False)[shown:])
    if printer is not None:
        printer.finish()
    else:
        print()
    if stamps:
        first = stamps[0] - start
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else float("nan")
        print(f"[{len(stamps)} tokens, {rate:.1f} tokens/s after the first (prompt + first token: {first:.2f} s)]",
              file=sys.stderr)
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model", description="Train, evaluate and sample spherical LMs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("create", help="Create a run directory and its config.json.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--arch", default=None, help="Default: dual-sphere.")
    p.add_argument("--dataset", default=None, help="Default: tinystories-10k.")
    p.add_argument("--force", action="store_true", help="Overwrite an existing config.json.")
    p.add_argument("--init-from", default=None, help="Fine-tune: copy this run's config and start from its weights.")
    p.add_argument("--init-checkpoint", default="latest", help="Checkpoint of --init-from: latest | best | <epoch>.")
    p.add_argument("--context-length", type=int, default=None, help="Set arch_params.window_size (e.g. 256 to extend).")
    p.set_defaults(func=cmd_create)

    p = sub.add_parser("config", help="Print or modify a run's config.json.")
    p.add_argument("--run-name", required=True)
    p.add_argument("action", nargs="?", choices=["set"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("train", help="Train or resume a run (auto-resumes if checkpoints exist).")
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs for this invocation only.")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Score a checkpoint on the full validation split.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--checkpoint", default="latest", help="latest | best | <epoch number>")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--context-length", type=int, default=None, help="Score at a different window (RoPE archs only).")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("generate", help="Generate a continuation (greedy by default).")
    p.add_argument("--run-name", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--context-length", type=int, default=None, help="Generate with a different window (RoPE archs only).")
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy argmax.")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument(
        "--stop-tokens", nargs="*", default=["<eos>"],
        help="Stop generation once any of these tokens is produced. Pass nothing to disable.",
    )
    p.add_argument(
        "--pretty", action="store_true",
        help="Color agent-format text by tag (system, tools, user, assistant, think, call, result) with rich.",
    )
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("distill-create", help="Create a distillation run (teacher run + student arch).")
    p.add_argument("--run-name", required=True)
    p.add_argument("--teacher-run", required=True, help="Name of the (already trained) teacher run.")
    p.add_argument("--teacher-checkpoint", default="best", help="latest | best | <epoch number>")
    p.add_argument("--student-arch", required=True)
    p.add_argument("--dataset", default=None, help="Defaults to the teacher run's own dataset.")
    p.add_argument("--force", action="store_true", help="Overwrite an existing distill_config.json.")
    p.set_defaults(func=cmd_distill_create)

    p = sub.add_parser("distill", help="Train (or resume) a distillation run.")
    p.add_argument("--run-name", required=True)
    p.add_argument("--epochs", type=int, default=None, help="Override distill_config epochs for this invocation only.")
    p.set_defaults(func=cmd_distill_train)

    p = sub.add_parser(
        "scripts",
        help="Run a one-off analysis script (see `model scripts <name> --help`).",
    )
    from model.scripts import SCRIPTS

    p.add_argument("script", choices=sorted(SCRIPTS), help="Name of the script to run.")
    p.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the script.")
    p.set_defaults(func=cmd_scripts)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (cfg_mod.ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
