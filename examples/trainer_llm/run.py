import argparse
import logging
import pathlib
import subprocess
import sys
import time
from typing import Any

# Some weird lib is calling logging.basicConfig on import, so we need to be brutal


def setup_logging() -> None:
    fmt = (
        "%(asctime)s.%(msecs)03d: %(levelname).1s "
        + "%(name)s.py:%(lineno)d] %(message)s"
    )
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    module_names_verbose = [
        __name__,
        "builder",
        "configurator",
        "datasets_hf",
        "metrics",
        "dwain_wrapper_module",
        "run_decompose_dwain",
        "run_finetune",
        "utils",
        "ptdeco",
    ]
    for module_name in module_names_verbose:
        logging.getLogger(module_name).setLevel(logging.INFO)


logger = logging.getLogger(__name__)
setup_logging()


import ptdeco  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import run_decompose_dwain  # noqa: E402
import run_finetune  # noqa: E402
import version  # noqa: E402

REPRO_SUBDIR = "repro"


def parse_args() -> argparse.Namespace:
    # Try to parse --version

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--version", action="store_true")
    args = arg_parser.parse_args()

    # If no --version, run parsing of trainign/decomposition arguments

    if not args.version:
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument(
            "--output-path",
            type=pathlib.Path,
            required=True,
        )
        arg_parser.add_argument(
            "--config",
            type=pathlib.Path,
            required=True,
        )
        args = arg_parser.parse_args()

    return args


def seconds_to_hh_mm_ss_str(seconds: float) -> str:
    # NOTE: it assumes that seconds are float
    # This is because time.perf_timer() returns float
    # Frequently you want to parse time difference
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:.0f}:{m:02.0f}:{s:02.2f}"


def read_config(fname: str) -> dict[str, Any]:
    with open(fname, "rt") as f:
        return yaml.safe_load(f)


def copy_config(config_path: pathlib.Path, output_path: pathlib.Path) -> None:
    config_copy_path = output_path / REPRO_SUBDIR / "config.yaml"
    if config_copy_path.exists():
        msg = f"Config copy already exists, please delete it first, {config_copy_path}"
        raise FileExistsError(msg)
    config_copy_path.parent.mkdir(exist_ok=True, parents=True)
    with open(config_path, "rt") as f_in, open(config_copy_path, "wt") as f_out:
        f_out.write(f'ptdeco_trainer_llm_version: "{version.__version__}"\n')
        f_out.write(f'ptdeco_version: "{ptdeco.__version__}"\n\n')
        for line in f_in:
            if not line.startswith(
                "ptdeco_trainer_llm_version:"
            ) and not line.startswith("ptdeco_version:"):
                f_out.write(f"{line}")


def save_requirements(
    requirements_path: pathlib.Path, requirements_unsafe_path: pathlib.Path
) -> None:
    # Dump "normal" requirements

    result = subprocess.run(
        [sys.executable, "-mpip", "freeze"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    requirements_safe = result.stdout.decode("utf-8").splitlines()

    with requirements_path.open("wt") as f:
        f.write(f"# Python {sys.version}\n\n")
        for r in requirements_safe:
            f.write(r + "\n")

    # Dump "unsafe" requirements (rarely needed)

    result = subprocess.run(
        [sys.executable, "-mpip", "freeze", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    requirements_all = result.stdout.decode("utf-8").splitlines()

    with requirements_unsafe_path.open("wt") as f:
        f.write(f"# Python {sys.version}\n\n")
        for r in requirements_all:
            if r not in requirements_safe:
                f.write(r + "\n")


def print_version() -> None:
    print(f"Using ptdeco LLM trainer {version.__version__}")
    print(f"Using ptdeco {ptdeco.__version__}")
    print(f"Using torch {torch.__version__}")


def main(args: argparse.Namespace) -> None:
    if args.version:
        print_version()
        return

    start = time.perf_counter()
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    copy_config(args.config, output_path)
    save_requirements(
        output_path / REPRO_SUBDIR / "requirements.txt",
        output_path / REPRO_SUBDIR / "requirements_unsafe.txt",
    )
    config = read_config(args.config)
    task = config.get("task")

    logger.info(f"Using ptdeco LLM trainer {version.__version__}")
    logger.info(f"Using ptdeco {ptdeco.__version__}")
    logger.info(f"Using torch {torch.__version__}")

    if task == "decompose_dwain":
        run_decompose_dwain.main(config_raw=config, output_path=output_path)
    elif task == "finetune":
        run_finetune.main(config_raw=config, output_path=output_path)

    else:
        if task is None:
            msg = "config.train_mode unspecified"
        else:
            msg = f"Unknown config.train_mode={task}"
        raise ValueError(msg)
    duration = time.perf_counter() - start
    logger.info(f"Run took: {seconds_to_hh_mm_ss_str(duration)}")


if __name__ == "__main__":
    main(parse_args())
