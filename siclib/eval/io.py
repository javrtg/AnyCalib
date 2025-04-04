import argparse
from pathlib import Path
from pprint import pprint
from typing import Optional

import pkg_resources
from hydra import compose, initialize
from omegaconf import OmegaConf

from siclib.models import get_model
from siclib.settings import TRAINING_PATH
from siclib.utils.experiments import load_experiment

# flake8: noqa
# mypy: ignore-errors


def parse_config_path(name_or_path: Optional[str], defaults: str) -> Path:
    default_configs = {}
    print(f"Looking for default config: {'siclib', str(defaults)}")
    for c in pkg_resources.resource_listdir("siclib.eval", str(defaults)):
        if c.endswith(".yaml"):
            default_configs[Path(c).stem] = Path(
                pkg_resources.resource_filename("siclib.eval", defaults + c)
            )
    if name_or_path is None:
        return None
    if name_or_path in default_configs:
        return default_configs[name_or_path]
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the config file: {name_or_path}. "
            f"Not in the default configs {list(default_configs.keys())} "
            "and not an existing path."
        )
    return Path(path)


def extract_benchmark_conf(conf, benchmark, only_model=True):
    if only_model:
        conf_ = OmegaConf.create({"model": conf.get("model", {})})
    else:
        conf_ = conf
        OmegaConf.set_struct(conf_, None)
    # mconf = OmegaConf.create({"model": conf.get("model", {})})
    if "benchmarks" in conf.keys():
        return OmegaConf.merge(conf_, conf.benchmarks.get(benchmark, {}))
        # return OmegaConf.merge(mconf, conf.benchmarks.get(benchmark, {}))
    else:
        return conf_
        # return mconf


def parse_eval_args(benchmark, args, configs_path, default=None, only_custom_model=True):
    conf = {"data": {}, "model": {}, "eval": {}}

    if args.conf:
        print(f"Loading config: {configs_path}")
        conf_path = parse_config_path(args.conf, configs_path)
        initialize(version_base=None, config_path=configs_path)
        custom_conf = compose(config_name=args.conf)
        conf = extract_benchmark_conf(
            OmegaConf.merge(conf, custom_conf), benchmark, only_custom_model
        )
        args.tag = args.tag if args.tag is not None else conf_path.name.replace(".yaml", "")

    cli_conf = OmegaConf.from_cli(args.dotlist)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.checkpoint = args.checkpoint or conf.get("checkpoint")

    if conf.checkpoint and not conf.checkpoint.endswith(".tar"):
        checkpoint_conf = OmegaConf.load(TRAINING_PATH / conf.checkpoint / "config.yaml")
        conf = OmegaConf.merge(extract_benchmark_conf(checkpoint_conf, benchmark), conf)

    if default:
        conf = OmegaConf.merge(default, conf)

    if args.tag is not None:
        name = args.tag
    elif args.conf and conf.checkpoint:
        name = f"{args.conf}_{conf.checkpoint}"
    elif args.conf:
        name = args.conf
    elif conf.checkpoint:
        name = conf.checkpoint
    if len(args.dotlist) > 0 and not args.tag:
        name = f"{name}_" + ":".join(args.dotlist)

    print("Running benchmark:", benchmark)
    print("Experiment tag:", name)
    print("Config:")
    pprint(OmegaConf.to_container(conf))
    return name, conf


def load_model(model_conf, checkpoint, get_last=False):
    if checkpoint:
        model = load_experiment(checkpoint, conf=model_conf, get_last=get_last).eval()
    else:
        model = get_model(model_conf.name)(model_conf).eval()
    return model


def get_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_eval", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    return parser
