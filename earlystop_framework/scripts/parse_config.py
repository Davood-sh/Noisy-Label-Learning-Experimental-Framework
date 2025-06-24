import argparse
import json

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a specified model with a JSON config"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the JSON config file"
    )
    return parser.parse_args()


def load_config(path):
    """
    Load and return the JSON config as a dict.
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def get_model(cfg):
    """
    Dispatch to the correct model factory based on cfg["model"].

    Expects:
      - cfg["model"]: "elr" or "sop"
      - cfg["num_classes"]: int
      - optional cfg["elr_args"] or cfg["sop_args"] dict for extra kwargs
    """
    model_name = cfg.get("model", "").lower()
    num_classes = cfg.get("num_classes")
    if model_name == "elr":
        from wrappers.elr_wrapper import elr
        return elr(num_classes, **cfg.get("elr_args", {}))

    elif model_name == "sop":
        from wrappers.sop_wrapper import sop
        return sop(num_classes, **cfg.get("sop_args", {}))

    else:
        raise ValueError(
            f"Unknown model type '{cfg.get('model')}'. "
            "Supported: 'elr', 'sop'."
        )


if __name__ == "__main__":
    # Sanity check: try building the model from a config
    args = parse_args()
    cfg = load_config(args.config)
    model = get_model(cfg)
    print(f"Built model: {model}")
