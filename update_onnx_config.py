import argparse
import yaml
import logging

from pathlib import Path
from shutil import copyfile

def load_yaml(yaml_file: Path):
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_data

def save_yaml(data, yaml_file: Path):
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def get_parser():
    """Get Argument Parser."""
    parser = argparse.ArgumentParser(
        description="Update config file of model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("./models/tts_jets_finetune"),
        help="The path of a model directory.",
    )
    
    return parser

def main():
    # Setting Logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Argument parsing
    parser = get_parser()
    args = parser.parse_args()
    model_dir = args.model_dir.absolute()

    # Load config file
    config_file = model_dir / "config.yaml"
    logging.info(f"Load config file: {config_file}")
    config_data = load_yaml(config_file)

    # Update config file
    stats_file_name = config_data["normalize"]["stats_file"].split("/")[-1]
    tts_model_name = config_data["tts_model"]["model_path"].split("/full/")[-1]

    logging.info(f"Update config data")
    config_data["normalize"]["stats_file"] = str(model_dir / stats_file_name)
    config_data["tts_model"]["model_path"] = str(model_dir / "full" / tts_model_name)

    # Copy old config file
    backup_config_file = config_file.with_name('config_back.yaml')
    logging.info(f"Copy original config file: {backup_config_file}")
    copyfile(config_file, backup_config_file)

    # Write new config file
    logging.info(f"Save new config file: {config_file}")
    save_yaml(config_data, config_file)

    logging.info(f"Done")


if __name__ == "__main__":
    main()
