#!/usr/bin/env python3
"""
Title:
    SpeechTools TTS API Server
    ESPNet2-TTS config update code.

Description:
    미세 조정된 모델에서 생성된 `config.yaml` 파일에는
    모델 파일 등 모델 구동에 필요한 파일 경로들이 담겨있다.
    
    이 코드는 본 과제의 시스템에 맞게
    `config.yaml` 파일의 파일 경로들을 수정한다.

Author: SeungHyun Lee(@SpeechTools)
Date: 2023-02-03
"""

import yaml
import logging
import argparse

from pathlib import Path
from shutil import copyfile
from typing import Dict


def load_yaml(yaml_file: Path) -> Dict[str, str]:
    """
    `yaml_file`을 읽어들여 `dict` 형태로 반환하는 함수
    
    Args:
        yaml_file: yaml 파일 경로
        
    Returns:
        읽어들인 yaml 파일을 딕셔너리 형태로 변환한 것
    """
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        
    return yaml_data


def save_yaml(data: Dict[str, str], yaml_file: Path):
    """
    딕셔너리 형태의 `data`를 yaml 형태로 저장하는 함수
    
    Args:
        data: 저장할 데이터
        yaml_file: 저장될 yaml 파일 경로
    """
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
    
    # Argument Parsing
    parser = get_parser()
    args = parser.parse_args()
    model_dir = args.model_dir
    
    # Load meta file
    meta_file = model_dir / "meta.yaml"
    logging.info(f"Load meta file: {meta_file}")
    meta = load_yaml(meta_file)
    
    config_file = model_dir / meta["yaml_files"]["train_config"]
    
    # Load config file
    logging.info(f"Load config file: {config_file}")
    config = load_yaml(config_file)
    
    # Upadte config file
    config["output_dir"] = str(model_dir) + "/" + config["output_dir"]
    
    new_train_shape_files = []
    for train_shape_file in config["train_shape_file"]:
        new_train_shape_files.append(str(model_dir) + "/" + train_shape_file)
        
    config["train_shape_file"] = new_train_shape_files
    
    new_valid_shape_files = []
    for valid_shape_file in config["valid_shape_file"]:
        new_valid_shape_files.append(str(model_dir) + "/" + valid_shape_file)
        
    config["valid_shape_file"] = new_valid_shape_files
    
    config["normalize_conf"]["stats_file"] = str(model_dir) + "/" + config["normalize_conf"]["stats_file"]
    config["pitch_normalize_conf"]["stats_file"] = str(model_dir) + "/" + config["pitch_normalize_conf"]["stats_file"]
    config["energy_normalize_conf"]["stats_file"] = str(model_dir) + "/" + config["energy_normalize_conf"]["stats_file"]
    
    # Copy old config file
    logging.info(f"Copy old config file")
    copyfile(config_file, config_file.with_name('config_back.yaml'))
    
    # Write new config file
    logging.info(f"Save new config file")
    save_yaml(config, config_file)
    
    logging.info("Done update config.")


if __name__ == "__main__":
    main()
    
