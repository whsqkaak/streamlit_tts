#!/usr/bin/env python3
"""
Title:
    ESPNet2-TTS Web application using Streamlit.

Description:
    This is a TTS code of TTS web app using Streamlit. 
    The TTS model is JETS.
    The TTS model was trained using ESPNet2.

Author: SeungHyun Lee(@SpeechTools)
Date: 2022-10-13
"""

from pathlib import Path

from espnet2.bin.tts_inference import Text2Speech

def load_TTS() -> Text2Speech:
    """
    Load ESPNet2-TTS model and return it.
    """

    model_dir = Path("models/tts_jets_pretrain/exp/tts_train_jets_raw_phn_null_g2pk")

    train_config = model_dir / "config.yaml"
    model_file = model_dir / "train.total_count.ave_5best.pth"

    tts = Text2Speech(
        train_config,
        model_file,
    )
    
    return tts

def synthesize(tts, text):
    return tts(text)["wav"]





