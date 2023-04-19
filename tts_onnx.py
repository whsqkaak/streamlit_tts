from espnet_onnx import Text2Speech
from pathlib import Path
import numpy as np

def load_tts_onnx(model_dir) -> Text2Speech:
    return Text2Speech(None, model_dir)

def load_tts_onnx_all():
    model_dir = Path("models/tts")

    tts_dict = {}
    tts_list = list(model_dir.iterdir())

    for tts_model in tts_list:
        if "jets" in str(tts_model) and "onnx" in str(tts_model):
            tts_dict[str(tts_model.name)] = load_tts_onnx(tts_model)

    return tts_dict

def synthesize(tts, text):
    wav = tts(text)["wav"]

    temp = np.zeros(12000)
    return np.concatenate([temp, wav, temp])
    
