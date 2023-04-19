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
import torch

from pathlib import Path

from espnet2.bin.tts_inference import Text2Speech

def load_TTS(model_dir) -> Text2Speech:
    """
    Load ESPNet2-TTS model and return it.
    """

    if "pretrain" in str(model_dir):
        model_dir = model_dir / "exp/tts_train_jets_raw_phn_null_g2pk"
    else:
        model_dir = model_dir / "exp/tts_finetune_jets_raw_phn_null_g2pk_init_parampretrained_model/tts_jets_pretrain/exp/tts_train_jets_raw_phn_null_g2pk/train.total_count.ave_5best.pth:tts:tts"


    train_config = model_dir / "config.yaml"
    model_file = model_dir / "train.total_count.ave_5best.pth"

    tts = Text2Speech(
        train_config,
        model_file,
    )
    
    return tts

def load_TTS_all():
    model_dir = Path("models/tts")

    o_TTS_dict = {}
    tts_list = list(model_dir.iterdir())
    tts_list.sort()

    for tts_model in tts_list:
        if "jets" in str(tts_model):
            o_TTS_dict[str(tts_model.name)] = load_TTS(tts_model)

    return o_TTS_dict



def synthesize(tts, text):
    wav = tts(text)["wav"]
    
    # Add Empty Tensor at end of wav for smooth TTS
    temp = torch.zeros(12000)
    output = torch.cat([wav, temp], 0)
    return output


if __name__ == "__main__":
    import sys
    from datetime import datetime
    import soundfile as sf

    model_dir = Path(sys.argv[1])
    tts = load_TTS(model_dir)
    synthesized_audio = synthesize(tts, "안녕하세요 음성합성 테스트입니다.")

    DATA_DIR = Path('.')
    now = datetime.now()
    wav_name = now.strftime("%Y-%m-%d_%H:%M:%S")
    wav_file = DATA_DIR / (wav_name + ".wav")

    sf.write(wav_file, synthesized_audio.numpy(), 24000, "PCM_16")


