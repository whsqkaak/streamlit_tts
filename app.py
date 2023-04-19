#!/usr/bin/env python3
"""
Title:
    ESPNet2-TTS Web application using Streamlit.

Description:
    This is a main source code of TTS web app using Streamlit. 
    The TTS model is JETS.
    The TTS model was trained using ESPNet2.

Author: SeungHyun Lee(@SpeechTools)
Date: 2022-10-13
"""

import logging
import soundfile as sf
import streamlit as st

from pathlib import Path

from tts_onnx import load_tts_onnx, synthesize, load_tts_onnx_all
#from tts import load_TTS, synthesize, load_TTS_all

# Setting directory
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

@st.cache_resource
def call_load_TTS():
    tts_dict = load_tts_onnx_all()
    return tts_dict

def main():
    st.header("Text-to-Speech app with streamlit")
    st.markdown(
        """
This app only process Korean.
        """
    )

    with st.spinner(text="Wait for loading TTS model..."):
        tts_dict = call_load_TTS()

    target_text = st.text_input("Write a text to synthesize.(Must be Korean)")

    if target_text != "":
        for key, tts in tts_dict.items():
            with st.spinner(text="Wait for synthesize..."):
                synthesized_audio = synthesize(tts, target_text)

            # Save audio
            num_files = len(sorted(DATA_DIR.iterdir()))
            wav_file = DATA_DIR / (str(num_files) + '.wav')
            sf.write(wav_file, synthesized_audio, 24000, "PCM_16")

            with open(wav_file, 'rb') as f:
                audio_bytes = f.read()
        
            st.header(key)
            st.audio(audio_bytes, format="audio/wav")

if __name__ == "__main__":
    # Setting logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    
    logging.info("Start TTS APP")
    
    main()
