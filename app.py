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

from tts import load_TTS, synthesize

# Setting directory
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

def main():
    st.header("Text-to-Speech app with streamlit")
    st.markdown(
        """
This TTS web app is using JETS model trained by kss dataset.
This app only process Korean.
        """
    )

    with st.spinner(text="Wait for loading TTS model..."):
        tts = load_TTS()

    target_text = st.text_input("Write a text to synthesize.(Must be Korean)")

    if target_text != "":
        with st.spinner(text="Wait for synthesize..."):
            synthesized_audio = synthesize(tts, target_text)

        # Save audio
        num_files = len(sorted(DATA_DIR.iterdir()))
        wav_file = DATA_DIR / (str(num_files) + '.wav')
        sf.write(wav_file, synthesized_audio.numpy(), 22050, "PCM_16")

        with open(wav_file, 'rb') as f:
            audio_bytes = f.read()
        
        st.audio(audio_bytes, format="audio/ogg")

if __name__ == "__main__":
    # Setting logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    
    logging.info("Start TTS APP")
    
    main()
