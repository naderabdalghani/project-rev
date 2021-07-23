import os

from app_config import MODELS_DIR, CACHE_DIR, OUTPUT_FILENAME
from exceptions import SpeechSynthesizerCannotBeLoaded
from .config import SAVED_INSTANCE_DIR, TED_VOICE_SAMPLE_PATH
from .synthesizer.inference import Synthesizer
from .encoder import inference as encoder
from .vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play

synthesizer = None
embedding = None


def get_bot_response_as_audio(bot_response_as_text):
    global synthesizer, embedding
    if synthesizer is None or embedding is None:
        load_speech_synthesizer()
    specs = synthesizer.synthesize_spectrograms([bot_response_as_text], [embedding])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    sf.write(os.path.join(CACHE_DIR, OUTPUT_FILENAME), generated_wav, synthesizer.sample_rate)
    output_audio = AudioSegment.from_wav(os.path.join(CACHE_DIR, OUTPUT_FILENAME))
    play(output_audio)


def load_speech_synthesizer():
    global synthesizer, embedding
    try:
        encoder.load_model(Path(os.path.join(MODELS_DIR, SAVED_INSTANCE_DIR, "encoder.pt")))
        synthesizer = Synthesizer(Path(os.path.join(MODELS_DIR, SAVED_INSTANCE_DIR, "synthesizer")))
        vocoder.load_model(Path(os.path.join(MODELS_DIR, SAVED_INSTANCE_DIR, "vocoder.pt")))
        reprocessed_wav = encoder.preprocess_wav(Path(TED_VOICE_SAMPLE_PATH))
        original_wav, sampling_rate = librosa.load(Path(TED_VOICE_SAMPLE_PATH))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embedding = encoder.embed_utterance(preprocessed_wav)
    except FileNotFoundError:
        raise SpeechSynthesizerCannotBeLoaded()
