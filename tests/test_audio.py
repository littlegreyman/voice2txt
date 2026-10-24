import numpy as np
import soundfile as sf
import tempfile
import os

from voice2txt.io.audio import to_wav, cleanup, load_audio


def test_to_wav_roundtrip():
    sr = 16000
    dur = 1.0
    x = (np.sin(2 * np.pi * 440 * np.arange(int(sr * dur)) / sr)).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, x, sr)
    try:
        wav = to_wav(tmp.name)
        arr, _ = load_audio(wav)
        assert abs(len(arr) / sr - dur) < 0.05
        cleanup(wav)
    finally:
        os.remove(tmp.name)
