
from transkun.Data import writeMidi
import transkun
import torch
import moduleconf
import numpy as np
from transkun.Util import computeParamSize
import soundfile as sf

import os
import numpy as np

def readAudio(path, normalize=True):
    """
    Reads an audio file using soundfile.

    Args:
        path (str or PathLike): Path to the audio file.
        normalize (bool): If True, audio data is returned as float32 in the range [-1.0, 1.0].
                          If False, audio data is returned in its native format (e.g., int16, int32, float32).

    Returns:
        tuple: (samplerate, audio_data)
               samplerate (int): The sample rate of the audio.
               audio_data (np.ndarray): The audio samples, always 2D (num_samples, num_channels).
    """
    if normalize:
        # When dtype is 'float32', soundfile automatically normalizes integer PCM
        # to [-1.0, 1.0]. If the file is already float PCM, it reads it directly.
        data, samplerate = sf.read(path, dtype='float32')
    else:
        # Read data in its native format (e.g., int16, int32, float32) without explicit normalization.
        data, samplerate = sf.read(path)

    # Ensure audio_data is always 2D (samples, channels), even for mono files.
    # soundfile.read returns 1D for mono audio.
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return samplerate, data

def transcribe_with_transkun(input_path: str, output_midi_path: str):
    
    import pkg_resources

    defaultWeight =  os.path.join(os.path.dirname(transkun.__file__), "pretrained/2.0.pt")
    defaultConf =  os.path.join(os.path.dirname(transkun.__file__), "pretrained/2.0.conf")
    path = defaultWeight
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO fix the conf
    confPath = defaultConf

    confManager = moduleconf.parseFromFile(confPath)
    TransKun = confManager["Model"].module.TransKun
    conf = confManager["Model"].config

    checkpoint = torch.load(path, map_location = device)
    model = TransKun(conf = conf).to(device)


    if not "best_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["best_state_dict"], strict=False)

    model.eval()


    audioPath = input_path
    outPath = str(output_midi_path)
    torch.set_grad_enabled(False)


    fs, audio= readAudio(audioPath)


    if(fs != model.fs):
        import soxr
        audio = soxr.resample(
                audio,          # 1D(mono) or 2D(frames, channels) array input
                fs,      # input samplerate
                model.fs# target samplerate
        )



    x = torch.from_numpy(audio).to(device)

    notesEst = model.transcribe(x, discardSecondHalf=False)

    outputMidi = writeMidi(notesEst)
    outputMidi.write(outPath)