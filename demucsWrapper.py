import torch
import torchaudio
import demucs
from demucs.pretrained import get_model_from_args
from demucs.apply import apply_model
from demucs.separate import load_track
from torch._C import device

def load_demucs_model():
    return get_model_from_args(type('args', (object,), dict(name='htdemucs', repo=None))).cpu().eval()


def demucs_audio(pathIn: str,
                 model=None,
                 device=None,
                 pathVocals: str = None,
                 pathOther: str = None):
    if model is None:
        model = load_demucs_model()

    audio = load_track(pathIn, model.audio_channels, model.samplerate)

    audio_dims = audio.dim()
    if audio_dims == 1:
        audio = audio[None, None].repeat_interleave(2, -2)
    else:
        if audio.shape[-2] == 1:
            audio = audio.repeat_interleave(2, -2)
        if audio_dims < 3:
            audio = audio[None]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Demucs using device: "+device)
    result = apply_model(model, audio, device=device, split=True, overlap=.25)
    if device != 'cpu':
        torch.cuda.empty_cache()
    
    for name in model.sources:
        print("Source: "+name)
        source_idx=model.sources.index(name)
        source=result[0, source_idx].mean(0)
        torchaudio.save(pathIn+"."+name+".wav", source[None], model.samplerate)
        

