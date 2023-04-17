# WhisperHallu
Experimental code: sound file preprocessing to optimize Whisper transcriptions without hallucinated texts

See this discussion: https://github.com/openai/whisper/discussions/679

# Algo
- remove noise by voice extraction using  [Facebook Demucs](https://github.com/facebookresearch/demucs) or [Deezer Spleeter](https://github.com/deezer/spleeter)
- remove silences, and normalize loudness with ffmpeg.
- remove noise parts using [Silero VAD](https://github.com/snakers4/silero-vad).
- add voice markers.
- apply speech compressor (requires `ffmpeg` 4.4, while Google Colab is 4.2, it has to be upgraded, see below).
- try to transcribe. If markers are present in output, transcription is OK.
- if not, try to invert markers. If markers are present in output, transcription is OK.
- if not, try without markers.

# Complement

May be used to produce "accurate transcriptions" for WhisperTimeSync:<br/>
https://github.com/EtienneAb3d/WhisperTimeSync

May be tested using NeuroSpell Dictaphone:<br/>
https://neurospell.com/

WhisperHallu and WhisperTimeSync are used to extract vocals and lyrics in karaok-AI:<br/>
https://github.com/EtienneAb3d/karaok-AI


# Google Colab

Standard Whisper:<br/>
https://colab.research.google.com/drive/1-GpXaNaGFXKX9VXl60JGVVrGO41t09KA?usp=sharing

Faster Whisper:<br/>
https://colab.research.google.com/drive/1RkvOtUTbUD5NVsRI4aKEqJO8BRo8BFIY?usp=sharing

# Install

**Upgrade ffmpeg to version 4.4 on Google Colab**
```
! add-apt-repository -y ppa:savoury1/ffmpeg4
! apt-get -qq install -y ffmpeg

!ffmpeg -version

Output:
==========
ffmpeg version 4.4.3-0ubuntu1~20.04.sav2 Copyright (c) 2000-2022 the FFmpeg developers
[...]
```

**Demucs (if used)**

```
pip install -U demucs
```

**Spleeter (if used)**

```
pip install spleeter
```

**Standard Whisper (if used)**

```
sudo apt update && sudo apt install ffmpeg

sudo apt install python3
sudo apt install python3-pip
sudo apt install virtualenv

virtualenv -p python3 ../venvWhisper
. ../venvWhisper/bin/activate

pip install -U openai-whisper

pip3 install torchaudio
```

**Faster Whisper (if used)**

```
sudo apt update && sudo apt install ffmpeg

sudo apt install python3
sudo apt install python3-pip
sudo apt install virtualenv

virtualenv -p python3 ../venvFasterWhisper
. ../venvFasterWhisper/bin/activate

git clone https://github.com/guillaumekln/faster-whisper.git
cd faster-whisper/

pip install -e .[conversion]
pip install -e .

cd ..

ct2-transformers-converter --model openai/whisper-medium --output_dir whisper-medium-ct2 --quantization float16
ct2-transformers-converter --model openai/whisper-large --output_dir whisper-large-ct2 --quantization float16

pip3 install torchaudio
```

# Code

```
from transcribeHallu import loadModel
from transcribeHallu import transcribePrompt

##### The audio language may be different from the one for the output transcription.
path="/path/to/your/en/sound/file"
lngInput="en"

##### Activate this for music file to get a minimal processing
isMusic=False

##### Need to be adapted for each language.
##### For prompt examples, see transcribeHallu.py getPrompt(lng:str)
lng="en"
prompt= "Whisper, Ok. "\
	+"A pertinent sentence for your purpose in your language. "\
	+"Ok, Whisper. Whisper, Ok. "\
	+"Ok, Whisper. Whisper, Ok. "\
	+"Please find here, an unlikely ordinary sentence. "\
	+"This is to avoid a repetition to be deleted. "\
	+"Ok, Whisper. "

##### Model size to use
modelSize="medium"
loadModel("0",modelSize=modelSize)

result = transcribePrompt(path=path, lng=lng, prompt=prompt, lngInput=lngInput,isMusic=isMusic)
```

<hr>
This tool is a demonstration of our know-how.<br/>
If you are interested in a commercial/industrial AI linguistic project, contact us:<br/>
https://cubaix.com
