# WhisperHallu
Experimental code: sound file preprocessing to optimize Whisper transcriptions without hallucinated texts

See this discussion: https://github.com/openai/whisper/discussions/679

# Algo
- remove silences, and normalize loudness.
- remove noise parts.
- add voice markers.
- apply speech compressor.
- try to transcribe. If markers are present in output, transcription is OK.
- if not, try to invert markers. If markers are present in output, transcription is OK.
- if not, try without markers.

# Complement

May be used to produce "accurate transcriptions" for WhisperTimeSync:<br/>
https://github.com/EtienneAb3d/WhisperTimeSync

May be tested using NeuroSpell Dictaphone:<br/>
https://neurospell.com/

# Install

**Standard Whisper**

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

**Faster Whisper**

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

##### Need to be adapted for each language ####
lng="en"
prompt= "Whisper, Ok. "\
	+"A pertinent sentence for your purpose in your language. "\
	+"Ok, Whisper. Whisper, Ok. "\
	+"Ok, Whisper. Whisper, Ok. "\
	+"Please find here, an unlikely ordinary sentence. "\
	+"This is to avoid a repetition to be deleted. "\
	+"Ok, Whisper. "
path="/path/to/your/en/sound/file"

#Example 
#lng="uk"
#prompt= "Whisper, Ok. "\
#	+"Доречне речення вашою мовою для вашої мети. "\
#	+"Ok, Whisper. Whisper, Ok. "\
#	+"Ok, Whisper. Whisper, Ok. "\
#	+"Будь ласка, знайдіть тут навряд чи звичайне речення. "\
#	+"Це зроблено для того, щоб уникнути повторення, яке потрібно видалити. "\
#	+"Ok, Whisper. "
#path="/path/to/your/uk/sound/file"

loadModel("0")
result = transcribePrompt(path=path, lng=lng, prompt=prompt)
```
