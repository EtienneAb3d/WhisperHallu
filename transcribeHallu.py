import sys
import os
import time
import re
 
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    print("Python >= 3.10")
    import collections.abc
    import collections
    collections.MutableMapping = collections.abc.MutableMapping
else:
    print("Python < 3.10")
    import collections
    
import traceback

import torch

#This code is using FasterWhisper: https://github.com/guillaumekln/faster-whisper
from faster_whisper import WhisperModel

from threading import Lock, Thread
lock = Lock()

SAMPLING_RATE = 16000
torch.set_num_threads(1)
modelVAD, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

model_path = "whisper-medium-ct2/"#"whisper-medium-ct2/" "whisper-large-ct2/"
beam_size=2
model = None

def loadModel(gpu: str):
    print("LOADING: "+model_path+" GPU: "+gpu+" BS: "+str(beam_size))
    global model
    device="cuda" #cuda cpu
    compute_type="float16"# float16 int8_float16 int8
    model = WhisperModel(model_path, device=device,device_index=int(gpu), compute_type=compute_type)
    print("LOADED")

def transcribePrompt(path: str,lng: str,prompt: str):
    """Whisper transcribe."""
    print("=====transcribePrompt",flush=True)
    print("PATH="+path,flush=True)
    print("LNG="+lng,flush=True)
    opts = dict(language=lng,initial_prompt=prompt)
    return transcribeOpts(path, opts)

def transcribeOpts(path: str,opts: dict):
    pathIn = path
    
    initTime = time.time()
    
    startTime = time.time()
    try:
        pathSILCUT = path+".SILCUT"+".wav"
        aCmd = "ffmpeg -y -i "+pathIn+" -af \"silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.2:stop_silence=0.2, loudnorm\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" "+pathSILCUT+" > "+pathSILCUT+".log 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T=",(time.time()-startTime))
        print("PATH="+pathSILCUT,flush=True)
        pathIn = pathSILCUT
    except:
         print("Warning: can't filter blanks")
    
    startTime = time.time()
    try:
        pathVAD = pathIn+".VAD.wav"
        wav = read_audio(pathIn, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(wav, modelVAD, sampling_rate=SAMPLING_RATE)
        save_audio(pathVAD,collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)
        print("T=",(time.time()-startTime))
        print("PATH="+pathVAD,flush=True)
        pathIn = pathVAD
    except:
         print("Warning: can't filter noises")
    
    result = transcribeMARK(pathIn, opts, mode=1)
    
    if len(result["text"]) <= 0:
        result["text"] = "--"
    
    print("T=",(time.time()-startTime))
    print("s/c=",(time.time()-startTime)/len(result["text"]))
    print("c/s=",len(result["text"])/(time.time()-startTime))
    print("TOT=",(time.time()-initTime))
    
    return result["text"]

def transcribeMARK(path: str,opts: dict,mode = 1,aLast=None):
    pathIn = path
    
    lng = opts["language"]
    noMarkRE = "^(ar|he|hi|ru|zh)$"
    if(lng != None and re.match(noMarkRE,lng)):
    	#Need special voice marks
    	mode = 0;
    
    if os.path.exists("markers/WOK-MRK-"+opts["language"]+".wav"):
        mark1="markers/WOK-MRK-"+opts["language"]+".wav"
    else:
        mark1="markers/WOK-MRK.wav"
    if os.path.exists("markers/OKW-MRK-"+opts["language"]+".wav"):
        mark2="markers/OKW-MRK-"+opts["language"]+".wav"
    else:
        mark2="markers/OKW-MRK.wav"
    
    if(mode == 2):
        mark = mark1
        mark1 = mark2
        mark2 = mark
    	
    if(mode == 0):
        print("["+str(mode)+"] PATH="+pathIn,flush=True)
    else:
    	startTime = time.time()
    	try:
            pathMRK = pathIn+".MRK"+".wav"
            aCmd = "ffmpeg -y -i "+mark1+" -i "+pathIn+" -i "+mark2+" -filter_complex \"[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]\" -map \"[a]\" -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" "+pathMRK+" > "+pathMRK+".log 2>&1"
            print("CMD: "+aCmd)
            os.system(aCmd)
            pathIn = pathMRK
            
            pathCPS = pathIn+".CPS"+".wav"
            aCmd = "ffmpeg -y -i "+pathIn+" -af \"speechnorm=e=50:r=0.0005:l=1\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" "+pathCPS+" > "+pathCPS+".log 2>&1"
            print("CMD: "+aCmd)
            os.system(aCmd)
            print("T=",(time.time()-startTime))
            print("["+str(mode)+"] PATH="+pathCPS,flush=True)
            pathIn = pathCPS
    	except:
    		 print("Warning: can't add markers")
    
    startTime = time.time()
    lock.acquire()
    try:
        transcribe_options = dict(**opts)#avoid to add beam_size opt several times
        if beam_size > 1:
        	transcribe_options = dict(beam_size=beam_size,**opts)
        
        segments, info = model.transcribe(pathIn,**transcribe_options)
        result = {}
        result["text"] = ""
        for segment in segments:
            result["text"] += segment.text
        print("TRANS="+result["text"],flush=True)
    except Exception as e: 
    	print(e)
    	traceback.print_exc()
    	lock.release()
    	result = {}
    	result["text"] = ""
    	return result
    
    lock.release()
    
    if(mode == 0):
        return result
        #Too restrictive
        #if(result["text"] == aLast):
        #    #Only if confirmed
        #    return result
        #result["text"] = ""
        #return result
    
    aWhisper="(Whisper|Wisper|Wyspę|Wysper|Wispa|Уіспер|Ου ίσπερ|ウィスパー|विस्पर)"
    aOk="(o[.]?k[.]?|okay|oké|okej|Окей|οκέι|オーケー|ओके)"
    aSep="[.,!? ]*"
    if(mode == 1):
        aCleaned = re.sub(r"(^ *"+aWhisper+aSep+aOk+aSep+"|"+aOk+aSep+aWhisper+aSep+" *$)", "", result["text"], 2, re.IGNORECASE)
        if(re.match(r"^ *"+aWhisper+aSep+aOk+"("+aSep+aOk+")?"+aSep+aWhisper+aSep+" *$", result["text"], re.IGNORECASE)):
        	#Empty sound ?
        	return transcribeMARK(path, opts, mode=2)
        
        if(re.match(r"^ *"+aWhisper+aSep+aOk+aSep+".*"+aOk+aSep+aWhisper+aSep+" *$", result["text"], re.IGNORECASE)):
        	#GOOD!
        	result["text"] = aCleaned
        	return result
        
        return transcribeMARK(path, opts, mode=2,aLast=aCleaned)
    
    if(mode == 2):
        aCleaned = re.sub(r"(^ *"+aOk+aSep+aWhisper+aSep+"|"+aWhisper+aSep+aOk+aSep+" *$)", "", result["text"], 2, re.IGNORECASE)
        if(aCleaned == aLast):
            #CONFIRMED!
            result["text"] = aCleaned
            return result
            
        if(re.match(r"^ *"+aOk+aSep+aWhisper+"("+aSep+aWhisper+")?"+aSep+aOk+aSep+" *$", result["text"], re.IGNORECASE)):
        	#Empty sound ? Confirmed...
        	result["text"] = ""
        	return result
        
        if(re.match(r"^ *"+aOk+aSep+aWhisper+aSep+".*"+aWhisper+aSep+aOk+aSep+" *$", result["text"], re.IGNORECASE)):
        	#GOOD!
        	result["text"] = aCleaned
        	return result
        
        return transcribeMARK(path, opts, mode=0,aLast=aCleaned)

