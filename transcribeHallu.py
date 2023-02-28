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

torch.set_num_threads(1)
useSileroVAD=False
if(useSileroVAD):
    modelVAD, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

useSpleeter=False
if(useSpleeter):
    from spleeter.audio import STFTBackend
    backend = STFTBackend.LIBROSA
    from spleeter.separator import Separator
    print("Using spleeter:2stems-16kHz")
    separator = Separator('spleeter:2stems-16kHz',stft_backend=backend)

useDemucs=True
if(useDemucs):
    import subprocess
    print("Using Demucs")

useCompressor=True

try:
    #Standard Whisper: https://github.com/openai/whisper
    import whisper
    print("Using standard Whisper")
    whisperFound = "STD"
except ImportError as e:
    pass

try:
    #FasterWhisper: https://github.com/guillaumekln/faster-whisper
    from faster_whisper import WhisperModel
    print("Using Faster Whisper")
    whisperFound = "FSTR"
    modelPath = "whisper-medium-ct2/"#"whisper-medium-ct2/" "whisper-large-ct2/"
    if not os.path.exists(modelPath):
        print("Faster installation found, but "+modelPath+" model not found")
        sys.exit(-1)
except ImportError as e:
    pass

beam_size=2
model = None
device = "cuda" #cuda / cpu
cudaIdx = 0

SAMPLING_RATE = 16000

from threading import Lock, Thread
lock = Lock()

def loadModel(gpu: str,modelSize=None):
    global model
    global device
    global cudaIdx
    cudaIdx = gpu
    try:
        if whisperFound == "FSTR":
            if(modelSize == "large"):
                modelPath = "whisper-large-ct2/"
            else:
                modelPath = "whisper-medium-ct2/"
            print("LOADING: "+modelPath+" GPU: "+gpu+" BS: "+str(beam_size))
            compute_type="float16"# float16 int8_float16 int8
            model = WhisperModel(modelPath, device=device,device_index=int(gpu), compute_type=compute_type)
        elif whisperFound == "STD":
            if(modelSize == None):
                modelSize="medium"#"tiny"#"medium" #"large"
            print("LOADING: "+modelSize+" GPU:"+gpu+" BS: "+str(beam_size))
            model = whisper.load_model(modelSize,device=torch.device("cuda:"+gpu)) #May be "cpu"
        print("LOADED")
    except:
        print("Can't load Whisper model: "+modelSize)
        sys.exit(-1)

def getDuration(aLog:str):
    with open(aLog) as f:
        lines = f.readlines()
        for line in lines:
            if(re.match(r"^ *Duration: [0-9][0-9]:[0-9][0-9]:[0-9][0-9][.][0-9][0-9], .*$", line, re.IGNORECASE)):
                duration = re.sub(r"(^ *Duration: *|[,.].*$)", "", line, 2, re.IGNORECASE)
                return sum(x * int(t) for x, t in zip([3600, 60, 1], duration.split(":")))
    
def getPrompt(lng:str):
    if(lng == "en"):
        aOk=""
        return "Whisper, Ok. "\
            +"A pertinent sentence for your purpose in your language. "\
            +"Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. "\
            +"Please find here, an unlikely ordinary sentence. "\
            +"This is to avoid a repetition to be deleted. "\
            +"Ok, Whisper. "
    
    if(lng == "fr"):
        return "Whisper, Ok. "\
            +"Une phrase pertinente pour votre propos dans votre langue. "\
            +"Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. "\
            +"Merci de trouver ci-joint, une phrase ordinaire improbable. "\
            +"Pour éviter une répétition à être supprimée. "\
            +"Ok, Whisper. "
    
    if(lng == "uk"):
        return "Whisper, Ok. "\
            +"Доречне речення вашою мовою для вашої мети. "\
            +"Ok, Whisper. Whisper, Ok. Ok, Whisper. Whisper, Ok. "\
            +"Будь ласка, знайдіть тут навряд чи звичайне речення. "\
            +"Це зроблено для того, щоб уникнути повторення, яке потрібно видалити. "\
            +"Ok, Whisper. "
    
    if(lng == "hi"):
        return "विस्पर, ओके. "\
            +"आपकी भाषा में आपके उद्देश्य के लिए एक प्रासंगिक वाक्य। "\
            +"ओके, विस्पर. विस्पर, ओके. ओके, विस्पर. विस्पर, ओके. "\
            +"कृपया यहां खोजें, एक असंभावित सामान्य वाक्य। "\
            +"यह हटाए जाने की पुनरावृत्ति से बचने के लिए है। "\
            +"ओके, विस्पर. "
    
    #Not Already defined?
    return ""


def transcribePrompt(path: str,lng: str,prompt=None,lngInput=None,isMusic=False):
    """Whisper transcribe."""

    if(lngInput == None):
        lngInput=lng
        print("Using output language as input language: "+lngInput)
    
    if(prompt == None):
        if(not isMusic):
            prompt=getPrompt(lng)
        else:
            prompt="";
    
    print("=====transcribePrompt",flush=True)
    print("PATH="+path,flush=True)
    print("LNGINPUT="+lngInput,flush=True)
    print("LNG="+lng,flush=True)
    print("PROMPT="+prompt,flush=True)
    opts = dict(language=lng,initial_prompt=prompt)
    return transcribeOpts(path, opts,lngInput,isMusic=isMusic)

def transcribeOpts(path: str,opts: dict,lngInput=None,isMusic=False):
    pathIn = path
    
    initTime = time.time()
    
    if(useSpleeter):
        startTime = time.time()
        try:
            spleeterDir=pathIn+".spleeter"
            if(not os.path.exists(spleeterDir)):
                os.mkdir(spleeterDir)
            pathSpleeter=spleeterDir+"/"+os.path.splitext(os.path.basename(pathIn))[0]+"/vocals.wav"
            separator.separate_to_file(pathIn, spleeterDir)
            print("T=",(time.time()-startTime))
            print("PATH="+pathSpleeter,flush=True)
            pathIn = pathSpleeter
        except:
             print("Warning: can't split vocals")
    
    if(useDemucs):
        startTime = time.time()
        try:
            demucsDir=pathIn+".demucs"
            if(not os.path.exists(demucsDir)):
                os.mkdir(demucsDir)
            pathDemucs=demucsDir+"/htdemucs/"+os.path.splitext(os.path.basename(pathIn))[0]+"/vocals.wav"
            #Demucs seems complex, using CLI cmd for now
            aCmd = "python -m demucs --two-stems=vocals -d "+device+":"+cudaIdx+" --out "+demucsDir+" "+pathIn
            print("CMD: "+aCmd)
            os.system(aCmd)
            print("T=",(time.time()-startTime))
            print("PATH="+pathDemucs,flush=True)
            pathIn = pathDemucs
        except:
             print("Warning: can't split vocals")

    duration = -1
    startTime = time.time()
    try:
        pathSILCUT = pathIn+".SILCUT"+".wav"
        aCmd = "ffmpeg -y -i "+pathIn+" -af \"silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.2:stop_silence=0.2, loudnorm\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" "+pathSILCUT+" > "+pathSILCUT+".log 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T=",(time.time()-startTime))
        print("PATH="+pathSILCUT,flush=True)
        pathIn = pathSILCUT
        duration = getDuration(pathSILCUT+".log")
        print("DURATION="+str(duration))
    except:
         print("Warning: can't filter blanks")
    
    if(not isMusic and useSileroVAD):
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
    
    mode=1
    if(duration > 30):
        print("NOT USING MARKS FOR DURATION > 30s")
        mode=0
    
    startTime = time.time()
    result = transcribeMARK(pathIn, opts, mode=mode,lngInput=lngInput,isMusic=isMusic)
    
    if len(result["text"]) <= 0:
        result["text"] = "--"
    
    print("T=",(time.time()-initTime))
    print("s/c=",(time.time()-initTime)/len(result["text"]))
    print("c/s=",len(result["text"])/(time.time()-initTime))
    
    return result["text"]

def transcribeMARK(path: str,opts: dict,mode = 1,lngInput=None,aLast=None,isMusic=False):
    pathIn = path
    
    lng = opts["language"]
    
    if(lngInput == None):
        lngInput = lng
        
    noMarkRE = "^(ar|he|ru|zh)$"
    if(lng != None and re.match(noMarkRE,lng)):
        #Need special voice marks
        mode = 0

    if(isMusic):
        #Markers are not really interesting with music
        mode = 0
        
    if os.path.exists("markers/WOK-MRK-"+lngInput+".wav"):
        mark1="markers/WOK-MRK-"+lngInput+".wav"
    else:
        mark1="markers/WOK-MRK.wav"
    if os.path.exists("markers/OKW-MRK-"+lngInput+".wav"):
        mark2="markers/OKW-MRK-"+lngInput+".wav"
    else:
        mark2="markers/OKW-MRK.wav"
    
    if(mode == 2):
        mark = mark1
        mark1 = mark2
        mark2 = mark
        
    if(mode == 0):
        print("["+str(mode)+"] PATH="+pathIn,flush=True)
    else:
        try:
            startTime = time.time()
            pathMRK = pathIn+".MRK"+".wav"
            aCmd = "ffmpeg -y -i "+mark1+" -i "+pathIn+" -i "+mark2+" -filter_complex \"[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]\" -map \"[a]\" -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" "+pathMRK+" > "+pathMRK+".log 2>&1"
            print("CMD: "+aCmd)
            os.system(aCmd)
            print("T=",(time.time()-startTime))
            print("["+str(mode)+"] PATH="+pathMRK,flush=True)
            pathIn = pathMRK
            
            if(useCompressor):
                startTime = time.time()
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
        
        if whisperFound == "FSTR":
            segments, info = model.transcribe(pathIn,**transcribe_options)
            result = {}
            result["text"] = ""
            for segment in segments:
                result["text"] += segment.text
        else:
            transcribe_options = dict(task="transcribe", **transcribe_options)
            result = model.transcribe(pathIn,**transcribe_options)
        
        print("T=",(time.time()-startTime))
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
    
    aWhisper="(Whisper|Wisper|Wyspę|Wysper|Wispa|Уіспер|Ου ίσπερ|위스퍼드|ウィスパー|विस्पर|विसपर)"
    aOk="(o[.]?k[.]?|okay|oké|okej|Окей|οκέι|오케이|オーケー|ओके)"
    aSep="[.,!? ]*"
    if(mode == 1):
        aCleaned = re.sub(r"(^ *"+aWhisper+aSep+aOk+aSep+"|"+aOk+aSep+aWhisper+aSep+" *$)", "", result["text"], 2, re.IGNORECASE)
        if(re.match(r"^ *("+aOk+"|"+aSep+"|"+aWhisper+")*"+aWhisper+"("+aOk+"|"+aSep+"|"+aWhisper+")* *$", result["text"], re.IGNORECASE)):
            #Empty sound ?
            return transcribeMARK(path, opts, mode=2,lngInput=lngInput,aLast="")
        
        if(re.match(r"^ *"+aWhisper+aSep+aOk+aSep+".*"+aOk+aSep+aWhisper+aSep+" *$", result["text"], re.IGNORECASE)):
            #GOOD!
            result["text"] = aCleaned
            return result
        
        return transcribeMARK(path, opts, mode=2,lngInput=lngInput,aLast=aCleaned)
    
    if(mode == 2):
        aCleaned = re.sub(r"(^ *"+aOk+aSep+aWhisper+aSep+"|"+aWhisper+aSep+aOk+aSep+" *$)", "", result["text"], 2, re.IGNORECASE)
        if(aCleaned == aLast):
            #CONFIRMED!
            result["text"] = aCleaned
            return result
            
        if(re.match(r"^ *("+aOk+"|"+aSep+"|"+aWhisper+")*"+aWhisper+"("+aOk+"|"+aSep+"|"+aWhisper+")* *$", result["text"], re.IGNORECASE)):
            #Empty sound ? 
            result["text"] = ""
            return result
        
        if(re.match(r"^ *"+aOk+aSep+aWhisper+aSep+".*"+aWhisper+aSep+aOk+aSep+" *$", result["text"], re.IGNORECASE)):
            #GOOD!
            result["text"] = aCleaned
            return result
        
        return transcribeMARK(path, opts, mode=0,lngInput=lngInput,aLast=aCleaned)

