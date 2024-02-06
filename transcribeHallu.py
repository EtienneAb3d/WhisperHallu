import sys
import os
import time
import re
from _io import StringIO
 
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
useSileroVAD=True
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
    from demucsWrapper import load_demucs_model
    from demucsWrapper import demucs_audio
    print("Using Demucs")
    modelDemucs = load_demucs_model()

useCompressor=True

try:
    #Standard Whisper: https://github.com/openai/whisper
    import whisper
    print("Using standard Whisper")
    whisperFound = "STD"
    from pathlib import Path
    from whisper.utils import WriteSRT
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

try:
    from seamless_communication.models.inference import Translator
    from lang2to3 import lang2to3
    lang2to3 = lang2to3()
    whisperFound = "SM4T"
except ImportError as e:
    pass

#large-v3 model seems to be bad with music, thus keep v2 as the default
whisperVersion = "-v2" #May be "", "-V1", "-v2, "-v3"
whisperLoaded = "??"
beam_size=2
patience=0
temperature=0
model = None
device = "cuda" #cuda / cpu
cudaIdx = 0

SAMPLING_RATE = 16000
MAX_DURATION = 600
TRUNC_DURATION = MAX_DURATION

from threading import Lock, Thread
lock = Lock()

def loadModel(gpu: str,modelSize=None):
    global model
    global device
    global cudaIdx
    global whisperLoaded
    cudaIdx = gpu
    try:
        if whisperFound == "FSTR":
            if(modelSize == "large"):
                modelPath = "whisper-large-ct2/"
            else:
                modelPath = "whisper-medium-ct2/"
            print("LOADING: "+modelPath+" GPU: "+gpu+" BS: "+str(beam_size)+" PTC="+str(patience)+" TEMP="+str(temperature))
            compute_type="float16"# float16 int8_float16 int8
            model = WhisperModel(modelPath, device=device,device_index=int(gpu), compute_type=compute_type)
        elif whisperFound == "STD":
            if(modelSize == None):
                modelSize="medium"#"tiny"#"medium" #"large"
            if(modelSize == "large"):
                modelSize = "large"+whisperVersion #"large-v1" "large-v2" "large-v3"
            print("LOADING: "+modelSize+" GPU:"+gpu+" BS: "+str(beam_size)+" PTC="+str(patience)+" TEMP="+str(temperature))
            model = whisper.load_model(modelSize,device=torch.device("cuda:"+gpu)) #May be "cpu"
        elif whisperFound == "SM4T":
            print("LOADING: "+"seamlessM4T_large"+" GPU:"+gpu)
            model = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cuda:"+gpu), torch.float16)
        print("LOADED")
        whisperLoaded = modelSize
    except Exception as e:
        print("Can't load Whisper model: "+whisperFound+"/"+modelSize)
        print(e)
        sys.exit(-1)

def loadedModel():
    return whisperFound+" "+whisperLoaded

def getDuration(aLog:str):
    duration = None
    time = None
    with open(aLog) as f:
        lines = f.readlines()
        for line in lines:
            if(re.match(r"^ *Duration: [0-9][0-9]:[0-9][0-9]:[0-9][0-9][.][0-9][0-9], .*$", line, re.IGNORECASE)):
                duration = re.sub(r"(.*Duration: *|[,. ].*)", "", line, re.IGNORECASE)
                return sum(x * int(t) for x, t in zip([3600, 60, 1], duration.split(":")))
            for aSub in line.split("[\r\n]"):
                if(re.match(r"^.*time=[0-9][0-9]:[0-9][0-9]:[0-9][0-9][.][0-9][0-9] .*$", aSub, re.IGNORECASE)):
                    #print("SUB="+aSub)
                    time = re.sub(r"(.*time=|[,. ].*)", "", aSub, re.IGNORECASE)
    #Return last found time value
    if(time != "00:00:00"):
        print("TIME="+str(time))
        return sum(x * int(t) for x, t in zip([3600, 60, 1], time.split(":")))
    return None

def formatTimeStamp(aT=0):
    aH = int(aT/3600)
    aM = int((aT%3600)/60)
    aS = (aT%60)
    return "%02d:%02d:%06.3f" % (aH,aM,aS)

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


def transcribePrompt(path: str,lng: str,prompt=None,lngInput=None,isMusic=False,addSRT=False,truncDuration=TRUNC_DURATION,maxDuration=MAX_DURATION):
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
    return transcribeOpts(path, opts,lngInput,isMusic=isMusic,addSRT=addSRT,subEnd=truncDuration,maxDuration=maxDuration)

def transcribeOpts(path: str,opts: dict
                   ,lngInput=None,isMusic=False,onlySRT=False,addSRT=False
                   ,subBeg="0",subEnd=str(TRUNC_DURATION)
                   ,maxDuration=MAX_DURATION
                   ,stretch=None
                   ,nbRun=1#Whisper is unstable, especially with music. Multiple run can provide with better results to eval afterward
                   ,remixFactor="0.3",speechnorm=True
                   ):
    pathIn = path
    pathClean = path
    pathNoCut = path
    
    initTime = time.time()
    
    startTime = time.time()
    duration = -1
    try:
        #Convert to WAV to avoid later possible decoding problem
        pathWAV = pathIn+".WAV"+".wav"
        aCmd = "ffmpeg -y"+" -i \""+pathIn+"\""+" -ss "+subBeg+" -to "+subEnd + " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathWAV+"\" > \""+pathWAV+".log\" 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        duration = getDuration(pathWAV+".log")
        print("T=",(time.time()-startTime))
        print("DURATION="+str(duration)+" subBeg="+str(subBeg)+" subEnd="+str(subEnd))
        print("PATH="+pathWAV,flush=True)
        pathIn = pathClean = pathWAV
    except Exception as e:
         print("Warning: can't convert to WAV")
         print(e)

    try:
        if(stretch != None):
            pathSTRETCH = pathIn+".STRETCH"+".wav"
            #ffmpeg STRECH
            aCmd = "ffmpeg -y -i \""+pathIn+"\""+" -t "+str(truncDuration) + " -filter:a \"atempo="+stretch+"\"" + " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathSTRETCH+"\" > \""+pathSTRETCH+".log\" 2>&1"
            #sox STRECH
            #aCmd = "sox \""+pathIn+"\""+" \""+pathSTRETCH+"\" tempo "+stretch+" > \""+pathSTRETCH+".log\" 2>&1"
            #soundstretch STRECH
            #aCmd = "soundstretch \""+pathIn+"\""+" \""+pathSTRETCH+"\" -tempo="+str(int(100*float(stretch)) - 100)+" > \""+pathSTRETCH+".log\" 2>&1"
            #rubberband STRECH
            #aCmd = "rubberband \""+pathIn+"\""+" \""+pathSTRETCH+"\" --tempo "+stretch+" > \""+pathSTRETCH+".log\" 2>&1"
            print("CMD: "+aCmd)
            os.system(aCmd)
            print("T=",(time.time()-startTime))
            print("PATH="+pathWAV,flush=True)
            pathIn = pathClean = pathWAV = pathSTRETCH
    except Exception as e:
         print("Warning: can't STRETCH")
         print(e)

    startTime = time.time()
    try:
        #Check for duration
        aCmd = "ffmpeg -y -i \""+pathIn+"\" "+ " -f null - > \""+pathIn+".dur\" 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T=",(time.time()-startTime))
        duration = getDuration(pathIn+".dur")
        print("DURATION="+str(duration)+" max "+str(maxDuration))
        if(duration > maxDuration):
            return "[Too long ("+str(duration)+"s)]"
    except Exception as e:
         print("Warning: can't analyze duration")
         print(e)

    try:
        if(useSpleeter):
            startTime = time.time()
            spleeterDir=pathIn+".spleeter"
            if(not os.path.exists(spleeterDir)):
                os.mkdir(spleeterDir)
            pathSpleeter=spleeterDir+"/"+os.path.splitext(os.path.basename(pathIn))[0]+"/vocals.wav"
            separator.separate_to_file(pathIn, spleeterDir)
            print("T=",(time.time()-startTime))
            print("PATH="+pathSpleeter,flush=True)
            pathNoCut = pathIn = pathSpleeter
    except Exception as e:
         print("Warning: can't split vocals")
         print(e)
    
    if(useDemucs):
        startTime = time.time()
        try:
            #demucsDir=pathIn+".demucs"
            #if(not os.path.exists(demucsDir)):
            #    os.mkdir(demucsDir)
            pathDemucsVocals=pathIn+".vocals.wav" #demucsDir+"/htdemucs/"+os.path.splitext(os.path.basename(pathIn))[0]+"/vocals.wav"
            pathDemucsDrums=pathIn+".drums.wav"
            pathDemucsBass=pathIn+".bass.wav"
            pathDemucsOther=pathIn+".other.wav"
            #Demucs seems complex, using CLI cmd for now
            #aCmd = "python -m demucs --two-stems=vocals -d "+device+":"+cudaIdx+" --out "+demucsDir+" "+pathIn
            #print("CMD: "+aCmd)
            #os.system(aCmd)
            demucs_audio(pathIn=pathIn,model=modelDemucs,device="cuda:"+cudaIdx,pathVocals=pathDemucsVocals,pathOther=pathIn+".other.wav")
            print("T=",(time.time()-startTime))
            print("PATH="+pathDemucsVocals,flush=True)
            pathNoCut = pathIn = pathDemucsVocals
        except Exception as e:
             print("Warning: can't split vocals")
             print(e)

    startTime = time.time()
    try:
        pathSILCUT = pathIn+".SILCUT"+".wav"
        aCmd = "ffmpeg -y -i \""+pathIn+"\" -af \"silenceremove=start_periods=1:stop_periods=-1:start_threshold=-50dB:stop_threshold=-50dB:start_silence=0.2:stop_silence=0.2, loudnorm\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathSILCUT+"\" > \""+pathSILCUT+".log\" 2>&1"
        print("CMD: "+aCmd)
        os.system(aCmd)
        print("T=",(time.time()-startTime))
        print("PATH="+pathSILCUT,flush=True)
        pathIn = pathSILCUT
    except Exception as e:
         print("Warning: can't filter blanks")
         print(e)
    
    try:
        if(not isMusic and useSileroVAD):
            startTime = time.time()
            
            pathVAD = pathIn+".VAD.wav"
            wav = read_audio(pathIn, sampling_rate=SAMPLING_RATE)
            #https://github.com/snakers4/silero-vad/blob/master/utils_vad.py#L161
            speech_timestamps = get_speech_timestamps(wav, modelVAD,threshold=0.5,min_silence_duration_ms=500, sampling_rate=SAMPLING_RATE)
            save_audio(pathVAD,collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)
            print("T=",(time.time()-startTime))
            print("PATH="+pathVAD,flush=True)
            pathIn = pathVAD
    except Exception as e:
         print("Warning: can't filter noises")
         print(e)

    try:
        if(float(remixFactor) >= 1):
            pathREMIXN = pathClean
        elif (float(remixFactor) <= 0 and useDemucs):
            pathREMIXN = pathDemucsVocals;
        elif (isMusic and useDemucs):
            startTime = time.time()
            
            if(speechnorm):
                pathNORM = pathDemucsVocals+".NORM.wav"
                aCmd = ("ffmpeg -y -i \""+pathDemucsVocals+"\""
                        #+ " -filter:a loudnorm"
                        +" -af \"speechnorm=e=50:r=0.0005:l=1\""
                        +" \""+pathNORM+"\" > \""+pathNORM+".log\" 2>&1")
                print("CMD: "+aCmd)
                os.system(aCmd)
                print("T=",(time.time()-startTime))
                print("PATH="+pathNORM,flush=True)
            else:
                pathNORM = pathDemucsVocals

            pathREMIXN = pathNORM+".REMIX.wav"
            aCmd = ("ffmpeg -y -i \""+pathNORM+"\" -i \""+pathDemucsDrums+"\" -i \""+pathDemucsBass+"\" -i \""+pathDemucsOther+"\""
                    +" -filter_complex amix=inputs=4:duration=longest:dropout_transition=0:weights=\"1 "+remixFactor+" "+remixFactor+" "+remixFactor+"\""
                    +" \""+pathREMIXN+"\" > \""+pathREMIXN+".log\" 2>&1")
            print("CMD: "+aCmd)
            os.system(aCmd)
            print("T=",(time.time()-startTime))
            print("PATH="+pathREMIXN,flush=True)
    except Exception as e:
         print("Warning: can't remix")
         print(e)

    mode=1
    if(duration > 30):
        print("NOT USING MARKS FOR DURATION > 30s")
        mode=0
    
    startTime = time.time()
    if(onlySRT):
        result = {}
        result["text"] = ""
    else:
        result = transcribeMARK(pathIn, opts, mode=mode,lngInput=lngInput,isMusic=isMusic)
        if len(result["text"]) <= 0:
            result["text"] = "--"
    
    if(onlySRT or addSRT):
        #Better timestamps using original music clip
        if(isMusic
               #V3 is very bad with music!?
               and not whisperVersion == "-v3"
               ):
            if(pathREMIXN is not None):
                resultSRT = transcribeMARK(pathREMIXN, opts, mode=3,lngInput=lngInput,isMusic=isMusic
                                           ,nbRun=nbRun)
            else:
                resultSRT = transcribeMARK(pathClean, opts, mode=3,lngInput=lngInput,isMusic=isMusic
                                           ,nbRun=nbRun)
        else:
            resultSRT = transcribeMARK(pathNoCut, opts, mode=3,lngInput=lngInput,isMusic=isMusic
                                           ,nbRun=nbRun)
        
        result["text"] += resultSRT["text"]
    
    print("T=",(time.time()-initTime))
    if(len(result["text"]) > 0):
        print("s/c=",(time.time()-initTime)/len(result["text"]))
    print("c/s=",len(result["text"])/(time.time()-initTime))
    
    return result["text"]

def transcribeMARK(path: str,opts: dict,mode = 1,lngInput=None,aLast=None,isMusic=False
                   #Whisper is unstable, especially with music. Multiple run can provide with better results to eval afterward
                   ,nbRun=1):
    print("transcribeMARK(): "+path)
    pathIn = path
    
    lng = opts["language"]
    
    if(lngInput == None):
        lngInput = lng
        
    noMarkRE = "^(ar|he|ru|zh)$"
    if(lng != None and re.match(noMarkRE,lng) and mode != 3):
        #Need special voice marks
        mode = 0
    
    if(isMusic and mode != 3):
        #Markers are not really interesting with music
        mode = 0
    
    if(whisperFound == "SM4T"):
        #Not marker with SM4T
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
            if(mode != 3):
                startTime = time.time()
                pathMRK = pathIn+".MRK"+".wav"
                aCmd = "ffmpeg -y -i "+mark1+" -i \""+pathIn+"\" -i "+mark2+" -filter_complex \"[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]\" -map \"[a]\" -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathMRK+"\" > \""+pathMRK+".log\" 2>&1"
                print("CMD: "+aCmd)
                os.system(aCmd)
                print("T=",(time.time()-startTime))
                print("["+str(mode)+"] PATH="+pathMRK,flush=True)
                pathIn = pathMRK
            
            if(useCompressor
                and not isMusic
                ):
                startTime = time.time()
                pathCPS = pathIn+".CPS"+".wav"
                aCmd = "ffmpeg -y -i \""+pathIn+"\" -af \"speechnorm=e=50:r=0.0005:l=1\" "+ " -c:a pcm_s16le -ar "+str(SAMPLING_RATE)+" \""+pathCPS+"\" > \""+pathCPS+".log\" 2>&1"
                print("CMD: "+aCmd)
                os.system(aCmd)
                print("T=",(time.time()-startTime))
                print("["+str(mode)+"] PATH="+pathCPS,flush=True)
                pathIn = pathCPS
        except Exception as e:
             print("Warning: can't add markers")
             print(e)
    
    startTime = time.time()
    lock.acquire()
    try:
        transcribe_options = dict(**opts)#avoid to add beam_size opt several times
        if beam_size > 1:
            transcribe_options["beam_size"] = beam_size
        if patience > 0:
            transcribe_options["patience"] = patience
        if temperature > 0:
            transcribe_options["temperature"] = temperature
        
        if whisperFound == "FSTR":
            result = {}
            result["text"] = ""
            multiRes = ""
            for r in range(nbRun):
                print("RUN: "+str(r))
                segments, info = model.transcribe(pathIn,**transcribe_options)
                resSegs = []
                if(mode == 3):
                    aSegCount = 0
                    for segment in segments:
                        if("word_timestamps" in transcribe_options):
                            for word in segment.words:
                                aSegCount += 1
                                resSegs.append("\n"+str(aSegCount)+"\n"+formatTimeStamp(word.start)+" --> "+formatTimeStamp(word.end)+"\n"+word.word.strip()+"\n")
                        else:
                            aSegCount += 1
                            resSegs.append("\n"+str(aSegCount)+"\n"+formatTimeStamp(segment.start)+" --> "+formatTimeStamp(segment.end)+"\n"+segment.text.strip()+"\n")
                else:
                    for segment in segments:
                        resSegs.append(segment.text)
                
                result["text"] = "".join(resSegs)
                if(r > 0):
                    multiRes += "=====\n"
                multiRes += result["text"]
                
            if(nbRun > 1):
                result["text"] = multiRes 
        elif whisperFound == "SM4T":
            src_lang = lang2to3[lngInput];
            tgt_lang = lang2to3[lng];
            # S2TT
            #translated_text, _, _ = translator.predict(<path_to_input_audio>, "s2tt", <tgt_lang>)
            translated_text, _, _ = model.predict(pathIn, "s2tt", tgt_lang)
            result = {}
            result["text"] = str(translated_text)
        else:
            transcribe_options = dict(task="transcribe", **transcribe_options)
            multiRes = ""
            for r in range(nbRun):
                print("RUN: "+str(r))
                result = model.transcribe(pathIn,**transcribe_options)
                if(mode == 3):
                    p = Path(pathIn)
                    writer = WriteSRT(p.parent)
                    srtOpts = { "max_line_width" : 80, "max_line_count" : 2, "highlight_words" : False}
                    if("word_timestamps" in transcribe_options and transcribe_options["word_timestamps"]):
                        srtOpts = { "max_line_width" : 30, "max_line_count" : 1, "highlight_words" : transcribe_options["word_timestamps"]}
                    writer(result, pathIn,srtOpts)
                    audio_basename = os.path.basename(pathIn)
                    audio_basename = os.path.splitext(audio_basename)[0]
                    output_path = os.path.join(
                        p.parent, audio_basename + ".srt"
                        )
                    with open(output_path) as f:
                        result["text"] = f.read()
                    
                    if("word_timestamps" in transcribe_options and transcribe_options["word_timestamps"]):
                        result["text"] = re.sub("(\n[^<\n]*<u>|</u>[^<\n]*\n)"#Remove lines without highlighted words
                                                ,"\n",re.sub(r"\n[^<\n]*\n\n","\n\n"#Keep only highlighted words
                                                             ,result["text"]))
                if(r > 0):
                    multiRes += "=====\n"
                multiRes += result["text"]
            if(nbRun > 1):
                result["text"] = multiRes
        
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
    
    if(mode == 0 or mode == 3):
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

