from transcribeHallu import loadModel
from transcribeHallu import transcribePrompt

##### The audio language may be different from the one for the output transcription.
path = "data/KatyPerry-Firework.mp3"

##### Activate this for music file to get a minimal processing
isMusic = False

##### Need to be adapted for each language.
##### For prompt examples, see transcribeHallu.py getPrompt(lng:str)
lng = "en"

##### Model size to use
modelSize = "medium"
loadModel("0", modelSize=modelSize)

result = transcribePrompt(path=path, lng=lng)
print(result)
