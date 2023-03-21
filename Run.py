import librosa as lb
import pandas as pd
import pickle
import numpy as np
import winsound as ws
import librosa.display as lbd
from matplotlib import pyplot as plt

def return_emotion_predict(em):
    if em == 1:
        return "ANGRY"
    elif em == 2:
        return "DISGUST"
    elif em == 3:
        return "FEAR"
    elif em == 4:
        return "HAPPY"
    elif em == 5:
        return "SAD"
    else:
        return "NEUTRAL"

path = "AudioWAV/hahaah.wav"
# path = "AudioWAV/1002_DFA_SAD_XX.wav"
#path = "AudioWAV/1005_ITH_ANG_XX.wav"


ws.PlaySound(path, ws.SND_FILENAME)

audio,  sr = lb.load(path)
svm = pickle.load(open("svm","rb"))

lbd.waveshow(audio,sr = sr)
plt.show()

rms = lb.feature.rms(y = audio)[0]
chromo = lb.feature.chroma_stft(y = audio)[0]
mel = lb.feature.melspectrogram(y = audio)[0]
mfcc = lb.feature.mfcc(y = audio)[0]

x = {
    "mfcc": [np.mean(mfcc)],
    "mel": [np.mean(mel)],
    "chromo": [np.mean(chromo)],
    "rms": [np.mean(rms)]
}

X = pd.DataFrame(x)

k = svm.predict(X[["mfcc", "mel", "chromo", "rms"]])[0]
print(return_emotion_predict(k))