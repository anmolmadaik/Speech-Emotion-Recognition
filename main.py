import librosa as lb
import pandas as pd
from sklearn.svm import SVC
import pickle
import numpy as np


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


df = pd.read_csv("Audio.csv")
df = df.astype({"File": 'string', "Emotion":'int'})


audio = []
for i in range(len(df)):
    data, _ = lb.load(df.iloc[i,0])
    audio.append(data)


df["Audio"] = audio

rms = []
chromo = []
mel = []
mfcc = []

for i in range(len(audio)):
    r = lb.feature.rms(y = audio[i])[0]
    c = lb.feature.chroma_stft(y = audio[i])[0]
    me = lb.feature.melspectrogram(y = audio[i])[0]
    mf = lb.feature.mfcc(y= audio[i])[0]

    rms.append(r)
    chromo.append(c)
    mel.append(me)
    mfcc.append(mf)



rmslen = np.empty((7442, 1))
chromolen = np.empty((7442, 1))
mellen = np.empty((7442, 1))
mfcclen = np.empty((7442, 1))

for i in range(len(audio)):
    rmslen[i] = len(rms[i])
    chromolen[i] = len(chromolen[i])
    mellen[i] = len(mellen[i])
    mfcclen[i] = len(mfcclen[i])

mx = int(np.max(rmslen))

for i in range(len(rms)):
    rms[i] = np.pad(rms[i],(0,int(mx-rmslen[i])),'constant')
    mfcc[i] = np.pad(mfcc[i],(0,int(mx-mfcclen[i])),'constant')
    mel[i] = np.pad(mel[i],(0,int(mx-mellen[i])),'constant')
    chromo[i] = np.pad(chromo[i],(0,int(mx-chromolen[i])),'constant')

df["rms"] = rms
df["chromo"] = chromo
df["mel"] = mel
df["mfcc"] = mfcc



svc = SVC(gamma = 2)

svc.fit(df[["mfcc", "mel", "chromo", "rms"]], df["Emotion"])

pickle.dump(svc, open("svm", "wb"))