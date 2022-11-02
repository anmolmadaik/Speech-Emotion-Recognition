import librosa as lb
import glob
import csv
import pandas as pd
from matplotlib import pyplot as plt
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


rmsmean = np.empty((7442, 1))
chromomean = np.empty((7442, 1))
melmean = np.empty((7442, 1))
mfccmean = np.empty((7442, 1))


for i in range(len(audio)):
    rmsmean[i] = np.mean(rms[i])
    chromomean[i] = np.mean(chromo[i])
    melmean[i] = np.mean(melmean[i])
    mfccmean[i] = np.mean(mfcc[i])

df["rms"] = rmsmean
df["chromo"] = chromomean
df["mel"] = melmean
df["mfcc"] = mfccmean


svc = SVC(gamma = 2)

svc.fit(df[["mfcc", "mel", "chromo", "rms"]], df["Emotion"])

pickle.dump(svc, open("svm", "wb"))