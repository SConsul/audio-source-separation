import librosa
import numpy as np
import os
import re
path='../Recovered_Songs_bigger2/'
destination_path='../Processed_bigger2/'
vocals_path=os.path.join(path,'vocals')
bass_path=os.path.join(path,'bass')
drums_path=os.path.join(path,'drums')
others_path=os.path.join(path,'others')
for subdirs, dirs, files in os.walk(bass_path):
    mean=np.zeros((1109760,))
    num=0
    for song in files:
        temp,_ = librosa.load(os.path.join(vocals_path,song))
        #print(temp)
        mean+= temp
        num+=1
    mean/=num
    #print(mean)
    for song in files:
        temp,sr = librosa.load(os.path.join(vocals_path,song))
        #print(os.path.join(vocals_path,song))
        temp-=mean
        out_bass_path=os.path.join(destination_path,'bass')
        print(song)
        if not os.path.exists(out_bass_path):
            os.makedirs(out_bass_path)
        sound_output_path=os.path.join(out_bass_path,song)
        librosa.output.write_wav(sound_output_path,temp,sr)
