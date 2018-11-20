import librosa
import subprocess
import numpy as np
import os
import re
import glob
destination_path='../Recovered_Songs/'
vocals_directory='../AudioResults/vocals'
drums_directory='../AudioResults/drums'
bass_directory='../AudioResults/bass'
others_directory='../AudioResults/others'
test_songs_list=[]
test_segment_length=[]
vocals_list=[]
if not os.path.exists(destination_path):
    os.makedirs(destination_path)
for subdirs, dirs, files in os.walk(vocals_directory):
    print('finding list of songs ')
    for file in files :
        regex = re.compile(r'\d+')
        index = regex.findall(file)
        if not (index[0] in test_songs_list) :
            test_songs_list.append(index[0])

for test_songs in (test_songs_list):
    combined_vocals=np.array([])
    sr=None
    print('testing,..'+test_songs)
    print('Stitching Vocals')
    vocals_list = sorted(glob.glob(os.path.join(vocals_directory,test_songs+"*")))
    vocals_path=os.path.join(destination_path,'vocals')
    if not os.path.exists(vocals_path):
        os.makedirs(vocals_path)
    sound_output_path = os.path.join(vocals_path,test_songs)
    for segment in (vocals_list) :
        seg, sr = librosa.load(segment)
        combined_song= np.append(combined_vocals,seg)
    librosa.output.write_wav(sound_output_path,combined_vocals,sr)


    print('Stitching Bass')
    combined_bass=np.array([])
    sr=None
    bass_list = sorted(glob.glob(os.path.join(vocals_directory,test_songs+"*")))
    bass_path=os.path.join(destination_path,'bass')
    if not os.path.exists(bass_path):
        os.makedirs(bass_path)
    sound_output_path = os.path.join(bass_path,test_songs)
    for segment in (bass_list) :
        seg, sr = librosa.load(segment)
        combined_bass= np.append(combined_bass,seg)
    librosa.output.write_wav(sound_output_path,combined_bass,sr)


    print('Stitching Drums')
    combined_drums=np.array([])
    sr=None
    drums_list = sorted(glob.glob(os.path.join(vocals_directory,test_songs+"*")))
    drums_path=os.path.join(destination_path,'drums')
    if not os.path.exists(drums_path):
        os.makedirs(drums_path)
    sound_output_path = os.path.join(drums_path,test_songs)
    for segment in (drums_list) :
        seg, sr = librosa.load(segment)
        combined_drums= np.append(combined_drums,seg)
    librosa.output.write_wav(sound_output_path,combined_drums,sr)

    print('Stitching Others')
    combined_others=np.array([])
    sr=None
    others_list = sorted(glob.glob(os.path.join(vocals_directory,test_songs+"*")))
    others_path=os.path.join(destination_path,'others')
    if not os.path.exists(others_path):
        os.makedirs(others_path)
    sound_output_path = os.path.join(others_path,test_songs)
    for segment in (others_list) :
        seg, sr = librosa.load(segment)
        combined_others= np.append(combined_others,seg)
    librosa.output.write_wav(sound_output_path,combined_others,sr)
