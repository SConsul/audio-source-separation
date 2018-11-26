import librosa
import numpy as np
#import mathplotlib.pyplot as plt
import pickle
import torch
import os
import re

def reconstruct(phase, bass_mag, vocals_mag, drums_mag,others_mag,song_num,segment_num,destination_path):
	# Retrieve complex STFT
	vocals = np.squeeze(vocals_mag.detach().numpy() * phase,axis= (0,1))
	#print(vocals.shape)
	bass = np.squeeze(bass_mag.detach().numpy() * phase, axis=(0,1))
	drums = np.squeeze(drums_mag.detach().numpy() * phase, axis=(0,1))
	others = np.squeeze(others_mag.detach().numpy() * phase, axis=(0,1))

	# Perform ISTFT
	vocals_audio = librosa.istft(vocals, win_length=1024,hop_length=256,window='hann',center='True')
	bass_audio = librosa.istft(bass, win_length=1024,hop_length=256,window='hann',center='True')
	drums_audio = librosa.istft(drums, win_length=1024,hop_length=256,window='hann',center='True')
	others_audio = librosa.istft(others, win_length=1024,hop_length=256,window='hann',center='True')

	# Save as wav files
	librosa.output.write_wav(os.path.join(destination_path,'vocals',str(song_num)+'_'+str(segment_num)+'.wav'), vocals_audio,sr=44100)
	librosa.output.write_wav(os.path.join(destination_path,'bass',str(song_num)+'_'+str(segment_num)+'.wav'), bass_audio, sr=44100)
	librosa.output.write_wav(os.path.join(destination_path,'drums',str(song_num)+'_'+str(segment_num)+'.wav'), drums_audio, sr=44100)
	librosa.output.write_wav(os.path.join(destination_path,'others',str(song_num)+'_'+str(segment_num)+'.wav'), others_audio, sr=44100)
	return
