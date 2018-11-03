import librosa
import numpy as np
#import mathplotlib.pyplot as plt
import pickle
import torch
import os
import re

path= "../DSD100subset/"
path_mixtures = path + "Mixtures/Dev/"
path_sources = path + "Sources/Dev/"
destination_path = "../Processed/Mixtures"
phase_path= "../Processed/Phases"
bass_path="../Processed/Bass"
vocals_path="../Processed/Vocals"
drums_path="../Processed/Drums"
others_path="../Processed/Others"
source_dest_paths=[vocals_path,bass_path,drums_path,others_path]

path_val_mixtures = path + "Mixtures/Test/"
path_val_sources = path + "Sources/Test/"
validation_path = "../Val/Mixtures"
val_phase_path= "../Val/Phases"
val_bass_path="../Val/Bass"
val_vocals_path="../Val/Vocals"
val_drums_path="../Val/Drums"
val_others_path="../Val/Others"
source_val_paths=[val_vocals_path,val_bass_path,val_drums_path,val_others_path]


def reconstruct(file_path,direc,destination_path,phase_bool):
	# Retrieve complex STFT
	vocals = torch.Tensor.numpy(vocals_mag) * phase
	base = torch.Tensor.numpy(base_mag) * phase
	drums = torch.Tensor.numpy(drums_mag) * phase
	others = torch.Tensor.numpy(others_mag) * phase

	# Perform ISTFT
	vocals_audio = librosa.istft(vocals, n_fft=1024,hop_length=256,window='hann',center='True')
	base_audio = librosa.istft(base, n_fft=1024,hop_length=256,window='hann',center='True')
	drums_audio = librosa.istft(drums, n_fft=1024,hop_length=256,window='hann',center='True')
	others_audio = librosa.istft(others, n_fft=1024,hop_length=256,window='hann',center='True')

	# Save as wav files
	librosa.output.write_wav('', vocals_audio, 44100)
	librosa.output.write_wav('', base_audio, 44100)
	librosa.output.write_wav('', drums_audio, 44100)
	librosa.output.write_wav('', others_audio, 44100)
	return
