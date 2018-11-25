import mir_eval  
import numpy as np
from scipy.io import wavfile
import librosa


####################### MODIFY ##############################
#### additional for loop to evaluate multiple songs #########
# increase step to decrease time
step  = 10
bass_gt_path = 'bass.wav'
bass_rec_path = 'bass_rec.wav'
vocal_gt_path = 'vocals.wav'
vocal_rec_path = 'vocals_rec.wav'
drums_gt_path = 'drums.wav'
drums_rec_path = 'drums_rec.wav'
other_gt_path = 'other.wav'
other_rec_path = 'other_rec.wav'
############################################################



bass_gt, rate11 = librosa.load(bass_gt_path,sr=44100, offset=30*0.3,duration = 170*0.3)
bass_rec, rate21 = librosa.load(bass_rec_path,sr=44100)

vocals_gt, rate12 = librosa.load(vocal_gt_path,sr=44100, offset=30*0.3,duration = 170*0.3)
vocals_rec, rate22 = librosa.load(vocal_rec_path,sr=44100)

drums_gt, rate13 = librosa.load(drums_gt_path,sr=44100, offset=30*0.3,duration = 170*0.3)
drums_rec, rate23 = librosa.load(drums_rec_path,sr=44100)

other_gt, rate14 = librosa.load(other_gt_path,sr=44100, offset=30*0.3,duration = 170*0.3)
other_rec, rate24 = librosa.load(other_rec_path,sr=44100)


bass_gt = bass_gt[0:bass_rec.shape[0]:step]
bass_gt = np.transpose(bass_gt.reshape(len(bass_gt), 1))

vocals_gt = vocals_gt[0:vocals_rec.shape[0]:step]
vocals_gt = np.transpose(vocals_gt.reshape(len(vocals_gt), 1))

drums_gt = drums_gt[0:drums_rec.shape[0]:step]
drums_gt = np.transpose(drums_gt.reshape(len(drums_gt), 1))

other_gt = other_gt[0:other_rec.shape[0]:step]
other_gt = np.transpose(other_gt.reshape(len(other_gt), 1))

final_gt = np.concatenate((bass_gt, vocals_gt, drums_gt, other_gt), axis = 0)
print(final_gt.shape)


bass_rec = bass_rec[0:bass_rec.shape[0]:step]
bass_rec = np.transpose(bass_rec.reshape(len(bass_rec), 1))

vocals_rec = vocals_rec[0:vocals_rec.shape[0]:step]
vocals_rec = np.transpose(vocals_rec.reshape(len(vocals_rec), 1))

drums_rec = drums_rec[0:drums_rec.shape[0]:step]
drums_rec = np.transpose(drums_rec.reshape(len(drums_rec), 1))

other_rec = other_rec[0:other_rec.shape[0]:step]
other_rec = np.transpose(other_rec.reshape(len(other_rec), 1))

final_rec = np.concatenate((bass_rec, vocals_rec, drums_rec, other_rec), axis = 0)
print(final_rec.shape)



SDR, SIR, SAR, perm = mir_eval.separation.bss_eval_sources(final_gt, final_rec)

print(SDR)
print(SIR)
print(SAR)
print(perm)