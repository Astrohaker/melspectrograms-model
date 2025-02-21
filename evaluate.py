import numpy as np
from scipy.io import wavfile
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchaudio


torch.cuda.empty_cache()
device = 'cuda'
n_mels = 64    
transform = torch.nn.Sequential(
   torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, hop_length=160, n_mels=n_mels),
   torchaudio.transforms.FrequencyMasking(freq_mask_param=int(n_mels*0.2)),
   torchaudio.transforms.TimeMasking(time_mask_param=int(0.2 * 16000/160)),
   torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)).to(device)


#model = M5(n_output=2)    
    
model = torch.jit.load("model_traced.pt")
model.to(device)
model.eval()

#optimized_traced_quantized_model = optimize_for_mobile(model)
#optimized_traced_quantized_model._save_for_lite_interpreter("streaming_asrv2.ptl")
#sys.exit(0)       
        
SAMPLE_RATE=16000
CHUNK=int(SAMPLE_RATE / 5)
#Load movix
#wav_data = wavfile.read("./dataset/test/HQ-SEI-Movix-test-window.wav")[1]
#wav_data=np.array(wav_data,dtype=np.float32)
wav_data, _ = torchaudio.load("e:\\tf20\\dataset\\test\\HQ-SEI-Movix-test-window.wav")
wav_data = wav_data.squeeze().numpy()

#waveform, sample_rate = torchaudio.load("./dataset/test/HQ-SEI-Movix-test-window.wav")
wav_data=np.array(wav_data,dtype=np.float32)

chunk_counter = int(len(wav_data) / CHUNK)#
offset = (chunk_counter+1) * CHUNK - len(wav_data) 
wav_movix = np.concatenate([wav_data, np.zeros(offset)])
#Load false

#waveform, sample_rate = torchaudio.load("./dataset/test/HQ-SEI-False-test-window.wav")
#wav_data_false=np.array(waveform[0].numpy(),dtype=np.float32)

#wav_data_false = wavfile.read("./dataset/test/HQ-SEI-False-test-window.wav")[1]

wav_data_false, _ = torchaudio.load("e:\\tf20\\dataset\\test\\HQ-SEI-Movix-test-window.wav")
wav_data_false = wav_data_false.squeeze().numpy()
wav_data_false=np.array(wav_data_false,dtype=np.float32)

chunk_counter = int(len(wav_data_false) / CHUNK)
offset = (chunk_counter+1) * CHUNK - len(wav_data_false) 
wav_false = np.concatenate([wav_data_false, np.zeros(offset)])
#Load noise
waveform, sample_rate = torchaudio.load("e:\\tf20\\dataset\\test\\HQ-SEI-Movix-test-window.wav")
wav_data_noise=np.array(waveform[0].numpy(),dtype=np.float32)

#wav_data_noise = wavfile.read("./dataset/test/SEI_office_Noise_Movix.wav")[1]
wav_data_noise = np.array(wav_data_noise,dtype=np.float32)
chunk_counter = int(len(wav_data_noise) / CHUNK)
offset = (chunk_counter+1) * CHUNK - len(wav_data_noise) 
wav_noise = np.concatenate([wav_data_noise, np.zeros(offset)])

wav_noise_false = np.concatenate([wav_data_noise, wav_data_false])

true_positives = wav_movix
false_positives = wav_noise_false

SAMPLE_RATE=16000
CHUNK=int(SAMPLE_RATE / 5)
positive_probabilities = []
    
t_holds = [0.6, 0.7, 0.8, 0.9]

for TRASHHOLD in t_holds: 
    positives_counter = 0
    buffer = np.zeros(SAMPLE_RATE).copy()
    wav = true_positives.copy()
    for i in range(0, int(len(wav) / CHUNK)):
        buffer_wav = np.concatenate([buffer[CHUNK:], wav[i * CHUNK:i * CHUNK + CHUNK]]).copy()
        buffer = buffer_wav.copy()
        input_data = np.float32(buffer_wav)
        input_data = torch.from_numpy(input_data).to(device)
        with torch.no_grad():
            predictions = model(transform(input_data.unsqueeze(0).unsqueeze(0)))
#        probabilities = predictions.softmax(dim=-1)
        probabilities = torch.exp(predictions)
        output = probabilities.cpu().detach().numpy()
        
        
        if (np.argmax(output) == 0 and output[0][0] > TRASHHOLD):
            positives_counter += 1
#            print(abs(predictions[0][0][0].item()))
            buffer = np.zeros(SAMPLE_RATE).copy()
    positive_probabilities.append(positives_counter)
print(positive_probabilities)

false_positive_probabilities = []
for TRASHHOLD in t_holds: 
    false_positives_counter=0
    buffer = np.zeros(SAMPLE_RATE).copy()
    wav = false_positives.copy()
    for i in range(0, int(len(wav) / CHUNK)):
        buffer_wav = np.concatenate([buffer[CHUNK:], wav[i * CHUNK:i * CHUNK + CHUNK]]).copy()
        buffer = buffer_wav.copy()
        input_data = np.float32(buffer_wav)
        input_data = torch.from_numpy(input_data).to(device)
        with torch.no_grad():
            predictions = model(transform(input_data.unsqueeze(0).unsqueeze(0)))
#        probabilities = predictions.softmax(dim=-1)
        probabilities = torch.exp(predictions)
        output = probabilities.cpu().detach().numpy()
                
        if (np.argmax(output) == 0 and output[0][0] > TRASHHOLD):
            false_positives_counter += 1
#            print(abs(predictions[0][0][0].item()))
            buffer = np.zeros(SAMPLE_RATE).copy()
    false_positive_probabilities.append(false_positives_counter)


print(false_positive_probabilities)


TOTAL_MOVIX = 385
precisions = []
recalls = []
probabilities = []
for i in range(0,len(positive_probabilities)):
    fn = TOTAL_MOVIX - positive_probabilities[i]
    tp = positive_probabilities[i]
    fp = false_positive_probabilities[i]
    if tp == 0 or fp == 0:
        f_m = 0
    else:        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_m = 2 * precision * recall / (precision + recall)  
    precisions.append(precision)
    recalls.append(recall)
    probabilities.append(f_m)
TRASHHOLD_index = probabilities.index (max(probabilities))
TRASHHOLD = t_holds[TRASHHOLD_index]

file = open("summary.txt", "w")
file.write(" Precision: " + str(precisions[TRASHHOLD_index]) + " Recall: " + str(recalls[TRASHHOLD_index]) + " F-measure: " + str(probabilities[TRASHHOLD_index])+ " Threshold: " + str(TRASHHOLD))
file.close()