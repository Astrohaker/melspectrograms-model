#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import librosa
import scipy.io.wavfile as sc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


interpreter = tf.lite.Interpreter('model.tflite')
interpreter.allocate_tensors()
input_model = interpreter.get_input_details()[0]
output_model = interpreter.get_output_details()[0]

MOVIX_DATAFILE = ".dataset/test/HQ-SEI-Movix-test-window.wav"
FALSE_DATAFILE = ".dataset/test/HQ-SEI-False-test-window.wav"
NOISE_DATAFILE = ".dataset/test/SEI_office_Noise_Movix.wav"
LABEL_DATAFILE = "labels.txt"
LIST_AUDIOFILES = [MOVIX_DATAFILE,FALSE_DATAFILE,NOISE_DATAFILE]
n_MFCC = 30
SAMPLE_RATE=16000
CHUNK=int(SAMPLE_RATE / 5)
TOTAL_MOVIX = 385
MIN_PROBABILITY = 0.5
MAX_PROBABILITY = 1.0
STEP_PROBABILITY = 0.05

def load_data():
    list_wavs=[]
    for audiofile in LIST_AUDIOFILES:
        wav_data = sc.read(audiofile)[1]
        wav_data = np.array(wav_data,dtype = np.float32)
        chunk_counter = int(len(wav_data) / CHUNK)
        offset = (chunk_counter+1) * CHUNK - len(wav_data)
        list_wavs.append(np.concatenate([wav_data, np.zeros(offset)]))
    wav_movix = list_wavs[0]
    wav_noise_false = np.concatenate([list_wavs[1], list_wavs[2]])
    #Load labels
    lines = []
    with open(LABEL_DATAFILE) as file:
        for line in file:
            line = line.strip()
            lines.append(line)
    file.close()
    index_movix = lines.index('movix')
    #Return
    return wav_movix, wav_noise_false, index_movix

true_positives, false_positives, index_movix = load_data()

def calc_probabilities():
    sm = cm.ScalarMappable(cmap='magma')

    positive_probabilities = []
    false_positive_probabilities = []
    list_probabilities = [positive_probabilities,false_positive_probabilities]
    list_datafiles = [true_positives, false_positives]
    for index_probability in range(0,len(list_probabilities)):
        for trashhold in np.arange(MIN_PROBABILITY, MAX_PROBABILITY, STEP_PROBABILITY):
            counter = 0
            buffer = np.zeros(SAMPLE_RATE).copy()
            wav = list_datafiles[index_probability].copy()
            for i in range(0, int(len(wav) / CHUNK)):
                fig, axs = plt.subplots(1, 1)

                buffer_wav = np.concatenate([buffer[CHUNK:], wav[i * CHUNK:i * CHUNK + CHUNK]]).copy()
                buffer = buffer_wav.copy()
                input_data = np.float32(buffer_wav)
                spectrogram = librosa.feature.melspectrogram(y=input_data, sr=16000, n_mels=256, hop_length=64,
                                                             n_fft=1024)
                spectrogram = librosa.power_to_db(spectrogram)
                # save spectrogram as PNG image
                sm.set_clim(spectrogram.min(), spectrogram.max())
                im = sm.to_rgba(spectrogram)
                im = axs.imshow(im)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.savefig("image.png", bbox_inches='tight', transparent="True", pad_inches=0)
                plt.close('all')

                img = Image.open("image.png").convert('RGB')
                img = np.array(img.resize((224, 224)))  # Resize to the input size of the model
                img = np.expand_dims(img, axis=0)  # Add batch dimension
                img = img.astype(np.float32)  # Convert to float32
                interpreter.set_tensor(input_model.get('index'), img)
                interpreter.invoke()
                output = interpreter.get_tensor(output_model.get('index'))
                if (np.argmax(output) == index_movix and output[0][index_movix] >= trashhold):
                    counter += 1
                    buffer = np.zeros(SAMPLE_RATE).copy()
            list_probabilities[index_probability].append(counter)
    return list_probabilities[0], list_probabilities[1]

positive_probabilities, false_positive_probabilities = calc_probabilities()

precisions = []
recalls = []
probabilities = []
for i in range(0,len(positive_probabilities)):
    fn = TOTAL_MOVIX - positive_probabilities[i]
    tp = positive_probabilities[i]
    fp = false_positive_probabilities[i]
    if tp == 0 or fp == 0:
        precision = 0
        recall = 0
        f_m = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_m = 2 * precision * recall / (precision + recall)
    precisions.append(precision)
    recalls.append(recall)
    probabilities.append(f_m)
trashhold_index = probabilities.index (max(probabilities))
trashhold = MIN_PROBABILITY + trashhold_index*STEP_PROBABILITY

file = open("summary.txt", "w")
file.write(" Precision: " + str(precisions[trashhold_index]) + " Recall: " + str(recalls[trashhold_index]) + " F-measure: " + str(probabilities[trashhold_index])+ " Trashhold: " + str(trashhold))
file.write("\n" + " Params librosa mfcc: " + " sr-" + str(SAMPLE_RATE) + " n_mfcc- " + str(n_MFCC))
file.close()

