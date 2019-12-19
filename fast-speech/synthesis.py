import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import scipy.io.wavfile as wavfile
from nltk import tokenize 
from pydub import AudioSegment

from fastspeech import FastSpeech
from text import text_to_sequence
import hparams as hp
import utils
import Audio
import glow
import waveglow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_FastSpeech(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path, checkpoint_path))['model'])
    model.eval()
    return model

def synthesis(model, text, alpha=1.0):
    text = np.array(text_to_sequence(text, hp.text_cleaners))
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    with torch.no_grad():
        sequence = torch.autograd.Variable(
            torch.from_numpy(text)).cuda().long()
        src_pos = torch.autograd.Variable(
            torch.from_numpy(src_pos)).cuda().long()
        mel, mel_postnet = model.module.forward(sequence, src_pos, alpha=alpha)
        return mel[0].cpu().transpose(0, 1), \
            mel_postnet[0].cpu().transpose(0, 1), \
            mel.transpose(1, 2), \
            mel_postnet.transpose(1, 2)

import time 

if __name__ == "__main__":

    # get words
    words = "Google makes money by advertising. People or companies who want people to buy their product, service, or ideas give Google money, and Google shows an advertisement to people Google thinks will click on the advertisement. Google only gets money when people click on the link, so it tries to know as much about people as possible to only show the advertisement to the right people. It does this with Google Analytics, which sends data back to Google whenever someone visits a web site. From this and other data, Google makes a profile about the person, and then uses this profile to figure out which advertisements to show."
    sentences = tokenize.sent_tokenize(words)

    # set parameters
    num = 112000
    alpha = 1.0
    sigma = 1.0
    sr = 22050
    silence_duration = 600
    fr_multiplier = 1.05

    # get models
    fast_speech = get_FastSpeech(num)
    wave_glow = utils.get_WaveGlow()

    # iteratively compute TTS
    i = 0
    for sentence in sentences:
        _, _, _, mel_postnet_torch = synthesis(fast_speech, sentence, alpha=alpha)
        waveglow.inference.inference(mel_postnet_torch, wave_glow, 'results/{}.wav'.format(i), sigma=sigma)
        i += 1
    
    # compute final file
    silence = AudioSegment.silent(duration=silence_duration)
    audio_result = AudioSegment.from_wav('results/0.wav') + silence
    for i in range(1, len(sentences)):
        audio_result += AudioSegment.from_wav('results/{}.wav'.format(i)) +  silence
    audio_result = audio_result.set_frame_rate(int(audio_result.frame_rate * fr_multiplier))
    audio_result.export('results/result.wav', format='wav')


