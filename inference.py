import torch
import numpy as np
from scipy.io.wavfile import write

# load waveglow model
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

# load tacotron2 model
tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

# specify target text
text = "Born and raised in Pretoria, South Africa, Musk moved to Canada when he was 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received a Bachelor's degree in economics from the Wharton School and a Bachelor's degree in physics from the College of Arts and Sciences. He began a Ph.D. in applied physics and material sciences at Stanford University in 1995 but dropped out after two days to pursue an entrepreneurial career. He subsequently co-founded Zip2 with his brother Kimbal, a web software company, which was acquired by Compaq for $340 million in 1999. Musk then founded X.com, an online bank. It merged with Confinity in 2000, which had launched PayPal the previous year and was bought by eBay for $1.5 billion in October 2002."

# pre-process text
sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

# run the models
with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()

# write audio file
rate = 22050
write("audio.wav", rate, audio_numpy)
