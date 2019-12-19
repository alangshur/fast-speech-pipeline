import scipy.io.wavfile as wavfile
from playsound import playsound
import scipy.ndimage as image
import numpy as np

# parameters
sr_mult = 1
sigma = 1

# input waveglow audio
sr, y = wavfile.read('waveglow.wav')

# gaussian blur audio
y = image.gaussian_filter1d(y, sigma)

# upsample and output processed audio
wavfile.write('test.wav', int(sr * sr_mult), y.astype(np.int16))
playsound('test.wav')