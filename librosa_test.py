import librosa

path = "data/vctk/VCTK-Corpus/wav48/p225/p225_001.wav"
#music, fs = librosa.load(path)
music, fs = librosa.audio.load(path)
print music, fs # [..], 22050

print librosa.samples_to_time(len(music), fs)
# [2.05156463]

tempo, beats = librosa.beat.beat_track(y=music, sr=fs)
print tempo, beats
# 60.09265988372093 [39]

D = librosa.stft(music)
print D.shape
print D
# (1025, 89)

import numpy as np
import matplotlib.pyplot as plt

librosa.display.specshow(D)
plt.savefig("stft.png")

x = np.abs(D)**2
data = librosa.logamplitude(x,ref_power=np.max)
librosa.display.specshow(data,y_axis='log', x_axis='time')
plt.savefig("stft_logamp.png")

S = np.log1p(np.abs(D)) #log(1+x)

plt.imshow(S)
plt.tight_layout()
plt.savefig("stft_log1p_no_aspect.png")

plt.imshow(S, aspect='auto')
plt.tight_layout()
plt.savefig("stft_log1p.png")

plt.imshow(S.T, aspect='auto')
plt.tight_layout()
plt.savefig("stft_log1p_t.png")

