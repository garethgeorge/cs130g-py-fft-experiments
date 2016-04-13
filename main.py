import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
import math

# UTIL CLASS FOR READING WAVE FILES
class WaveReader:
    def __init__(self, w):
        self._w = w
        self._samples = self.readFloats()
        self._nchannels = self._w.getnchannels()
        self._channels = [self._samples[x::self._nchannels] for x in range(0, self._nchannels)]

    def readFloats(self):
        self._w.rewind()

        astr = self._w.readframes(self._w.getnframes())
        # convert binary chunks to short 
        a = struct.unpack("%ih" % (self._w.getnframes() * self._w.getnchannels()), astr)
        maxval = pow(2, 15)
        a = np.array([float(val) / maxval for val in a], dtype=float)

        return a

    def getSamples(self):
        return self._samples

    def getSamplesForChannel(self, channel):
        return self._channels[channel]

def takeSoundOffsetRange(list, position, count):
    return np.array(list[position:(position+count)], dtype=float)

def applyHanningWindow(list):
    N = len(list)

    def w(n):
        return 0.5 * (1 - math.cos((2 * math.pi * n) / (N - 1)))

    return np.array([w(n) * list[n] for n in range(0, N)], dtype=float)

def applyRectangularWindow(list):
    return list


def takeAndPlotFft(sound, offset, size, windowFunc, color='r'):
    samples = windowFunc(takeSoundOffsetRange(sound, offset, size))
    sp = np.fft.fft(samples)
    freq = np.fft.fftfreq(np.arange(size).shape[-1])
    plt.plot(freq, sp.real, freq, sp.imag, color=color)    

theSong = wave.open('radioactive.wav', 'rb')
waveReader = WaveReader(wave.open('radioactive.wav', 'rb'))
samples = waveReader.getSamplesForChannel(0)

takeAndPlotFft(samples, 44100 * 40, 1024, applyHanningWindow, color='r')
takeAndPlotFft(samples, 44100 * 40, 1024, applyRectangularWindow, color='g')
takeAndPlotFft(samples, 44100 * 40, 1024, applyTriangleWindow, color='b')

plt.show()


'''
np.fft.fft(x)

x = np.random.random(1024 * 8)
print(np.allclose(DFT_slow(x), np.fft.fft(x)))
'''
