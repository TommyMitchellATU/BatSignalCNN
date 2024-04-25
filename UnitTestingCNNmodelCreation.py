import unittest
import numpy as np
import os
import librosa
import scipy

class TestBatCallAnalyser(unittest.TestCase):
    def test_load_wav_file(self):
        y, sr = librosa.load('test.wav')
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(sr, int)

    def test_convert_to_spectrogram(self):
        y, sr = librosa.load('test.wav')
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        self.assertIsInstance(spectrogram, np.ndarray)

    def test_resize_spectrogram(self):
        y, sr = librosa.load('test.wav')
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        fixed_size = (640, 640)
        zoom_factor = (fixed_size[0] / spectrogram.shape[0], fixed_size[1] / spectrogram.shape[1])
        spectrogram = scipy.ndimage.zoom(spectrogram, zoom_factor)
        self.assertEqual(spectrogram.shape, fixed_size)

if __name__ == '__main__':
    unittest.main()