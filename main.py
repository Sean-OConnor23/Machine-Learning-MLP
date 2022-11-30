import numpy as np
import scipy.io.wavfile as wav
import random

GENRE_LIST = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

def get_data():
    X = []
    y = []
    for genre in GENRE_LIST:
        path = "data/genres_original/" + genre + "/" + genre + "."
        for i in range(0, 100):
            file = path + str(i).zfill(5) + ".wav"
            # print(str(file))
            samplerate, data = wav.read(str(file))
            X = np.append(X, (samplerate, data))
            y = np.append(y, genre)

    X = np.array(X)
    y = np.array(y)
    return X, y


def separate_data():
    # split 80/20
    pass


X, y = get_data()
print(X, y)