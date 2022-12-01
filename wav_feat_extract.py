import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import librosa
import numpy as np

def wav_extract(wav_in):
    data = []
    y, sr = librosa.load(wav_in)
    audio_file, _ = librosa.effects.trim(y)

    #Default FFT window size
    n_fft = 2048
    hop_length = 512

    #Fourier transformation (can be used to transform into frequencies)
    #Doesnt look like we will use but will leave for the time being
    D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))

    #length (is the length of the audio_file variable) 
        #actual length (in seconds) is "len(audio_file) / sr"
    print("Length --> " + str(len(audio_file)))
    data.append(len(audio_file))

    #chroma_stft_mean -- chroma_stft_var
    chroma_stft = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)
    chroma_stft_mean = np.average(chroma_stft)
    print("Chroma STFT Mean--> " + str(chroma_stft_mean))
    data.append(chroma_stft_mean)
    chroma_stft_var = np.var(chroma_stft)
    print("Chroma STFT Variance --> " + str(chroma_stft_var))
    data.append(chroma_stft_var)

    #rms_mean -- rms_var
    rms = librosa.feature.rms(y=y)
    rms_mean = np.average(rms)
    print("Root Mean Squared Mean --> " + str(rms_mean))
    data.append(rms_mean)
    rms_var = np.var(rms)
    print("Root Mean Squared Variance --> " + str(rms_var))
    data.append(rms_var)

    #spectral_centroid_mean -- spectral_centroid_var
    spectral_centroid = librosa.feature.spectral_centroid(audio_file, sr=sr)[0]
    spectral_centroid_mean = np.average(spectral_centroid)
    print("Spectral Centroid Mean --> " + str(spectral_centroid_mean))
    data.append(spectral_centroid_mean)
    spectral_centroid_var = np.var(spectral_centroid)
    print("Spectral Centroid Variance --> " + str(spectral_centroid_var))
    data.append(spectral_centroid_var)

    #spectral_bandwidth_mean -- spectral_bandwidth_var
    spectral_bandwidth = librosa.feature.spectral_bandwidth(audio_file, sr=sr)
    spectral_bandwidth_mean = np.average(spectral_bandwidth)
    print("Spectral Bandwidth Mean --> " + str(spectral_bandwidth_mean))
    data.append(spectral_bandwidth_mean)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    print("Spectral Bandwidth Variance --> " + str(spectral_bandwidth_var))
    data.append(spectral_bandwidth_var)

    #rolloff_mean -- rolloff_var
    spectral_rolloff = librosa.feature.spectral_rolloff(audio_file, sr=sr)[0]
    rolloff_mean = np.average(spectral_rolloff)
    print("Rolloff Mean --> " + str(rolloff_mean))
    data.append(rolloff_mean)
    rolloff_var = np.var(spectral_rolloff)
    print("Rolloff Variance --> " + str(rolloff_var))
    data.append(rolloff_var)

    #zero_crossing_rate_mean -- zero_crossing_rate_var
    zero_crossing = librosa.feature.zero_crossing_rate(audio_file)
    z_cross_mean = np.average(zero_crossing)
    print("Zero Crossing Mean --> " + str(z_cross_mean))
    data.append(z_cross_mean)
    z_cross_var = np.var(zero_crossing)
    print("Zero Crossing Variance --> " + str(z_cross_var))
    data.append(z_cross_var)

    #Returns the arrays of harmony and perceptrual
    y_harm, y_perc = librosa.effects.hpss(audio_file)
    
    #harmony_mean -- harmony_var    Needs further testing
    harm_mean = np.average(y_harm)
    print("Harmonic Mean --> " + str(harm_mean))
    data.append(harm_mean)
    harm_var = np.var(y_harm)
    print("Harmonic Variance --> " + str(harm_var))
    data.append(harm_var)

    #perceptr_mean -- perceptr_var    Needs further testing
    perceptr_mean = np.average(y_perc)
    print("Perceptrual Mean --> " + str(perceptr_mean))
    data.append(perceptr_mean)
    perceptr_var = np.var(y_perc)
    print("Perceptrual Variance --> " + str(perceptr_var))
    data.append(perceptr_var)

    #tempo
    tempo, _ = librosa.beat.beat_track(y, sr = sr)
    print("Tempo --> " + str(tempo))
    data.append(tempo)

    #mfcc1-20_mean -- mfcc1-20_var
    mfccs = librosa.feature.mfcc(audio_file, sr=sr)
    index = 1
    for mfcc in mfccs:
        temp_mean = np.average(mfcc)
        print("MFCC_" + str(index) + " Mean --> " + str(temp_mean))
        data.append(temp_mean)
        temp_var = np.var(mfcc)
        print("MFCC_" + str(index) + " Variance --> " + str(temp_var))
        data.append(temp_var)
        index += 1

    return data
