import librosa
import load_eeg
from matplotlib import pyplot as plt
import numpy as np
import read_xml as rx
from scipy.signal import butter, filtfilt, hilbert


def load_test_data():
    p = rx.load_participant(10, add_words=True)
    block = p.blocks[3]
    load_eeg.load_eeg_data_block(block)
    audio,_ = librosa.load('fn001161_60db.wav',sr = 16_000,offset = 37.51)
    return block, audio

def zero_mean_eeg_data(data):
    temp = data.transpose() - np.mean(data, axis=1)
    return temp.transpose()

def filter_eeg_data(data, low_frequency = 4, high_frequency = 10, order = 4, 
    sample_rate = 1000):
    data = zero_mean_eeg_data(data)
    b, a = butter(order, [low_frequency, high_frequency], btype='band', 
        fs=sample_rate)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal

def apply_hilbert_transform_on_channel(eeg_channel):
    return hilbert(eeg_channel)

def apply_hilbert_transform_on_all_channels(data, apply_filter = True, 
    low_frequency = 4, high_frequency = 10):
    if apply_filter: data = filter_eeg_data(data, low_frequency, 
        high_frequency)
    hilbert_transformed_data = np.zeros(data.shape, dtype=complex)
    for i in range(data.shape[0]):
        hilbert_transformed_data[i] = apply_hilbert_transform_on_channel(data[i])
    return hilbert_transformed_data


def compute_phase(hilbert_transformed_data):
    return np.angle(hilbert_transformed_data)

def plot_phase(data, apply_hilbert = True, start_index = 0, end_index = 2000):
    if apply_hilbert: 
        data = apply_hilbert_transform_on_all_channels(data)
        data = compute_phase(data)
    plt.clf()
    for channel in data:
        plt.plot(np.unwrap(channel[start_index:end_index]))
    yticks = np.array(list(range(0,10000,628)))/100
    plt.yticks(yticks)
    plt.grid()

    plt.show()


def compute_fft(signal, sample_rate = 44100):
    '''compute the fast fourier transform of a signal
    returns only the positive frequencies
    frequencies         a list of frequencies corresponding to the fft_result
    fft_result          a list of complex numbers -> fourier decomposition
                        of the signal
    '''

    fft_result= np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1.0/sample_rate)
    frequencies = frequencies[:int(len(frequencies)/2)]
    fft_result = fft_result[:int(len(fft_result)/2)]
    return frequencies, fft_result

def compute_power_spectrum(signal, sample_rate = 44100):
    '''compute the power spectrum of a signal
    frequencies         a list of frequencies corresponding to the fft_result
    power_spectrum      a list of real numbers -> power of the signal at each
                        frequency in frequencies
    '''
    frequencies, fft_result = compute_fft(signal, sample_rate)
    # the factor of 4 is to account for the fact that we only use the positive
    # frequencies
    power_spectrum = 10 * np.log10(4 * np.abs(fft_result)**2)
    return frequencies, power_spectrum

def plot_power_spectrum(signal, sample_rate = 44100):
    '''plot the power spectrum of a signal'''
    frequencies, power_spectrum = compute_power_spectrum(signal, sample_rate)
    plt.ion()
    plt.clf()
    plt.plot(frequencies, power_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(alpha=0.3)
    plt.show()
