import numpy as np
import scipy
import os
from glob import glob
import json
import matplotlib.pyplot as plt
import pandas as pd

def load_results(result_dir, tmin:int, tmax:int):
    result_paths = glob(os.path.join(result_dir, f'*{tmin}_{tmax}*.json'))
    print(f'Found {len(result_paths)} results in {result_dir}')
    
    results = []
    for p in result_paths:
        with open(p, 'r') as f:
            result = json.load(f)
            results.append(result)
    print(f'Loaded {len(results)} results from {result_dir}')
    print(f'Keys: {results[0].keys()}')

    return results

def make_filter(high_pass_freq=None, low_pass_freq=None):
    # adapted from regression_linear_forward_model.py
    # Create the filter
    if high_pass_freq and low_pass_freq:
        filter_ = scipy.signal.butter(N= 1,
                                        Wn =[high_pass_freq, low_pass_freq],
                                        btype= "bandpass",
                                        fs=64,
                                        output="sos")
    if high_pass_freq and not low_pass_freq:
        filter_ = scipy.signal.butter(N= 1,
                                        Wn = high_pass_freq,
                                        btype= "highpass",
                                        fs=64,
                                        output="sos")
    if not high_pass_freq and low_pass_freq:
        filter_ = scipy.signal.butter(N= 1,
                                        Wn = low_pass_freq,
                                        btype= "lowpass",
                                        fs=64,
                                        output="sos")
    if not high_pass_freq and not low_pass_freq:
        filter_ = None
    
    return filter_

def time_lag_matrix(input_, tmin, tmax):
    """Create a time-lag matrix from a 2D numpy array.
    Adapted from regression_linear_forward_model.py

    Parameters
    ----------
    eeg: np.ndarray
        2D numpy array with shape (n_samples, n_channels)
    num_lags: int
        Number of time lags to use.

    Returns
    -------
    np.ndarray
        2D numpy array with shape (n_samples, n_channels* num_lags)
    """
    # Create a time-lag matrix
    numChannels = input_.shape[1]

    final_array = np.zeros((input_.shape[0], numChannels * (tmax - tmin)))

    for index, shift in enumerate(range(tmin, tmax)):
        # roll the array to the right
        shifted_data = np.roll(input_, -shift, axis=0)
        final_array[:, index * numChannels: (index + 1) * numChannels] = shifted_data

    if tmin < 0:
        return final_array[np.abs(tmin):-tmax+1, :]
    else:
        return final_array[:-tmax+1, :]

def plot_coef(model, tmin, tmax, fs=64, ylim=(None, None)):
    delays = np.arange(tmin, tmax)/fs
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(delays, model)
    ax.set_xlabel('Time delay (s)')
    ax.set_ylabel('Model coefficients')
    # ax.set_title('Model coefficients')
    ax.set_ylim(ylim)
    ax.axhline(0, color='k', linestyle='--')
    ax.axvline(0, color='k', linestyle='--')
    ax.grid()
    plt.tight_layout()

    return ax
    

def main():
    fs = 64
    tmin = -6
    tmax = 26
    freq_bands = {
        'Delta': (0.5, 4.0),
        'Theta': (4, 8.0),
        'Alpha': (8, 14.0),
        'Beta': (14, 30.0),
        'Broadband': (0.5, 32.0)
    }
    figure_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    env_dir = './derivatives/preprocessed_stimuli'
    result_dir = './technical_validation/experiments/results_linear_forward'

    results = load_results(result_dir, tmin, tmax)

    subjects = [result['subject'] for result in results]
    high_pass_freqs = np.array([result['highpass'] for result in results])
    low_pass_freqs = np.array([result['lowpass'] for result in results])
    heard_stims = [result['stim_filename'] for result in results]

    models = np.stack([result['model_weights'] for result in results], axis=0)
    print(models.shape)

    model_avg = np.mean(models, axis=0)
    ax = plot_coef(model_avg, -6, 26)
    plt.savefig(os.path.join(figure_dir, 'model_avg.jpg'), dpi=800)

    stim_paths = glob(os.path.join(env_dir, '*.npy'))
    stim_names = ["_".join(os.path.basename(stim_path).split('_')[:-1]) for stim_path in stim_paths]

    envs = [np.load(stim_path) for stim_path in stim_paths]
    envs_dict = dict(zip(stim_names, envs))
    unheard_stims = []
    for heard_stim in heard_stims:
        unheard_stim = list(set(stim_names) - set(heard_stim))
        unheard_stims.append(unheard_stim)
    
    save_dir = './technical_validation/experiments/predictions'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for idx in range(len(models)):
        test_env_name = np.random.choice(unheard_stims[idx], 1)[0]
        test_env = envs_dict[test_env_name]
        # Apply the filter to the test envelope
        filter_ = make_filter(high_pass_freq=high_pass_freqs[idx], low_pass_freq=low_pass_freqs[idx])
        if filter_ is not None:
            test_env = scipy.signal.sosfiltfilt(filter_, test_env, axis=0)
        # Create the time-lagged matrix
        test_env = time_lag_matrix(test_env, tmin, tmax)

        # Apply the model to the filtered envelope
        pred_eeg = np.matmul(test_env, models[idx])
        # print(pred_eeg.shape)
        save_filename = os.path.join(save_dir, f'{subjects[idx]}_{test_env_name}_{high_pass_freqs[idx]}_{low_pass_freqs[idx]}.npy')
        np.save(save_filename, pred_eeg)
        print(f'Saved {save_filename}')
    
    plt.figure(figsize=(10, 5))
    plt.plot(pred_eeg[0:1000,:], label='Predicted EEG')
    # plt.plot(test_env[0:1000, 0], label='Test Envelope')
    # plt.legend()
    plt.savefig(os.path.join(figure_dir, 'pred_eeg.jpg'), dpi=800)
    # plt.show()

if __name__ == "__main__":
    main()
