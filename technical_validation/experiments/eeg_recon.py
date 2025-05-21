import numpy as np
import scipy
import os
import matplotlib.pyplot as plt

def make_filter(high_pass_freq=None, low_pass_freq=None):
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

def main():
    # Load the model and the test envelope
    model_dir = './technical_validation/experiments/results_linear_forward'
    model_filename = 'model_sub-001_-6_26_None_4.npy'
    model_path = os.path.join(model_dir, model_filename)

    env_dir = './derivatives/preprocessed_stimuli'
    env_filename = 'audiobook_7_1_envelope.npy'
    env_path = os.path.join(env_dir, env_filename)

    model = np.load(model_path)
    test_env = np.load(env_path)
    print(model.shape, test_env.shape)

    # Get the parameters from the filename
    subject = model_filename.split('_')[1]
    high_pass_freq = model_filename.split('_')[-2]
    low_pass_freq = model_filename.split('_')[-1].split('.')[0]
    high_pass_freq = None if high_pass_freq == 'None' else float(high_pass_freq)
    low_pass_freq = None if low_pass_freq == 'None' else float(low_pass_freq)
    tmin = -np.round(0.1*64).astype(int)
    tmax = np.round(0.4*64).astype(int)

    # Apply the filter to the test envelope
    filter_ = make_filter(high_pass_freq=high_pass_freq, low_pass_freq=low_pass_freq)
    if filter_ is not None:
        test_env = scipy.signal.sosfiltfilt(filter_, test_env, axis=0)

    # Create the time-lagged matrix
    test_env = time_lag_matrix(test_env, tmin, tmax)

    # Apply the model to the filtered envelope
    pred_eeg = np.matmul(test_env, model)
    print(pred_eeg.shape)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(pred_eeg[0:1000,:], label='Predicted EEG')
    plt.plot(test_env[0:1000, 0], label='Test Envelope')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
