from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = ['#472A7A', '#375A8C', '#26828E', '#22A884', '#63CB5F', '#CAE11F']

def mat2csv(mat_path):
    data = loadmat(mat_path)
    hpc_df = pd.DataFrame(data['hpcfull'])
    rsc_df = pd.DataFrame(data['rscfull'])
    hpc_df.to_csv('hpcfull.csv', index=False)
    rsc_df.to_csv('rscfull.csv', index=False)
    return hpc_df, rsc_df


def plot_activation(df, neuron_list=None, n_sample=6, title='Neuron activation'):
    data = df.to_numpy(dtype=np.float64)
    N, T_steps = data.shape
    if neuron_list is None:
        rng = np.random.default_rng()
        neuron_list = sorted(rng.choice(N, size=min(n_sample, N), replace=False).tolist())

    norm = data / (data.max(axis=1, keepdims=True) + 1e-12)
    t = np.arange(T_steps)

    plt.figure(figsize=(12, 4))
    for k, i in enumerate(neuron_list):
        plt.plot(t, norm[i], label=f'neuron {i}', color=colors[k % len(colors)])
    plt.xlabel('Time bin')
    plt.ylabel('Activation (normalized)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_snr(data, bg_frac=0.25):
    """
    Per-neuron SNR: peak amplitude above baseline, compared to the noise floor (std of the background).
    """
    values = data.to_numpy(dtype=np.float64) if hasattr(data, 'to_numpy') else np.asarray(data, dtype=np.float64)
    baseline = values.min(axis=1, keepdims=True)
    peak = values.max(axis=1, keepdims=True)
    signal = (peak - baseline).flatten()
    threshold = (baseline + bg_frac * (peak - baseline)).flatten()

    N = values.shape[0]
    snr_per_neuron = np.empty(N)
    for i in range(N):
        background = values[i][values[i] <= threshold[i]]
        noise_std = np.std(background) if background.size > 1 else np.nan
        snr_per_neuron[i] = 20 * np.log10(signal[i] / (noise_std + 1e-12))

    return {
        'mean_snr': float(np.nanmean(snr_per_neuron)),
        'snr_per_neuron': snr_per_neuron,
    }