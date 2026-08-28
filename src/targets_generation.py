import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d

colors = ['#472A7A', '#375A8C', '#26828E', '#22A884', '#63CB5F', '#CAE11F']

class Gaussian_seq:
    """
    Generates gaussian sequences where each neuron's activation is maximal at a specific time-stamp (t_center).
    """
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.t = np.arange(0, T, dt)

    def gaussian(self, sigma, t_center):
        return np.exp(-0.5 * ((self.t - t_center)/sigma)**2)
    
    def target_functions(self, T, sigma, N, epsilon):
        t_centers = np.linspace(0, T, N)
        targets_rate = np.array([self.gaussian(sigma, t_center = tc) for tc in t_centers])
        targets_clip  = np.clip(targets_rate, epsilon, 1-epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        return targets_logit

    def gaussian_graph(self, T, sigma, N, neuron_list):
        t_centers = np.linspace(0, T, N)
        targets_rate = np.array([self.gaussian(sigma, t_center = tc) for tc in t_centers])
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors) :
            plt.plot(t, targets_rate[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Gaussian targets')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def target_graph(self, T, sigma, N, neuron_list, epsilon):
        targets_rate = self.target_functions(T, sigma, N, epsilon)
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, targets_rate[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Logit targets')
        plt.legend()
        plt.tight_layout()
        plt.show()


class Random_firing_gaussian_seq(Gaussian_seq):
    """
    Generates sequences where each neuron's maximal activation is drawn randomly on the time sequence.
    """
    def __init__(self, T, dt):
        super().__init__(T, dt)
        self.activations = None
        self.targets_rate = None

    def target_functions(self, T, sigma, N, epsilon):
        self.activations = [random.uniform(0, T) for _ in range(N)]
        targets_rate = np.array([self.gaussian(sigma, t_center = tc) for tc in self.activations])
        self.targets_rate = targets_rate
        targets_clip  = np.clip(targets_rate, epsilon, 1-epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        return targets_logit

    def gaussian_graph(self, T, sigma, N, neuron_list):
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, self.targets_rate[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Random firing gaussian targets')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def target_graph(self, T, sigma, N, neuron_list, epsilon):
        targets_clip = np.clip(self.targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, targets_logit[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Random firing gaussian logit targets')
        plt.legend()
        plt.tight_layout()
        plt.show()


class Multiple_gaussian_seq(Random_firing_gaussian_seq):
    """
    Generates sequences where each neuron has a random number of activation bumps (between 1 and max_bump).
    """
    def __init__(self, T, dt):
        super().__init__(T, dt)

    def _get_bump_list(self, N, max_bump):
        return [random.randint(1, max_bump) for _ in range(N)]

    def target_functions(self, T, sigma, N, epsilon, max_bump):
        n_bumps = self._get_bump_list(N, max_bump)
        self.activations = [[random.uniform(0, T) for _ in range(k)] for k in n_bumps]
        targets_rate = np.array([
            np.max([self.gaussian(sigma, t_center=tc) for tc in centers], axis=0)
            for centers in self.activations
        ])
        self.targets_rate = targets_rate
        targets_clip = np.clip(targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        return targets_logit


class Active_unactive_seq(Multiple_gaussian_seq):
    """
    Identical to Multiple_gaussian_seq, but allows for unactive neurons.
    Those neurons' target is a constant activity level.
    """
    def __init__(self, T, dt):
        super().__init__(T, dt)

    def _get_bump_list(self, N, max_bump):
            return [random.randint(0, max_bump) for _ in range(N)]

    def target_functions(self, T, sigma, N, epsilon, max_bump, silent_level=(0.0, 1.0)):
        """
        silent_level: (low, high) range from which each unactive neuron's constant activity level is drawn uniformly.
        """
        n_bumps = self._get_bump_list(N, max_bump)
        self.activations = [[random.uniform(0, T) for _ in range(k)] for k in n_bumps]
        targets_rate = np.array([
            np.max([self.gaussian(sigma, t_center=tc) for tc in centers], axis=0)
            if centers else np.full_like(self.t, random.uniform(*silent_level))
            for centers in self.activations
        ])
        self.targets_rate = targets_rate
        targets_clip = np.clip(targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        return targets_logit


class Indicator_seq(Gaussian_seq):
    """
    Generates targets shaped as indicator functions.
    """
    def __init__(self, T, dt):
        super().__init__(T, dt)
        self.activations = None
        self.targets_rate = None

    def _get_space_list(self, N):
        return [random.uniform(0, self.T) for _ in range(N)]

    def _get_width_list(self, N, width_range):
        return [random.uniform(*width_range) for _ in range(N)]

    def characteristic(self, t_center, width):
        return ((self.t >= t_center - width / 2) & (self.t <= t_center + width / 2)).astype(float)

    def target_functions(self, T, sigma, N, epsilon, width_range=None):
        if width_range is None:
            width_range = (0.3 * sigma, 3 * sigma)
        self.activations = self._get_space_list(N)
        self.widths = self._get_width_list(N, width_range)
        targets_rate = np.array([
            self.characteristic(tc, w) for tc, w in zip(self.activations, self.widths)
        ])
        self.targets_rate = targets_rate
        targets_clip = np.clip(targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        return targets_logit

    def target_graph(self, T, sigma, N, neuron_list, epsilon):
        targets_clip = np.clip(self.targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, targets_logit[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Characteristic (boxcar) logit targets')
        plt.legend()
        plt.tight_layout()
        plt.show()


class Grid_cells_seq(Gaussian_seq):
    """
    Generates sequences where every neuron fires three times, at the same fixed instants (tCOM = 3, 6 and 9 s),
    each occurrence with a random peak height (amp_range), and a width porportional to it.
    """
    def __init__(self, T, dt, tCOM_list):
        super().__init__(T, dt)
        self.activations = None
        self.targets_rate = None
        self.tCOM = tCOM_list

    def _get_amplitude_list(self, n, amp_range):
        return [random.uniform(*amp_range) for _ in range(n)]

    def target_functions(self, T, sigma, N, epsilon, amp_range=(0.2, 1.0)):
        n_bumps = len(self.tCOM)
        self.amplitudes = [self._get_amplitude_list(n_bumps, amp_range) for _ in range(N)]
        self.activations = [list(self.tCOM) for _ in range(N)]

        targets_rate = np.array([
            np.max([
                amp * self.gaussian(sigma * amp, t_center=tc)
                for tc, amp in zip(self.tCOM, amps)
            ], axis=0)
            for amps in self.amplitudes
        ])
        self.targets_rate = targets_rate
        targets_clip = np.clip(targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        return targets_logit

    def gaussian_graph(self, T, sigma, N, neuron_list):
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, self.targets_rate[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Grid cells targets')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def target_graph(self, T, sigma, N, neuron_list, epsilon):
        targets_clip = np.clip(self.targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, targets_logit[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Grid cells logit targets')
        plt.legend()
        plt.tight_layout()
        plt.show()


class Realistic_seq(Gaussian_seq):
    """
    Generates sequences with properties similar to the unavailable training data of Rajan, Harvey & Tank (2016):
    - each neuron fires transiently once during the trial, at its tCOM (time of center of mass);
    - bVar = 40% (~);
    - non-negative firing rates.
    Strategy: inject per-neuron variability (width jitter, amplitude jitter, colored rectified noise) to the idealized Gaussian sequences.
    """
    def __init__(self, T, dt, seed=42):
        super().__init__(T, dt)
        self.rng = np.random.default_rng(seed)
        self.t_centers = None
        self.targets_rate = None

    def _colored_noise(self, noise_tau):
        raw = self.rng.standard_normal(self.t.shape)
        sigma_samples = max(noise_tau / self.dt, 1e-9)
        smoothed = gaussian_filter1d(raw, sigma=sigma_samples)
        smoothed = smoothed / (smoothed.std() + 1e-9)
        return np.abs(smoothed)

    def realistic_gaussian(self, sigma, t_center, 
                           width_jitter, amp_jitter, 
                           noise_level, noise_tau, noise_floor, 
                           baseline_frac, baseline_range
                           ):
        width = sigma * np.exp(self.rng.normal(0, width_jitter))
        amp   = np.exp(self.rng.normal(0, amp_jitter))
        trace = amp * self.gaussian(width, t_center)
        if self.rng.uniform() < baseline_frac:
            trace = trace + self.rng.uniform(*baseline_range) * amp
        envelope = trace + noise_floor * amp
        trace = trace + noise_level * envelope * self._colored_noise(noise_tau)
        trace = np.clip(trace, 0, None)
        return trace

    def target_functions(self, T, sigma, N, epsilon, 
                         width_jitter=0.20, amp_jitter=0.10, 
                         noise_level=0.03, noise_tau=0.3, noise_floor=0.15, 
                         baseline_frac=0.3, baseline_range=(0.1, 0.4)
                         ):
        t_centers = np.sort(self.rng.uniform(0, T, N))
        targets_rate = np.array([
            self.realistic_gaussian(sigma, tc, width_jitter, amp_jitter, noise_level, noise_tau, noise_floor,
                                     baseline_frac, baseline_range)
            for tc in t_centers
        ])
        scale = np.percentile(targets_rate.max(axis=1), 99) + 1e-9
        targets_rate = np.clip(targets_rate / scale, 0, None)
        self.t_centers = t_centers
        self.targets_rate = targets_rate
        targets_clip = np.clip(targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        return targets_logit

    def gaussian_graph(self, T, sigma, N, neuron_list):
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, self.targets_rate[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Realistic targets')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def target_graph(self, T, sigma, N, neuron_list, epsilon):
        targets_clip = np.clip(self.targets_rate, epsilon, 1 - epsilon)
        targets_logit = np.log(targets_clip / (1 - targets_clip))
        t = self.t
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(t, targets_logit[i], label=f'Neuron {i}', color=color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Realistic logit targets')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def sequence_graph(self):
        order = np.argsort(self.t_centers)
        T = self.t[-1] + (self.t[1] - self.t[0])
        plt.figure(figsize=(10, 5))
        plt.imshow(self.targets_rate[order], aspect='auto', extent=[0, T, len(order), 0], cmap='viridis')
        plt.colorbar()
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron # (sorted by tCOM)')
        plt.title('Realistic sequence (sorted by tCOM)')
        plt.tight_layout()
        plt.show()