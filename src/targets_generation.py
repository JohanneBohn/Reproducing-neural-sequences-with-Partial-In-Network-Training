import numpy as np
import matplotlib.pyplot as plt

colors = ['#472A7A', '#375A8C', '#26828E', '#22A884', '#63CB5F', '#CAE11F']

class Gaussian_seq:
    """
    Generates gaussian sequences.
    """
    def __init__(self, T, dt):
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

# class Realistic_seq: for more realistic generated data