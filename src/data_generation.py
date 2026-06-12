import numpy as np
import matplotlib.pyplot as plt

class Gaussian_seq:
    """
    Generates gaussian sequences.
    """
    def __init__(self, T, dt):
        self.t = np.arange(0, T, dt)

    def gaussian(self, sigma, t_center):
        return np.exp(-0.5 * ((self.t - t_center)/sigma)**2)
    
    def target_functions(
            self, 
            T, # sequence length
            sigma, # width of the gaussian curve
            N # number of neurons
        ):
        t_centers = np.linspace(0, T, N)
        return np.array([self.gaussian(sigma, t_center = tc) for tc in t_centers])

    def target_graph(self, T, sigma, N, neuron_list):
        targets = self.target_functions(T, sigma, N)
        t = self.t
        plt.figure(figsize=(12, 4))
        for i in neuron_list:
            plt.plot(t, targets[i], label=f'Neuron {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Gaussian targets')
        plt.legend()
        plt.tight_layout()
        plt.show()