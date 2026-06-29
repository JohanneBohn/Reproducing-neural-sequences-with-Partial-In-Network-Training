import numpy as np
import matplotlib.pyplot as plt

colors = ['#472A7A', '#375A8C', '#26828E', '#22A884', '#63CB5F', '#CAE11F']

class Inputs:
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.T_steps = int(self.T / dt)
        self.t = np.arange(0, T, dt)
        
    def inputs(self, N, WN_tau, h0):
        """
        Returns inputs of shape (N, T_steps)
        """
        inputs = np.zeros((N, self.T_steps))
        h = np.zeros(N)
        for i_t in range(self.T_steps):
            eta = np.random.randn(N) # maybe utiliser un meilleur générateur
            dh = (-h + h0*eta) / WN_tau
            h += self.dt*dh
            inputs[:, i_t] = h.copy()
        return inputs

    def display_inputs(self, N, WN_tau, h0):
        inputs = self.inputs(N, WN_tau, h0)
        plt.figure(figsize=(12, 4))
        plt.plot(inputs, color="#F9B522D7", linewidth=1.5)
        plt.axhline(inputs.mean(), color="gray", linestyle='--', label='mean')
        plt.xlabel('t')
        plt.ylabel('inputs')
        plt.title('Inputs')
        plt.legend()
        plt.tight_layout()
        plt.show()