import numpy as np
import matplotlib.pyplot as plt
from src.targets_generation import Gaussian_seq
from src.evaluation import Metrics

colors = ['#472A7A', '#375A8C', '#26828E', '#22A884', '#63CB5F', '#CAE11F']

class RNN:
    """
    Recurrent Neural Network without learning
    """
    def __init__(self, N, g, dt, tau, theta, p0):
        self.N = N
        self.g = g
        self.dt = dt
        self.tau = tau
        self.theta = theta
        self.p0 = p0
        self.J = np.random.normal(0, g/np.sqrt(N), (N,N)) # random connectivity matrix
        self.x = np.random.randn(N) * 0.1 # activation variable
        self.r = self.sigm(self.x) # firing rate
    
    def sigm(self, x):
        return 1 / (1 + np.exp(-(x - self.theta)))
    
    def step(self, h):
        self.z = self.J @ self.r
        self.x = self.x + self.dt * (-self.x + self.z + h) / self.tau
        self.r = self.sigm(self.x)
        return self.r.copy()

    def run(self, inputs):
        T_steps = inputs.shape[1]
        rates = np.zeros((self.N, T_steps))
        for t in range(T_steps):
            rates[:, t] = self.step(inputs[:, t])
        return rates
    
    
class PINning:
    """
    Partial In Network training (Rajan, Harvey & Tank, 2016),
    using the FORCE learning rule (Sussillo & Abbott, 2009).
    """
    def __init__(self, p, rnn, targets, p0):
        self.rnn = rnn
        self.targets = targets
        self.p = p
        self.pN = max(1, int(p*rnn.N)) # pN = number of neurons whose outgoing synapses are plastic
        self.plastic_neurons = np.random.choice(rnn.N, self.pN, replace=False) # random selection of plastic neurons
        self.P = p0 * np.eye(self.pN) # initialization of the P matrix

    def _get_plastic_rates(self):
        return self.rnn.r[self.plastic_neurons]
    
    def t_step(self, t):
        r_plastic = self._get_plastic_rates()
        Pr = self.P @ r_plastic
        c = 1 / (1 + r_plastic @ Pr) # effective learning rate
        self.P -= c*np.outer(Pr, Pr)
        self.P = 0.5 * (self.P + self.P.T) # to ensure symetry
        self.e = self.rnn.z - self.targets[:, t]
        delta_J = np.outer(self.e, c*Pr)
        self.rnn.J[:, self.plastic_neurons] -= delta_J

    def train(self, inputs, n_runs, cv_threshold, DEBUG):
        """
        Minimizes the mean squared error
        """
        T_steps = inputs.shape[1]
        errors = []
        for run in range(n_runs):
            self.rnn.x = np.random.randn(self.rnn.N) * 0.1
            self.rnn.r = self.rnn.sigm(self.rnn.x)
            run_error = 0
            for t in range(T_steps):
                self.rnn.step(inputs[:, t])
                self.t_step(t)
                run_error += np.mean(self.e**2)
            run_error /= T_steps
            errors.append(run_error)

            if DEBUG:
                print(f"Run {run+1} | run_error={run_error:.6f} | "
                    f"e mean={self.e.mean():.4f} | "
                    f"e std={self.e.std():.4f} | "
                    f"z mean={self.rnn.z.mean():.4f} | "
                    f"target mean={self.targets.mean():.4f}")

            if (run+1) % 50 == 0:
                print(f"Run {run+1}/{n_runs}, run_error = {run_error:.4f}") # displays progression
            if run_error < cv_threshold:
                print(f"Converged at run {run+1}, run_error = {run_error:.4f}")
                break
        return errors
    
    def display_cv(self, errors):
        plt.figure(figsize=(8, 4))
        plt.plot(errors, color="#F9B522D7", linewidth=1.5)
        plt.axhline(0.02, color="gray", linestyle='--', label='convergence threshold (0.02)')
        plt.xlabel('Run')
        plt.ylabel('run_error')
        plt.title('PINning convergence')
        plt.legend()
        plt.tight_layout()
        plt.show()
      
    def simulate(self, inputs, n_steps):
        all_rates = []
        for _ in range(n_steps):
            self.rnn.x = np.random.randn(self.rnn.N) * 0.1
            self.rnn.r = self.rnn.sigm(self.rnn.x)
            rates = self.rnn.run(inputs)
            all_rates.append(rates)
        return np.mean(all_rates, axis=0)
    
    def display_simulation(self, T, dt, rates, neuron_list):
        norm_rates = Metrics.normalize(rates)
        plt.figure(figsize=(12, 4))
        for i, color in zip(neuron_list, colors):
            plt.plot(Gaussian_seq(T, dt).t, norm_rates[i], label=f'neuron {i}', color = color)
        plt.xlabel('Time (s)')
        plt.ylabel('Activation')
        plt.title('Simulated normalized rate')
        plt.legend()
        plt.show()