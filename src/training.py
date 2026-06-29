import numpy as np
import matplotlib.pyplot as plt
from numba import njit
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
        return 1.0 / (1.0 + np.exp(-(x - self.theta)))
    
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

@njit(cache=True, fastmath=True)
def _pin_train(J, P, x, plastic, theta, dt, tau, inputs, targets,
                n_runs, cv_threshold, x_init_scale, p_reg=0.0):
    N = J.shape[0]
    pN = plastic.shape[0]
    T_steps = inputs.shape[1]
    errors = np.empty(n_runs, dtype=np.float64)

    r = 1.0 / (1.0 + np.exp(-(x - theta)))
    z = np.empty(N, dtype=np.float64)
    rp = np.empty(pN, dtype=np.float64)
    Pr = np.empty(pN, dtype=np.float64)
    e = np.empty(N, dtype=np.float64)
    n_done = n_runs

    for run in range(n_runs):
        for i in range(N):
            x[i] = np.random.standard_normal() * x_init_scale
            r[i] = 1.0 / (1.0 + np.exp(-(x[i] - theta)))

        run_error = 0.0
        for t in range(T_steps):
            for i in range(N):
                acc = 0.0
                Ji = J[i]
                for j in range(N):
                    acc += Ji[j] * r[j]
                z[i] = acc

            for a in range(pN):
                rp[a] = r[plastic[a]]

            for a in range(pN):
                acc = 0.0
                Pa = P[a]
                for b in range(pN):
                    acc += Pa[b] * rp[b]
                Pr[a] = acc

            denom = 1.0
            for a in range(pN):
                denom += rp[a] * Pr[a]
            c = 1.0 / denom

            for a in range(pN):
                cPa = c * Pr[a]
                Pa = P[a]
                for b in range(pN):
                    Pa[b] -= cPa * Pr[b]

            for a in range(pN): # ensures P's symetry
                for b in range(a + 1, pN):
                    avg = 0.5 * (P[a, b] + P[b, a])
                    P[a, b] = avg
                    P[b, a] = avg

            if p_reg > 0.0:
                for a in range(pN):
                    P[a, a] += p_reg

            step_err = 0.0
            for i in range(N):
                ei = z[i] - targets[i, t]
                e[i] = ei
                step_err += ei * ei
            run_error += step_err / N

            for i in range(N):
                cei = c * e[i]
                Ji = J[i]
                for b in range(pN):
                    Ji[plastic[b]] -= cei * Pr[b]

            for i in range(N):
                x[i] = x[i] + dt * (-x[i] + z[i] + inputs[i, t]) / tau
                r[i] = 1.0 / (1.0 + np.exp(-(x[i] - theta)))

        run_error /= T_steps
        errors[run] = run_error
        if run_error < cv_threshold:
            n_done = run + 1
            break

    return errors[:n_done]
    
    
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
        # self.e = np.zeros(rnn.N)

    def _get_plastic_rates(self):
        return self.rnn.r[self.plastic_neurons]
    
    def t_step(self, pre_rate, target_t):
        """
        One FORCE update
        """
        r_plastic = pre_rate[self.plastic_neurons]
        Pr = self.P @ r_plastic
        c = 1.0 / (1.0 + r_plastic @ Pr) # effective learning rate
        self.P -= c*np.outer(Pr, Pr) # RLS
        self.P = 0.5 * (self.P + self.P.T) # to ensure symetry
        self.e = self.rnn.z - target_t
        self.rnn.J[:, self.plastic_neurons] -= np.outer(self.e, c*Pr)

    # def train(self, inputs, n_runs, cv_threshold, DEBUG=False):
    #     """
    #     Minimizes the mean squared error
    #     """
    #     T_steps = inputs.shape[1]
    #     errors = []
    #     for run in range(n_runs):
    #         self.rnn.x = np.random.randn(self.rnn.N) * 0.1
    #         self.rnn.r = self.rnn.sigm(self.rnn.x)
    #         run_error = 0.0
    #         for t in range(T_steps):
    #             pre_rate = self.rnn.r.copy()
    #             self.rnn.step(inputs[:, t])
    #             self.t_step(pre_rate, self.targets[:, t])
    #             run_error += np.mean(self.e**2)
    #         run_error /= T_steps
    #         errors.append(run_error)

    #         if DEBUG:
    #             print(f"Run {run+1} | run_error={run_error:.6f} | "
    #                 f"e mean={self.e.mean():.4f} | "
    #                 f"e std={self.e.std():.4f} | "
    #                 f"z mean={self.rnn.z.mean():.4f} | "
    #                 f"target mean={self.targets.mean():.4f}")

    #         if (run+1) % 50 == 0:
    #             print(f"Run {run+1}/{n_runs}, run_error = {run_error:.4f}") # displays progression
    #         if run_error < cv_threshold:
    #             print(f"Converged at run {run+1}, run_error = {run_error:.4f}")
    #             break
    #     return errors

    def train(self, inputs, n_runs, cv_threshold, DEBUG=False, p_reg=1e-9):
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        targets = np.ascontiguousarray(self.targets, dtype=np.float64)
        self.rnn.J = np.ascontiguousarray(self.rnn.J, dtype=np.float64)
        self.P = np.ascontiguousarray(self.P, dtype=np.float64)
        x = np.ascontiguousarray(self.rnn.x, dtype=np.float64)
        plastic = np.ascontiguousarray(self.plastic_neurons, dtype=np.int64)
        
        errors = _pin_train(self.rnn.J, self.P, x, plastic, float(self.rnn.theta), float(self.rnn.dt), float(self.rnn.tau), inputs, targets, int(n_runs), float(cv_threshold), 0.1, float(p_reg))
        self.rnn.x = x
        self.rnn.r = self.rnn.sigm(self.rnn.x)
        errors = list(errors)
        
        for k in range(0, len(errors)):
            if DEBUG:
                print(f"Run {k+1} | run_error={errors[k]:.6f}")
            if (k+1) % 50 == 0:
                print(f"Run {k+1}/{n_runs}, run_error = {errors[k]:.4f}") # displays progression
        if len(errors) < n_runs:
            print(f"Converged at run {len(errors)}, run_error = {errors[-1]:.4f}")
        
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
        # add the norm fo the weight update -> should decay towards 0
      
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