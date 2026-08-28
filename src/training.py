import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.optimize import nnls
from src.targets_generation import Gaussian_seq
from src.evaluation import Metrics
from typing import Literal

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
        self.J_init = self.J.copy()
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
def _pin_train(J, P, x, plastic, theta, dt, tau, inputs, targets, n_runs, cv_threshold, x_init_scale, p_reg):
    N = J.shape[0]
    pN = plastic.shape[0]
    T_steps = inputs.shape[1]
    errors = np.empty(n_runs, dtype=np.float64)
    j_norms = np.empty(n_runs, dtype=np.float64)

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

        norm = 0.0
        for i in range(N):
            for j in range(N):
                norm += J[i,j] * J[i,j]
        j_norms[run] = norm ** 0.5

        if run_error < cv_threshold:
            n_done = run + 1
            break

    return errors[:n_done], j_norms[:n_done]


class PINning:
    """
    Partial In Network training (Rajan, Harvey & Tank, 2016), using the FORCE learning rule (Sussillo & Abbott, 2009).
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

    def train(self, inputs, n_runs, cv_threshold, DEBUG=False, p_reg=1e-9):
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        self.inputs = inputs # kept so save() can persist the exact frozen input used
        targets = np.ascontiguousarray(self.targets, dtype=np.float64)
        self.rnn.J = np.ascontiguousarray(self.rnn.J, dtype=np.float64)
        self.P = np.ascontiguousarray(self.P, dtype=np.float64)
        x = np.ascontiguousarray(self.rnn.x, dtype=np.float64)
        plastic = np.ascontiguousarray(self.plastic_neurons, dtype=np.int64)

        errors, j_norms = _pin_train(self.rnn.J, self.P, x, plastic, float(self.rnn.theta), float(self.rnn.dt), float(self.rnn.tau), inputs, targets, int(n_runs), float(cv_threshold), 0.1, float(p_reg))
        self.rnn.x = x
        self.rnn.r = self.rnn.sigm(self.rnn.x)
        errors = list(errors)
        j_norms  = list(j_norms)

        for k in range(0, len(errors)):
            delta_norm = j_norms[k] - (j_norms[k-1] if k > 0 else j_norms[0])
            if DEBUG:
                print(f"Run {k+1} | run_error={errors[k]:.6f} | ||J||={j_norms[k]:.2f} | Δ||J||={delta_norm:+.2f}")
            if (k+1) % 50 == 0:
                print(f"Run {k+1}/{n_runs}, run_error = {errors[k]:.4f} | ||J||={j_norms[k]:.2f} | Δ||J||={delta_norm:+.2f}") # displays progression
        if len(errors) < n_runs:
            print(f"Converged at run {len(errors)}, run_error = {errors[-1]:.4f} | ||J||={j_norms[-1]:.2f}")

        return errors, j_norms

    def display_weight_distribution(self, value: Literal['positive', 'negative', 'all'], bins=60):
        """
        Log-probability-density of the individual elements of J, before (J_init) and after (J) training.
        """
        if value == 'positive':
            J_init_full = self.rnn.J_init
            J_full = self.rnn.J
            J_init = J_init_full[J_init_full >= 0]
            J = J_full[J_full >= 0]
        elif value == 'negative':
            J_init_full = self.rnn.J_init
            J_full = self.rnn.J
            J_init = J_init_full[J_init_full <= 0]
            J = J_full[J_full <= 0]
        else:
            J_init = self.rnn.J_init
            J = self.rnn.J
        combined = np.concatenate([J_init.flatten(), J.flatten()])
        bin_range = (combined.min(), combined.max())
        def log_density(J):
            counts, edges = np.histogram(J.flatten(), bins=bins, range=bin_range, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            mask = counts > 0
            return centers[mask], np.log10(counts[mask])
        x_rand, y_rand = log_density(self.rnn.J_init)
        x_pinned, y_pinned = log_density(self.rnn.J)
        plt.figure(figsize=(6, 6))
        plt.plot(x_rand, y_rand, 's', color='#472A7A', markersize=4, label='$J_{Rand}$')
        plt.plot(x_pinned, y_pinned, 's', color='#26828E', markersize=4,
                 label=f'$J_{{PINned,\\ {self.p*100:.0f}\\%}}$')
        plt.grid(visible = True, which = 'both', axis = 'both')
        plt.xlabel('Synaptic strength')
        plt.ylabel('Log probability density, $10^x$')
        plt.title(f'Distribution of {value} weights')
        plt.legend()
        plt.tight_layout()
        plt.show()

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
        total = None
        for _ in range(n_steps):
            self.rnn.x = np.random.randn(self.rnn.N) * 0.1
            self.rnn.r = self.rnn.sigm(self.rnn.x)
            rates = self.rnn.run(inputs)
            total = rates.copy() if total is None else total + rates
        return total / n_steps

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

    _last_run_attrs = (
        'last_method', 'last_eta', 'last_eta_decay', 'last_lam', 'last_n_runs',
        'last_cv_threshold', 'last_stagnation_window', 'last_stagnation_tol', 'last_eta_inh_mult', 'last_K'
    )

    def save(self, filepath):
        """
        Saves the model so its weigths can be used later without re-training, and logs its parameters.
        """
        kwargs = dict(
            J=self.rnn.J,
            J_init=self.rnn.J_init,
            inputs=self.inputs,
            targets=self.targets,
            theta=self.rnn.theta,
            dt=self.rnn.dt,
            tau=self.rnn.tau,
            N=self.rnn.N,
            g=self.rnn.g,
            p0=self.rnn.p0,
            p=self.p,
            P=self.P,
            plastic_neurons=self.plastic_neurons,
            t=np.arange(self.inputs.shape[1]) * self.rnn.dt
        )
        for attr in self._last_run_attrs:
            if hasattr(self, attr):
                kwargs[attr] = getattr(self, attr)
        np.savez(filepath, **kwargs)

    @staticmethod
    def load(filepath):
        """
        Reloads the model previously saved.
        """
        data = np.load(filepath)
        rnn = RNN(
            N=int(data['N']), g=float(data['g']), dt=float(data['dt']),
            tau=float(data['tau']), theta=float(data['theta']), p0=float(data['p0']),
        )
        rnn.J = data['J']
        rnn.J_init = data['J_init']
        rnn.t = data['t']
        model = PINning(p=float(data['p']), rnn=rnn, targets=data['targets'], p0=float(data['p0']))
        model.plastic_neurons = data['plastic_neurons']
        model.P = data['P']
        for attr in PINning._last_run_attrs:
            if attr in data.files:
                value = data[attr]
                setattr(model, attr, value.item() if value.ndim == 0 else value)
        return model, data['inputs']


@njit(cache=True, fastmath=True)
def _pgd_train(J, x, plastic, sign, theta, dt, tau, inputs, targets, n_runs, cv_threshold, x_init_scale, eta, eta_decay, lam):
    """
    Projected gradient descent, projected at every timestep t onto the Dale's-law sign constraint.
    """
    N = J.shape[0]
    pN = plastic.shape[0]
    T_steps = inputs.shape[1]
    errors = np.empty(n_runs, dtype=np.float64)
    j_norms = np.empty(n_runs, dtype=np.float64)

    r = 1.0 / (1.0 + np.exp(-(x - theta)))
    z = np.empty(N, dtype=np.float64)
    rp = np.empty(pN, dtype=np.float64)
    n_done = n_runs

    for run in range(n_runs):
        eta_run = eta / (1.0 + eta_decay * run)

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

            step_err = 0.0
            for i in range(N):
                ei = z[i] - targets[i, t]
                step_err += ei * ei
                Ji = J[i]
                for b in range(pN):
                    jb = plastic[b]
                    grad = ei * rp[b] + lam * Ji[jb]
                    new_val = Ji[jb] - eta_run * grad
                    if sign[b] > 0:
                        if new_val < 0.0:
                            new_val = 0.0
                    else:
                        if new_val > 0.0:
                            new_val = 0.0
                    Ji[jb] = new_val
            run_error += step_err / N

            for i in range(N):
                x[i] = x[i] + dt * (-x[i] + z[i] + inputs[i, t]) / tau
                r[i] = 1.0 / (1.0 + np.exp(-(x[i] - theta)))

        run_error /= T_steps
        errors[run] = run_error

        norm = 0.0
        for i in range(N):
            for j in range(N):
                norm += J[i, j] * J[i, j]
        j_norms[run] = norm ** 0.5

        if run_error < cv_threshold:
            n_done = run + 1
            break

    return errors[:n_done], j_norms[:n_done]


@njit(cache=True, fastmath=True)
def _bptt_train(J, x, plastic, sign, theta, dt, tau, inputs, targets, n_runs, cv_threshold, x_init_scale, eta, eta_decay, lam, K, clip_norm):
    """
    Truncated backpropagation-through-time (BPTT), projected at every weight update onto the Dale's-law sign constraint.

    K = rolling window of K timesteps through which the gradients are back-propagated.
    K = 1 => reproduction of _pgd_train's gradient.
    """
    N = J.shape[0]
    pN = plastic.shape[0]
    T_steps = inputs.shape[1]
    errors = np.empty(n_runs, dtype=np.float64)
    j_norms = np.empty(n_runs, dtype=np.float64)

    r = 1.0 / (1.0 + np.exp(-(x - theta)))
    n_done = n_runs

    r_hist = np.empty((K, N), dtype=np.float64)
    e_hist = np.empty((K, N), dtype=np.float64)
    delta = np.empty(N, dtype=np.float64)
    eff = np.empty(N, dtype=np.float64)
    Jt_eff = np.empty(N, dtype=np.float64)
    z = np.empty(N, dtype=np.float64)
    grad = np.empty((N, pN), dtype=np.float64)

    for run in range(n_runs):
        eta_run = eta / (1.0 + eta_decay * run)
        run_max_grad_norm = 0.0

        for i in range(N):
            x[i] = np.random.standard_normal() * x_init_scale
            r[i] = 1.0 / (1.0 + np.exp(-(x[i] - theta)))

        run_error = 0.0
        chunk_start = 0
        while chunk_start < T_steps:
            chunk_end = chunk_start + K
            if chunk_end > T_steps:
                chunk_end = T_steps
            K_actual = chunk_end - chunk_start

            # forward pass through the chunk:
            for k in range(K_actual):
                t = chunk_start + k
                for i in range(N):
                    r_hist[k, i] = r[i]
                for i in range(N):
                    acc = 0.0
                    Ji = J[i]
                    for j in range(N):
                        acc += Ji[j] * r[j]
                    z[i] = acc
                step_err = 0.0
                for i in range(N):
                    ei = z[i] - targets[i, t]
                    e_hist[k, i] = ei
                    step_err += ei * ei
                run_error += step_err / N
                for i in range(N):
                    x[i] = x[i] + dt * (-x[i] + z[i] + inputs[i, t]) / tau
                    r[i] = 1.0 / (1.0 + np.exp(-(x[i] - theta)))

            # truncated backward pass within the chunk:
            for i in range(N):
                delta[i] = 0.0
            for b in range(pN):
                for i in range(N):
                    grad[i, b] = 0.0

            for k in range(K_actual - 1, -1, -1):
                for i in range(N):
                    eff[i] = e_hist[k, i] + (dt / tau) * delta[i]

                for b in range(pN):
                    rpb = r_hist[k, plastic[b]]
                    for i in range(N):
                        grad[i, b] += eff[i] * rpb

                for j in range(N):
                    acc = 0.0
                    for i in range(N):
                        acc += J[i, j] * eff[i]
                    Jt_eff[j] = acc

                for j in range(N):
                    rj = r_hist[k, j]
                    Dj = rj * (1.0 - rj)
                    delta[j] = (1.0 - dt / tau) * delta[j] + Dj * Jt_eff[j]

            # gradient clipping:
            gnorm_sq = 0.0
            for i in range(N):
                for b in range(pN):
                    gnorm_sq += grad[i, b] * grad[i, b]
            gnorm = gnorm_sq ** 0.5
            if gnorm > run_max_grad_norm:
                run_max_grad_norm = gnorm

            if clip_norm > 0.0 and gnorm > clip_norm:
                scale = clip_norm / (gnorm + 1e-8)
                for i in range(N):
                    for b in range(pN):
                        grad[i, b] *= scale

            # apply the projected accumulated gradient:
            for i in range(N):
                for b in range(pN):
                    jb = plastic[b]
                    g = grad[i, b] + lam * J[i, jb]
                    new_val = J[i, jb] - eta_run * g
                    if sign[b] > 0:
                        if new_val < 0.0:
                            new_val = 0.0
                    else:
                        if new_val > 0.0:
                            new_val = 0.0
                    J[i, jb] = new_val

            chunk_start = chunk_end

        run_error /= T_steps
        errors[run] = run_error

        norm = 0.0
        for i in range(N):
            for j in range(N):
                norm += J[i, j] * J[i, j]
        j_norms[run] = norm ** 0.5

        if run_error < cv_threshold:
            n_done = run + 1
            break

    return errors[:n_done], j_norms[:n_done]


@njit(cache=True, fastmath=True)
def _pgd_train_scaled(J, x, plastic, sign, eta_scale, theta, dt, tau, inputs, targets, n_runs, cv_threshold, x_init_scale, eta, eta_decay, lam):
    """
    Same as _pgd_train, but each plastic neuron has its own learning-rate multiplier
    -> during the training, inhibitory & excitatory plastic weights change at different magnitudes.
    """
    N = J.shape[0]
    pN = plastic.shape[0]
    T_steps = inputs.shape[1]
    errors = np.empty(n_runs, dtype=np.float64)
    j_norms = np.empty(n_runs, dtype=np.float64)

    r = 1.0 / (1.0 + np.exp(-(x - theta)))
    z = np.empty(N, dtype=np.float64)
    rp = np.empty(pN, dtype=np.float64)
    n_done = n_runs

    for run in range(n_runs):
        eta_run = eta / (1.0 + eta_decay * run)

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

            step_err = 0.0
            for i in range(N):
                ei = z[i] - targets[i, t]
                step_err += ei * ei
                Ji = J[i]
                for b in range(pN):
                    jb = plastic[b]
                    grad = ei * rp[b] + lam * Ji[jb]
                    new_val = Ji[jb] - eta_run * eta_scale[b] * grad
                    if sign[b] > 0:
                        if new_val < 0.0:
                            new_val = 0.0
                    else:
                        if new_val > 0.0:
                            new_val = 0.0
                    Ji[jb] = new_val
            run_error += step_err / N

            for i in range(N):
                x[i] = x[i] + dt * (-x[i] + z[i] + inputs[i, t]) / tau
                r[i] = 1.0 / (1.0 + np.exp(-(x[i] - theta)))

        run_error /= T_steps
        errors[run] = run_error

        norm = 0.0
        for i in range(N):
            for j in range(N):
                norm += J[i, j] * J[i, j]
        j_norms[run] = norm ** 0.5

        if run_error < cv_threshold:
            n_done = run + 1
            break

    return errors[:n_done], j_norms[:n_done]


def _fnnls_core(ZTZ, ZTx, tolerance=None):
    """
    Core active-set loop of Bro & de Jong's (1997) Fast NNLS, adapted from the fnnls.fnnls package to directly take already-computed:
    - Gram matrix ZTZ = ZᵀZ
    - moment vector ZTx = Zᵀx
    instead of recomputing them on every call.
    """
    n = ZTZ.shape[0]
    if tolerance is None:
        tolerance = 2.2204e-16 * n

    P = np.zeros(n, dtype=bool)
    d = np.zeros(n)
    w = ZTx - ZTZ @ d
    s = np.zeros(n)
    no_update = 0
    max_repetitions = 5

    while (not np.all(P)) and np.max(w[~P]) > tolerance:
        current_P = P.copy()
        P[np.argmax(w * ~P)] = True
        s[P] = np.linalg.solve(ZTZ[np.ix_(P, P)], ZTx[P])

        while np.any(P) and np.min(s[P]) <= tolerance:
            q = P & (s <= tolerance)
            alpha = np.min(d[q] / (d[q] - s[q]))
            d = d + alpha * (s - d)
            P[d <= tolerance] = False
            s[P] = np.linalg.solve(ZTZ[np.ix_(P, P)], ZTx[P])
            s[~P] = 0.0

        d = s.copy()
        w = ZTx - ZTZ @ d

        if np.all(current_P == P):
            no_update += 1
        else:
            no_update = 0
        if no_update >= max_repetitions:
            break

    return d


class Dale_PINning(PINning):
    """
    PINning on a network checking Dale's principle.
    """
    def set_dale_constraint(self, p_exc):
        """
        Sets a fraction p_exc of excitatory plastic neurons (all of their outgoing synapses are positive),
        the rest are set as inhibitory plastic neurons (all of their outgoing synapses are negative).
        """
        pN = self.pN
        n_exc = int(round(p_exc * pN))
        sign = -np.ones(pN, dtype=np.int64)
        exc_slots = np.random.choice(pN, n_exc, replace=False)
        sign[exc_slots] = 1
        self.sign = sign
        for b in range(pN):
            j = self.plastic_neurons[b]
            if sign[b] > 0:
                self.rnn.J[:, j] = np.abs(self.rnn.J[:, j])
            else:
                self.rnn.J[:, j] = -np.abs(self.rnn.J[:, j])
        self.rnn.J_init = self.rnn.J.copy()

    def set_dale_constraint_full(self, p_exc):
        """
        Like set_dale_constraint, but applies Dale's principle to all N neurons (not just the pN plastic ones).
        """
        N = self.rnn.N
        n_exc = int(round(p_exc * N))
        global_sign = -np.ones(N, dtype=np.int64)
        exc_neurons = np.random.choice(N, n_exc, replace=False)
        global_sign[exc_neurons] = 1
        self.global_sign = global_sign
        self.sign = global_sign[self.plastic_neurons]
        for j in range(N):
            if global_sign[j] > 0:
                self.rnn.J[:, j] = np.abs(self.rnn.J[:, j])
            else:
                self.rnn.J[:, j] = -np.abs(self.rnn.J[:, j])
        self.rnn.J_init = self.rnn.J.copy()

    def set_dale_constraint_mixed(self, p_exc, p_exc_plastic):
        """
        Applies Dale's principle to all N neurons according to the proportion p_exc,
        while ensuring that the E/I proportion p_exc_plastic is implemented in the plastic sub-sample.
        """
        N = self.rnn.N
        pN = self.pN
        plastic_mask = np.zeros(N, dtype=np.bool_)
        plastic_mask[self.plastic_neurons] = True
        rest_neurons = np.nonzero(~plastic_mask)[0]

        global_sign = np.empty(N, dtype=np.int64)

        n_exc_plastic = int(round(p_exc_plastic * pN))
        sign_plastic = -np.ones(pN, dtype=np.int64)
        exc_plastic = np.random.choice(pN, n_exc_plastic, replace=False)
        sign_plastic[exc_plastic] = 1
        global_sign[self.plastic_neurons] = sign_plastic
        self.sign = sign_plastic

        n_rest = rest_neurons.shape[0]
        n_exc_rest = int(round(p_exc * n_rest))
        sign_rest = -np.ones(n_rest, dtype=np.int64)
        exc_rest = np.random.choice(n_rest, n_exc_rest, replace=False)
        sign_rest[exc_rest] = 1
        global_sign[rest_neurons] = sign_rest

        self.global_sign = global_sign
        for j in range(N):
            if global_sign[j] > 0:
                self.rnn.J[:, j] = np.abs(self.rnn.J[:, j])
            else:
                self.rnn.J[:, j] = -np.abs(self.rnn.J[:, j])
        self.rnn.J_init = self.rnn.J.copy()

    def set_dale_constraint_balanced(self, p_exc, inh_scale=1.0):
        """
        Like set_dale_constraint, but scales the magnitude of the inhibitory plastic weights by inh_scale relative to the excitatory ones.
        Motivated by E/I balanced-network theory (Van Vreeswijk & Sompolinsky): inhibitory neurons are a numerical minority but individually stronger,
        so that total inhibitory drive can match total excitatory drive.
        """
        pN = self.pN
        n_exc = int(round(p_exc * pN))
        sign = -np.ones(pN, dtype=np.int64)
        exc_slots = np.random.choice(pN, n_exc, replace=False)
        sign[exc_slots] = 1
        self.sign = sign
        for b in range(pN):
            j = self.plastic_neurons[b]
            if sign[b] > 0:
                self.rnn.J[:, j] = np.abs(self.rnn.J[:, j])
            else:
                self.rnn.J[:, j] = -np.abs(self.rnn.J[:, j]) * inh_scale
        self.rnn.J_init = self.rnn.J.copy()

    def train_dale_pgd(self, inputs, n_runs, eta, cv_threshold, lam=0.0, eta_decay=0.0, DEBUG=False, x_init_scale=0.1):
        """
        On-line training via projected gradient descent onto the Dale's-law sign constraint.
        """
        if not hasattr(self, 'sign'):
            raise RuntimeError("call set_dale_constraint(p_exc) before train_dale_pgd()")
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        self.inputs = inputs
        targets = np.ascontiguousarray(self.targets, dtype=np.float64)
        self.rnn.J = np.ascontiguousarray(self.rnn.J, dtype=np.float64)
        x = np.ascontiguousarray(self.rnn.x, dtype=np.float64)
        plastic = np.ascontiguousarray(self.plastic_neurons, dtype=np.int64)
        sign = np.ascontiguousarray(self.sign, dtype=np.int64)

        errors, j_norms = _pgd_train(
            self.rnn.J, x, plastic, sign, float(self.rnn.theta), float(self.rnn.dt), float(self.rnn.tau),
            inputs, targets, int(n_runs), float(cv_threshold), float(x_init_scale), float(eta), float(eta_decay), float(lam)
        )
        self.rnn.x = x
        self.rnn.r = self.rnn.sigm(self.rnn.x)
        errors = list(errors)
        j_norms = list(j_norms)

        for k in range(len(errors)):
            if DEBUG:
                print(f"Run {k+1}/{n_runs} | run_error={errors[k]:.6f} | ||J||={j_norms[k]:.2f}")
            elif (k + 1) % 10 == 0:
                print(f"Run {k+1}/{n_runs} | run_error={errors[k]:.6f} | ||J||={j_norms[k]:.2f}")
        if len(errors) < n_runs:
            print(f"Converged at run {len(errors)}, run_error={errors[-1]:.6f} | ||J||={j_norms[-1]:.2f}")

        self.last_method = 'pgd'
        self.last_eta = eta
        self.last_eta_decay = eta_decay
        self.last_lam = lam
        self.last_cv_threshold = cv_threshold
        self.last_n_runs = len(errors)

        return errors, j_norms

    def train_dale_bptt(self, inputs, n_runs, eta, cv_threshold, K=200, lam=0.0, eta_decay=0.0, DEBUG=False, x_init_scale=0.1, clip_norm=None):
        """
        Truncated backpropagation-through-time (BPTT), projected at every weight update onto the Dale's-law sign constraint.

        K = the truncation window length in timesteps.
        K = 1 => reproduction of train_dale_pgd.
        """
        if not hasattr(self, 'sign'):
            raise RuntimeError("call set_dale_constraint(p_exc) before train_dale_bptt()")
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        self.inputs = inputs
        targets = np.ascontiguousarray(self.targets, dtype=np.float64)
        self.rnn.J = np.ascontiguousarray(self.rnn.J, dtype=np.float64)
        x = np.ascontiguousarray(self.rnn.x, dtype=np.float64)
        plastic = np.ascontiguousarray(self.plastic_neurons, dtype=np.int64)
        sign = np.ascontiguousarray(self.sign, dtype=np.int64)

        clip_val = float(clip_norm) if clip_norm is not None else -1.0

        errors, j_norms = _bptt_train(
            self.rnn.J, x, plastic, sign, float(self.rnn.theta), float(self.rnn.dt), float(self.rnn.tau),
            inputs, targets, int(n_runs), float(cv_threshold), float(x_init_scale), float(eta), float(eta_decay), float(lam), int(K), clip_val
        )
        self.rnn.x = x
        self.rnn.r = self.rnn.sigm(self.rnn.x)
        errors = list(errors)
        j_norms = list(j_norms)

        for k in range(len(errors)):
            if DEBUG:
                print(f"Run {k+1}/{n_runs} | run_error={errors[k]:.6f} | ||J||={j_norms[k]:.2f}")
            elif (k + 1) % 10 == 0:
                print(f"Run {k+1}/{n_runs} | run_error={errors[k]:.6f} | ||J||={j_norms[k]:.2f}")
        if len(errors) < n_runs:
            print(f"Converged at run {len(errors)}, run_error={errors[-1]:.6f} | ||J||={j_norms[-1]:.2f}")

        self.last_method = 'bptt'
        self.last_eta = eta
        self.last_eta_decay = eta_decay
        self.last_lam = lam
        self.last_cv_threshold = cv_threshold
        self.last_n_runs = len(errors)
        self.last_K = K
        self.last_clip_norm = clip_norm

        return errors, j_norms

    def train_dale_pgd_scaled(self, inputs, n_runs, eta, cv_threshold, lam=0.0, eta_decay=0.0, eta_inh_mult=1.0, DEBUG=False, x_init_scale=0.1):
        """
        Like train_dale_pgd, but applies a separate learning-rate multiplier eta_inh_mult to the inhibitory plastic synapses.
        => Lets training itself grow the inhibitory weights faster/slower relative to the excitatory ones.
        
        eta_inh_mult = 1.0 => reproduces train_dale_pgd exactly.
        """
        if not hasattr(self, 'sign'):
            raise RuntimeError("call set_dale_constraint(p_exc) before train_dale_pgd_scaled()")
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        self.inputs = inputs
        targets = np.ascontiguousarray(self.targets, dtype=np.float64)
        self.rnn.J = np.ascontiguousarray(self.rnn.J, dtype=np.float64)
        x = np.ascontiguousarray(self.rnn.x, dtype=np.float64)
        plastic = np.ascontiguousarray(self.plastic_neurons, dtype=np.int64)
        sign = np.ascontiguousarray(self.sign, dtype=np.int64)
        eta_scale = np.where(sign < 0, float(eta_inh_mult), 1.0).astype(np.float64)

        errors, j_norms = _pgd_train_scaled(
            self.rnn.J, x, plastic, sign, eta_scale, float(self.rnn.theta), float(self.rnn.dt), float(self.rnn.tau),
            inputs, targets, int(n_runs), float(cv_threshold), float(x_init_scale), float(eta), float(eta_decay), float(lam)
        )
        self.rnn.x = x
        self.rnn.r = self.rnn.sigm(self.rnn.x)
        errors = list(errors)
        j_norms = list(j_norms)

        for k in range(len(errors)):
            if DEBUG:
                print(f"Run {k+1}/{n_runs} | run_error={errors[k]:.6f} | ||J||={j_norms[k]:.2f}")
            elif (k + 1) % 10 == 0:
                print(f"Run {k+1}/{n_runs} | run_error={errors[k]:.6f} | ||J||={j_norms[k]:.2f}")
        if len(errors) < n_runs:
            print(f"Converged at run {len(errors)}, run_error={errors[-1]:.6f} | ||J||={j_norms[-1]:.2f}")

        self.last_method = 'pgd_scaled'
        self.last_eta = eta
        self.last_eta_decay = eta_decay
        self.last_lam = lam
        self.last_cv_threshold = cv_threshold
        self.last_n_runs = len(errors)
        self.last_eta_inh_mult = eta_inh_mult

        return errors, j_norms

    def train_dale_fnnls(self, inputs, n_runs, lam, DEBUG=False, stagnation_window=5, stagnation_tol=1e-4):
        """
        Batch Dale-constrained training via Fast NNLS (Bro & de Jong, 1997).
        """
        if not hasattr(self, 'sign'):
            raise RuntimeError("call set_dale_constraint(p_exc) before train_fnnls_dale()")
        N = self.rnn.N
        pN = self.pN
        plastic = self.plastic_neurons
        sign = self.sign.astype(np.float64)
        T_steps = inputs.shape[1]
        inputs = np.ascontiguousarray(inputs, dtype=np.float64)
        self.inputs = inputs
        errors = []

        for run in range(n_runs):
            self.rnn.x = np.random.randn(N) * 0.1
            self.rnn.r = self.rnn.sigm(self.rnn.x)
            z_traj = np.empty((N, T_steps))
            rp_traj = np.empty((pN, T_steps))
            for t in range(T_steps):
                r_pre = self.rnn.r[plastic]
                self.rnn.step(inputs[:, t])
                z_traj[:, t] = self.rnn.z
                rp_traj[:, t] = r_pre

            run_error = float(np.mean((z_traj - self.targets) ** 2))
            errors.append(run_error)

            A_signed = (rp_traj * sign[:, None]).T  # (T_steps, pN)
            H = A_signed.T @ A_signed
            H[np.diag_indices(pN)] += lam
            F = A_signed.T @ self.targets.T  # (pN, N), column i is A_signedᵀ targets[i, :]

            for i in range(N):
                w_hat = _fnnls_core(H, F[:, i])
                self.rnn.J[i, plastic] = sign * w_hat

            if DEBUG:
                print(f"Run {run+1}/{n_runs} | run_error={run_error:.6f} | ||J||={np.linalg.norm(self.rnn.J):.2f}")

            if stagnation_tol and len(errors) >= stagnation_window:
                recent = errors[-stagnation_window:]
                spread = (max(recent) - min(recent)) / (abs(np.mean(recent)) + 1e-12)
                if spread < stagnation_tol:
                    print(f"Stagnated at run {run+1}/{n_runs}, run_error={run_error:.6f} "
                          f"(spread over last {stagnation_window} runs: {spread:.2e} < {stagnation_tol})")
                    break

        self.last_method = 'fnnls'
        self.last_lam = lam
        self.last_n_runs = len(errors)
        self.last_stagnation_window = stagnation_window
        self.last_stagnation_tol = stagnation_tol

        return errors
