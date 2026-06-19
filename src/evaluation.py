import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Metrics:
    """
    Evaluation metrics from Rajan, Harvey & Tank (2016)
    """
    @staticmethod
    def normalize(rates):
        max_per_neuron = np.max(rates, axis=1, keepdims=True)
        return rates / (max_per_neuron + 1e-10)
    
    @staticmethod
    def compute_tCOM(rates, t):
        """
        tCOM = time of center of mass -> computed for each neuron
        """
        total = np.sum(rates, axis=1) + 1e-10
        return np.sum(rates * t[None, :], axis=1) / total
    
    @staticmethod
    def compute_bVar(rates, t, sigma):
        """
        bVar    = stereotypy of the neural sequence
                = the amount of variability in the data explained by the bump
        """
        norm_rates = Metrics.normalize(rates)
        N = norm_rates.shape[0]
        tCOM = Metrics.compute_tCOM(norm_rates, t)
        R_bar = norm_rates.mean(axis=0)
        num = 0.0
        denom = 0.0
        for i in range(N):
            R_ave_i = np.exp(-0.5 * ((t - tCOM[i])  / sigma)**2)
            scale = np.dot(norm_rates[i], R_ave_i) / (np.dot(R_ave_i, R_ave_i) + 1e-10)
            R_fitted = scale * R_ave_i
            num += np.mean((norm_rates[i] - R_fitted)**2)
            denom += np.mean((norm_rates[i] - R_bar)**2)
        bVar = 1 - num / (denom + 1e-10)
        return float(bVar)

    @staticmethod
    def compute_pVar(targets, rates):
        """
        pVar = amount of variance of the data that is captured by the model
        """
        norm_rates = Metrics.normalize(rates)
        target_mean = targets.mean()
        num = np.mean((targets - norm_rates)**2)
        denom = np.mean ((targets - target_mean)**2)
        pVar = 1 - num / (denom + 1e-10)
        return float(pVar)

    @staticmethod
    def compute_Qeff(rates, threshold):
        """
        Qeff    = dimensionality of network activity 
                = number of principal components that capture [threshold]% of the variance in the dynamics
        """
        norm_rates = Metrics.normalize(rates)
        centered_rates = norm_rates - norm_rates.mean(axis=1, keepdims=True)
        Q = centered_rates @ centered_rates.T / norm_rates.shape[1]
        eigenval = np.linalg.eigvalsh(Q)
        eigenval = np.sort(eigenval)[::-1]
        eigenval = np.maximum(eigenval, 0)
        cumvar = np.cumsum(eigenval) / (eigenval.sum() + 1e-10)
        Qeff = int(np.searchsorted(cumvar, threshold)) + 1
        return Qeff

    # @staticmethod
    # def compute_NActive(rates, t, sigma):
    #     """
    #     NActive = fraction of neurons active at any moment
    #     """
    #     N = rates.shape[0]

    
    # @staticmethod
    # def compute_selectivity():


    @staticmethod
    def general_evaluation(rates, targets, t, sigma, threshold):
        results = {
            '': ["Model's metrics", "Article's metrics"],
            'bVar': [Metrics.compute_bVar(rates, t, sigma), '1'],
            'pVar': [Metrics.compute_pVar(targets, rates), '0.9'],
            'Qeff': [Metrics.compute_Qeff(rates, threshold), 'trueval']#,
            # 'NActive': [Metrics.compute_NActive, "truevao"]
        }
        # then compute their selectivity and add it to the dictionnaire avec la valeur attendue
        result_df = pd.DataFrame(results)
        return result_df
    
    @staticmethod
    def network_output(rates, T, dt):
        t = np.arange(0, T, dt)
        norm_rates = Metrics.normalize(rates)
        tCOM = Metrics.compute_tCOM(norm_rates, t)
        sort_idx = np.argsort(tCOM)
        plt.figure(figsize=(10,5))
        plt.imshow(norm_rates[sort_idx], aspect='auto', extent=[0, T, N, 0], cmap='hot')
        plt.colorbar()
        plt.xlabel('Time (s)')
        plt.ylabel('Neurons')
        plt.title('Network output')
        plt.show()