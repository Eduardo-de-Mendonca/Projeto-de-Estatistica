from math import isclose

from scipy.stats import gamma, norm, binom
from scipy import stats

import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from Models.settings import *

class VariableData:
    'Representa uma lista de dados associada a uma variável específica'
    def __init__(self):
        self.data = []

    def average(self):
        return sum(self.data)/len(self.data)
    
    def median(self):
        return self.percentile(50)
        
    def variance(self):
        s = 0
        avg = self.average()

        for d in self.data:
            s += (d - avg)**2
        
        return s/len(self.data) # Dividimos por n, e não n - 1
    
    def std_deviation(self):
        return self.variance()**(1/2)

    def percentile(self, p):
        'Retorna o p-ésimo percentil. Isto é, o menor x nos dados tal que ao menos p% das amostras são menores que ou iguais a x'
        sorted_data = sorted(self.data)

        n = len(sorted_data)
        assert n >= 1
        i = (p*(n - 1))//100 # Arredondei para baixo

        # Exemplo: p = 50 e n = 3
        # i = 1 (deu certo)
        # 66% das amostras são menores que ou iguais

        # Outro exemplo: p = 75 e n = 4
        # i = 2
        # É o penúltimo. 75% das amostras são menores ou iguais
        # Se fosse o anterior, 50% seriam menores ou iguais. Então tá certo
        return sorted_data[i]
    
    def split_data(self, train_frac):
        """
        Divide os dados em treino (exemplo: 70%) e teste (30%).
        Retorna dois novos objetos VariableData.
        """
        n = len(self.data)
        n_train = int(n * train_frac)
        
        data_train = self.data[:n_train]
        data_test = self.data[n_train:]
        
        vd_train = VariableData()
        vd_train.data = data_train
        
        vd_test = VariableData()
        vd_test.data = data_test
        
        return vd_train, vd_test

    def join_data(self, other):
        '''
        Retorna um novo VariableData, que é este concatenado com o outro
        '''
        result = VariableData()
        result.data = self.data + other.data
        return result

class Model(ABC):
    @abstractmethod
    def draw(self, scale_factor): pass

    @abstractmethod
    def from_mle(variable_data): pass

    @abstractmethod
    def draw_qq_plot(self, data, ax):
        pass

class NormalModel(Model):
    def __init__(self, average, std_dev):
        self.average = average
        self.std_dev = std_dev

    def from_mle(variable_data):
        assert isinstance(variable_data, VariableData)
        avg = variable_data.average()
        std_dev = variable_data.variance()**(1/2)
        return NormalModel(avg, std_dev)
    
    def draw(self, scale_factor):
        """
        Desenha a PDF da Normal sobre o histograma existente, multiplicada por um fator de escala (o qual deve ser calculado pelo caller para alinhar a PDF ao histograma)
        """

        x_min, x_max = plt.xlim() # Pega os limites do histograma
        x = np.linspace(x_min, x_max, MLE_PLOT_POINT_AMOUNT)

        # Calcula a PDF e a plota
        y = norm.pdf(x, loc=self.average, scale=self.std_dev)
        plt.plot(x, y * scale_factor, 'r-', linewidth=LINEWIDTH, label='Fit Normal')

    def __repr__(self):
        return f'NormalModel(average={self.average},std_dev={self.std_dev})'

    def draw_qq_plot(self, data, filename):
        assert data
        assert self.std_dev > 0
        assert not(pd.isna(self.average))
        assert not(pd.isna(self.std_dev))
            
        plt.figure() # Cria nova figura
        ax = plt.gca() # Pega o eixo atual
        
        stats.probplot(data, dist=stats.norm, sparams=(self.average, self.std_dev), plot=ax)
        ax.set_title('QQ Plot Normal')
        
        plt.savefig(filename)
        plt.close()

    def expected_value(self):
        """Retorna o valor esperado da distribuição"""
        return self.average
    
    def variance(self):
        return self.std_dev**2

    def calculate_posterior(self, variable_data, sigma_sq_fixed):
        """
        Premissas: self representa a prior para o parâmetro média de um outro modelo normal. A variância do modelo é fixa em sigma_sq_fixed.

        Retorna a posterior Normal, tratando 'self' como a prior.
        """
        assert isinstance(variable_data, VariableData)
        r_bar = variable_data.average()
        n = len(variable_data.data)
        
        # Lida com dados vazios ou variância MLE inválida
        assert n > 0
        assert not(pd.isna(r_bar))
        assert not(pd.isna(sigma_sq_fixed))
        assert sigma_sq_fixed > 0

        mu_0 = self.average
        tau_0_sq = self.std_dev**2

        precision_0 = 1 / tau_0_sq
        precision_data = n / sigma_sq_fixed
        
        tau_n_sq = 1 / (precision_0 + precision_data)
        mu_n = tau_n_sq * ( (mu_0 * precision_0) + (r_bar * precision_data) )
        
        return NormalModel(mu_n, tau_n_sq**(1/2)) # Retorna a posterior

    def from_posterior(posterior_normal, sigma_sq_known):
        """
        Premissas: posterior_normal é um NormalModel representando a distribuição de probabilidades da média de uma outra Normal, cuja variância é fixa igual a sigma_sq_known

        Cria e retorna a PosteriorPredictive
        """
        assert isinstance(posterior_normal, NormalModel)
        
        mu_n = posterior_normal.average
        # tau_n_sq é a variância da posterior
        tau_n_sq = posterior_normal.std_dev**2
        
        predictive_variance = sigma_sq_known + tau_n_sq
        
        return NormalModel(mu_n, predictive_variance)

class BinomialModel(Model):
    def __init__(self, p, n):
        self.p = p
        self.n = n

    def from_mle(variable_data):
        '''
        Assumimos que cada dado é uma taxa percentual x. Interpretamos cada dado como sendo uma quantidade de sucessos, igual a (x %)(fixed_n).
        '''
        assert isinstance(variable_data, VariableData)
        fixed_n = BINOMIAL_FIXED_N
        assert fixed_n >= 1

        # O MLE binomial é a soma de todas as quantidades de sucessos dividido pela soma de todas as quantidades de experimentos
        successes = 0
        total_experiments = 0
        for x in variable_data.data:
            successes += x
            total_experiments += fixed_n # Em cada ponto, temos mais n experimentos

        p = successes/total_experiments
        return BinomialModel(p, fixed_n)

    def draw(self, scale_factor):
        """
        Desenha a PMF da Binomial sobre o histograma existente, multiplicada pelo fator de escala
        """
        x_min, x_max = plt.xlim() # Pega os limites do histograma
        # Cria o eixo X (contagens) e o eixo Y (PMF)
        x_counts = np.arange(0, int(x_max) + 2) # Vamos até int(x_max) + 1, isto é, o menor inteiro maior que x_max
        y_pmf = binom.pmf(x_counts, n=self.n, p=self.p)
        
        # Plota como "pirulitos" (stem plot) para dados discretos
        plt.plot(x_counts, y_pmf * scale_factor, 'ro', markersize=BINOMIAL_MARKERSIZE, label='Fit Binomial')
        plt.vlines(x_counts, 0, y_pmf * scale_factor, colors='r', lw=BINOMIAL_LINEWIDTH, alpha=BINOMIAL_ALPHA)

    def __repr__(self):
        return f"BinomialModel(p={self.p}, n={self.n})"

    def draw_qq_plot(self, data, filename):
        assert data
        assert not(pd.isna(self.p))

        plt.figure() # Cria nova figura
        ax = plt.gca() # Pega o eixo atual

        stats.probplot(data, dist=stats.binom, sparams=(self.n, self.p), plot=ax)
        ax.set_title('QQ Plot Binomial')
        
        plt.savefig(filename)
        plt.close()

class GammaModel(Model):
    def __init__(self, k, beta):
        self.k = k
        self.beta = beta

    def from_mle(variable_data):
        assert isinstance(variable_data, VariableData)
        for d in variable_data.data: assert d > 0

        shape, loc_result, scale = gamma.fit(variable_data.data, floc=0)
        assert isclose(loc_result, 0)

        k = shape
        beta = 1/scale
        return GammaModel(k, beta)
    
    def from_mle_fixed_k(variable_data: VariableData, k: float):
        for d in variable_data.data: assert d > 0

        avg = variable_data.average()
        beta_hat = k / avg
        return GammaModel(k, beta_hat)

    def draw(self, scale_factor):
        """
        Desenha a PDF da Gamma sobre o histograma existente.
        """
        # Cria o eixo X para a curva da PDF
        x_min, x_max = plt.xlim() # Pega os limites do histograma
        x = np.linspace(x_min, x_max, MLE_PLOT_POINT_AMOUNT)

        # Repare: o gamma do scipy usa scale = 1/beta
        # Calcula a PDF e a plota (scipy usa 'a' para shape/k, 'scale' para beta)
        y = gamma.pdf(x, a=self.k, loc=0, scale=1/self.beta)
        plt.plot(x, y * scale_factor, 'r-', linewidth=LINEWIDTH, label='Fit Gamma')
        
    def __repr__(self):
        return f"GammaModel(k={self.k}, beta={self.beta})"

    def draw_qq_plot(self, data, filename):
        assert data
        assert not(pd.isna(self.k))
        assert not(pd.isna(self.beta))
        for d in data: assert d > 0
            
        plt.figure() # Cria nova figura
        ax = plt.gca() # Pega o eixo atual
        
        scale_param = 1.0 / self.beta
        stats.probplot(data, dist=stats.gamma, sparams=(self.k, 0, scale_param), plot=ax)
        ax.set_title('QQ Plot Gamma')
        
        plt.savefig(filename)
        plt.close()

    def expected_value(self):
        """Retorna o valor esperado da distribuição"""
        assert self.beta != 0
        return self.k / self.beta

    def calculate_posterior(self, variable_data, k_fixed):
        """
        Premissas: self é a prior para o parâmetro beta (rate) de outra Gamma, com k fixo.
        Retorna a posterior Gamma.
        self.k -> a_0
        self.beta -> b_0 (rate)
        k_fixed -> k (shape da likelihood)
        """
        assert isinstance(variable_data, VariableData)
        n = len(variable_data.data)
        assert n > 0

        a_n = self.k + n * k_fixed
        b_n = self.beta + sum(variable_data.data)
        
        return GammaModel(a_n, b_n) # Retorna a posterior

    def pdf(self, x: float):
        '''
        Retorna o valor da minha pdf, avaliada em x
        '''
        return gamma.pdf(x, a=self.k, loc=0, scale=1/self.beta)

class BetaModel(Model):
    """Representa a distribuição Beta, usada como prior/posterior para Binomial."""
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return f"BetaModel(a={self.a}, b={self.b})"

    def from_mle(variable_data):
        # Não aplicável, esta classe não é um fit MLE de dados
        raise RuntimeError('Não implementado')

    def draw(self, scale_factor):
        # Não aplicável, prior/posterior não é desenhada sobre histograma
        raise RuntimeError('Não implementado')
    
    def draw_qq_plot(self, data, filename):
        # Não aplicável
        raise RuntimeError('Não implementado')

    # --- NOVOS MÉTODOS BAYESIANOS ---
    def expected_value(self):
        """Retorna o valor esperado da distribuição, que representa o parâmetro p"""
        return self.a / (self.a + self.b)

    def calculate_posterior(self, variable_data, fixed_n):
        """
        Premissas: self é a prior para o parâmetro p de um modelo binomial com n fixo.
        Retorna a posterior Beta
        self.a -> a_0
        self.b -> b_0
        """
        assert isinstance(variable_data, VariableData)
        # variable_data.data contém contagens x_t
        x_tot = sum(variable_data.data)
        n_tot = len(variable_data.data) * fixed_n
        
        a_n = self.a + x_tot
        b_n = self.b + (n_tot - x_tot)
        
        return BetaModel(a_n, b_n) # Retorna a posterior
    
class BetaBinomialPredictiveModel(Model):
    """ 
    Modelo Preditivo Posterior Beta-Binomial.
    """
    def __init__(self, a_n, b_n, n_star):
        self.a_n = a_n     # Parâmetro 'a' da posterior Beta
        self.b_n = b_n     # Parâmetro 'b' da posterior Beta
        self.n_star = n_star

    def __repr__(self):
        return f"BetaBinomialPredictiveModel(a_n={self.a_n}, b_n={self.b_n}, n_star={self.n_star})"

    def from_posterior(posterior_beta, n_star):
        """
        Cria modelo preditivo a partir da posterior Beta.
        """
        assert isinstance(posterior_beta, BetaModel)
        # posterior_beta.a é a_n, posterior_beta.b é b_n
        return BetaBinomialPredictiveModel(posterior_beta.a, posterior_beta.b, n_star)

    def expected_value(self):
        """Retorna E[X_novo|D] (contagem de perdas predita)."""
        return self.n_star * (self.a_n / (self.a_n + self.b_n))
    
    def variance(self):
        """Retorna Var[X_novo|D] (variância da contagem predita)."""
        num = self.n_star * self.a_n * self.b_n * (self.a_n + self.b_n + self.n_star)

        den = (self.a_n + self.b_n)**2 * (self.a_n + self.b_n + 1)
        
        assert den != 0

        return num / den
        
    def from_mle(variable_data):
        raise NotImplementedError
    def draw(self, scale_factor):
        raise NotImplementedError
    def draw_qq_plot(self, data, filename):
        raise NotImplementedError
    
class ScaledBetaPrimePredictiveModel(Model):
    """ 
    Modelo Preditivo Posterior Beta-Prime Escalada.
    """
    def __init__(self, a_n, k, b_n):
        self.a_n = a_n # Parâmetro 'a' (shape) da posterior Gamma(a_n, b_n)
        self.k = k     # Parâmetro 'k' (shape) *conhecido* da likelihood
        self.b_n = b_n # Parâmetro 'b' (rate) da posterior Gamma(a_n, b_n)
    
    def __repr__(self):
        return f"ScaledBetaPrimePredictiveModel(a_n={self.a_n}, k={self.k}, b_n={self.b_n})"
    
    def from_posterior(posterior_gamma, k_known):
        """Cria modelo preditivo a partir da posterior Gamma."""
        assert isinstance(posterior_gamma, GammaModel)
        # posterior_gamma.k é a_n (shape da posterior)
        # posterior_gamma.beta é b_n (rate da posterior)
        # k_known é o k (shape da likelihood)
        return ScaledBetaPrimePredictiveModel(posterior_gamma.k, k_known, posterior_gamma.beta)
    
    def expected_value(self):
        assert self.a_n > 1
        return (self.k * self.b_n) / (self.a_n - 1)
        
    def variance(self):
        assert self.a_n > 2
        num = self.k * (self.k + self.a_n - 1) * (self.b_n**2)
        den = ((self.a_n - 1)**2) * (self.a_n - 2)

        assert den != 0

        return num / den
        
    def from_mle(variable_data):
        raise NotImplementedError
    def draw(self, scale_factor):
        raise NotImplementedError
    def draw_qq_plot(self, data, filename):
        raise NotImplementedError