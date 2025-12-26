from abc import ABC, abstractmethod
from math import log, exp

from Models.results import Results
from Models.models import VariableData, Model, GammaModel, NormalModel

from Utilities.utilities import read_data, chi2_1df_pvalue
from Utilities.network_entity import NetworkEntity

from MainClasses.settings import *

class Part2Question(ABC):
    client: NetworkEntity
    server: NetworkEntity

    client_data: VariableData
    server_data: VariableData
    full_data: VariableData

    full_model: Model
    client_model: Model
    server_model: Model

    out_path: str
    attribute: str

    @abstractmethod
    def __init__(self, in_path: str, out_path: str, attribute: str): pass

    @abstractmethod
    def statistic(self): pass

    def init_data(self, in_path: str, out_path: str, attribute: str):
        '''
        A ser chamado 1 vez no início do init, lê o arquivo e inicializa client, server, client_data, server_data, full_data
        '''
        clients, servers = read_data(in_path)

        self.client = clients[CHOSEN_CLIENT]
        self.server = servers[CHOSEN_SERVER]

        self.out_path = out_path
        self.attribute = attribute

        # Calcular o MLE sob H_0
        self.client_data = self.client.__getattribute__(self.attribute)
        self.server_data = self.server.__getattribute__(self.attribute)

        # Com o setting de debug, usamos apenas 10 por cento dos dados
        if DEBUG_USE_LESS_DATA:
            self.client_data, _ = self.client_data.split_data(DEBUG_REDUCED_DATA_AMOUNT)
            self.server_data, _ = self.server_data.split_data(DEBUG_REDUCED_DATA_AMOUNT)

        self.full_data = self.client_data.join_data(self.server_data)

    def H_0_log_likelihood(self):
        '''
        Retorna ln da likelihood sob H0
        '''
        model = self.full_model
        data = self.full_data

        # Calcular a log likelihood sob H_0
        result = 0
        for x in data.data:
            l = model.pdf(x)
            result += log(l)

        return result

    def H_1_log_likelihood(self):
        '''
        Retorna ln da likelihood sob H1
        '''

        client_data = self.client_data
        server_data = self.server_data
        client_model = self.client_model
        server_model = self.server_model

        result = 0
        for x in client_data.data:
            l = client_model.pdf(x)
            result += log(l)
        for x in server_data.data:
            l = server_model.pdf(x)
            result += log(l)

        return result

    def p_value(self, statistic: float):
        x = statistic
        return chi2_1df_pvalue(x)

    def write(self):
        r = Results()

        r.write(f'Atributo: {self.attribute}')
        
        if DEBUG_USE_LESS_DATA:
            r.write(f'DEBUG: estamos usando apenas {DEBUG_REDUCED_DATA_AMOUNT} dos dados')

        r.skipline()

        h0l = self.H_0_log_likelihood()
        h1l = self.H_1_log_likelihood()
        r.write(f'Log-likelihood sob H0: {h0l}')
        r.write(f'Log-likelihood sob H1: {h1l}')
        
        x = h0l - h1l
        r.write(f'Subtração das log-likelihoods (log Lambda): {x}')
        r.write(f'Lambda: {exp(x)}')
        r.write(f'-2 log Lambda: {-2*x}')
        r.skipline()

        w = self.statistic()
        r.write(f'Estatística do teste (é pra dar igual): {w}')
        r.write(f'p-valor: {self.p_value(w)}')
        r.write(f'Valor crítico: {CHI2_CRITICAL_VALUE_5}')
        r.skipline()

        r.write(f'Se w <= valor crítico e p >= 0.05, não rejeitamos H0')
        r.write(f'Se w > valor crítico e p < 0.05, rejeitamos H0')

        r.generate_file(self.out_path)

class Part2Q1(Part2Question):
    client_model: GammaModel
    server_model: GammaModel
    full_model: GammaModel

    def __init__(self, in_path: str, out_path: str, attribute: str):
        self.init_data(in_path, out_path, attribute)

        # Calcular MLE sob H_0
        self.full_model = GammaModel.from_mle(self.full_data)
        k = self.full_model.k

        # Calcular o MLE sob H_1
        self.client_model = GammaModel.from_mle_fixed_k(self.client_data, k)
        self.server_model = GammaModel.from_mle_fixed_k(self.server_data, k)
    
    def statistic(self):
        k = self.full_model.k

        nA = len(self.client_data.data)
        nB = len(self.server_data.data)

        avgY = self.full_data.average()
        avgYa = self.client_data.average()
        avgYb = self.server_data.average()

        factor1 = 2*k
        factor2 = nA*log(avgY/avgYa) + nB*log(avgY/avgYb)
        return factor1*factor2
    
class Part2Q2(Part2Question):
    client_model: NormalModel
    server_model: NormalModel
    full_model: NormalModel

    def __init__(self, in_path: str, out_path: str, attribute: str):
        self.init_data(in_path, out_path, attribute)

        # Calcular o MLE sob H_0
        self.full_model = NormalModel.from_mle(self.full_data)
        std_dev = self.full_model.std_dev

        # Calcular o MLE sob H_1
        self.client_model = NormalModel.from_mle_fixed_stddev(self.client_data, std_dev)
        self.server_model = NormalModel.from_mle_fixed_stddev(self.server_data, std_dev)

    def statistic(self):
        sigma = self.full_model.std_dev

        nA = len(self.client_data.data)
        nB = len(self.server_data.data)

        avgYa = self.client_data.average()
        avgYb = self.server_data.average()

        factor1 = 1/(sigma**2)
        factor2 = (nA*nB)/(nA + nB)
        factor3 = (avgYa - avgYb)**2

        return factor1*factor2*factor3

