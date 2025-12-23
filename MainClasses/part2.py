from abc import ABC, abstractmethod
from math import log

from Models.results import Results
from Models.models import GammaModel
from Models.models import VariableData

from Utilities.utilities import read_data, chi2_1df_pvalue
from Utilities.network_entity import NetworkEntity

from MainClasses.settings import *

class Part2q1:
    def __init__(self, in_path: str, out_path: str, attribute: str):
        clients, servers = read_data(in_path)

        self.client = clients[CHOSEN_CLIENT]
        self.server = servers[CHOSEN_SERVER]

        self.out_path = out_path
        self.attribute = attribute

        # Calcular o MLE sob H_0
        self.client_data = self.client.__getattribute__(self.attribute)
        self.server_data = self.server.__getattribute__(self.attribute)
        assert isinstance(self.client_data, VariableData)
        assert isinstance(self.server_data, VariableData)

        # Com o setting de debug, usamos apenas 10 por cento dos dados
        if DEBUG_USE_LESS_DATA:
            self.client_data, _ = self.client_data.split_data(DEBUG_REDUCED_DATA_AMOUNT)
            self.server_data, _ = self.server_data.split_data(DEBUG_REDUCED_DATA_AMOUNT)

        self.full_data = self.client_data.join_data(self.server_data)

        self.full_model = GammaModel.from_mle(self.full_data)

        # Calcular o MLE sob H_1
        self.client_model = GammaModel.from_mle(self.client_data)
        self.server_model = GammaModel.from_mle(self.server_data)

    def H_0_likelihood(self):
        model = self.full_model
        data = self.full_data

        # Calcular a likelihood sob H_0
        prod = 1
        for x in data.data:
            prod *= model.pdf(x)

        return prod
    
    def H_1_likelihood(self):
        assert isinstance(self.client_data, VariableData)
        assert isinstance(self.server_data, VariableData)

        client_data = self.client_data
        server_data = self.server_data
        client_model = self.client_model
        server_model = self.server_model

        prod = 1
        for x in client_data.data:
            prod *= client_model.pdf(x)
        for x in server_data.data:
            prod *= server_model.pdf(x)

        return prod
    
    def statistic(self):
        assert isinstance(self.client_data, VariableData)
        assert isinstance(self.server_data, VariableData)

        k = self.full_model.k

        nA = len(self.client_data.data)
        nB = len(self.server_data.data)

        avgY = self.full_data.average()
        avgYa = self.client_data.average()
        avgYb = self.server_data.average()

        factor1 = 2*k
        factor2 = nA*log(avgY/avgYa) + nB*log(avgY/avgYb)
        return factor1*factor2
    
    def p_value(self, statistic: float):
        x = statistic
        return chi2_1df_pvalue(x)
    
    def write(self):
        r = Results()

        r.write(f'Atributo: {self.attribute}')
        
        if DEBUG_USE_LESS_DATA:
            r.write(f'DEBUG: estamos usando apenas {DEBUG_REDUCED_DATA_AMOUNT} dos dados')

        r.skipline()

        h0l = self.H_0_likelihood()
        h1l = self.H_1_likelihood()
        r.write(f'Likelihood sob H0: {h0l}')
        r.write(f'Likelihood sob H1: {h1l}')
        
        Lambda = h0l/h1l
        r.write(f'Razão: {Lambda}')
        r.write(f'-2 log Lambda: {-2*log(Lambda)}')
        r.skipline()

        w = self.statistic()
        r.write(f'Estatística do teste (é pra dar igual): {w}')
        r.write(f'p-valor: {self.p_value(w)}')
        r.write(f'Valor crítico: {CHI2_CRITICAL_VALUE_5}')
        r.skipline()

        r.write(f'Se w <= valor crítico e p >= 0.05, não rejeitamos H0')
        r.write(f'Se w > valor crítico e p < 0.05, rejeitamos H0')

        r.generate_file(self.out_path)