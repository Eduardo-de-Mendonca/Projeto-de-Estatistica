from Models.models import GammaModel, NormalModel, BinomialModel
from Models.settings import *

# Gráficos
BOXPLOT_FIGSIZE = (8, 6)
BOXPLOT_GRID_ALPHA = 0.7

SCATTER_PLOT_FIGSIZE = (10, 6)
SCATTER_PLOT_ALPHA = 0.5
SCATTER_PLOT_GRID_ALPHA = 0.7

HISTOGRAM_FIGSIZE = (10, 6)
HISTOGRAM_ALPHA = 0.75
#HISTOGRAM_GRID_ALPHA = 0.7

# Modelo para cada variável
VAR_DETAILS = {
    'throughput_down': {
        'name': 'Throughput de Download',
        'unit': 'bps', 
        'model_mle': GammaModel, 
    },
    'throughput_up':   {
        'name': 'Throughput de Upload',
        'unit': 'bps', 
        'model_mle': GammaModel, 
    },
    'rtt_down':        {
        'name': 'RTT de Download',
        'unit': 's', 
        'model_mle': NormalModel, 
    },
    'rtt_up':          {
        'name': 'RTT de Upload',
        'unit': 's', 
        'model_mle': NormalModel, 
    },
    'packet_loss':     {
        'name': f'Perda de Pacotes (contagem de {BINOMIAL_FIXED_N})',
        'unit': '(quantidade)', 
        'model_mle': BinomialModel, 
    }
}