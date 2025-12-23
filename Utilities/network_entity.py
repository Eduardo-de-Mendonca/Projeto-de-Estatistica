from Models.models import VariableData

class NetworkEntity:
    'Representa uma entidade de rede (cliente ou servidor)'
    def __init__(self):
        # Em bps
        self.throughput_down = VariableData()
        self.throughput_up = VariableData()

        # Em segundos
        self.rtt_down = VariableData()
        self.rtt_up = VariableData()

        # Em %
        self.packet_loss = VariableData()