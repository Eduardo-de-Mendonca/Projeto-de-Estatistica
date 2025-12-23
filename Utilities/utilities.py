import csv
from scipy.stats import chi2

from Models.models import *
from Models.settings import *

from Utilities.network_entity import NetworkEntity

def choose_prior(mle_model):
    '''
    Escolhe uma prior adequada para modelar uma variável_aleatória, com base em um modelo de MLE ajustado para a mesma variável aleatória
    '''
    assert isinstance(mle_model, Model)

    if isinstance(mle_model, BinomialModel):
        return BetaModel(1, 1) # Prior uniforme

    if USE_MLE_AS_REFERENCE_FOR_PRIORS:
        # Calcularemos a média e desvio-padrão do modelo a ajustar com base no mle
        # A média será o valor do mle, e o desvio padrão será C vezes a média
        # A desvantagem dessa abordagem é que enviesa a inferência bayesiana, fazendo com que seus resultados tenham convergência muito rápido para os do MLE e subestimem as incertezas
        C = 2000
        if isinstance(mle_model, BinomialModel):
            return BetaModel(1, 1) # Prior uniforme
        
        if isinstance(mle_model, NormalModel):
            mle_avg = mle_model.average
            # StdDev da Prior = C * Média da Prior (que é o MLE)
            prior_std_dev = abs(mle_avg) * C
            
            prior = NormalModel(average=mle_avg, std_dev=prior_std_dev)
            return prior
        
        if isinstance(mle_model, GammaModel):
            mle_rate = mle_model.beta
            average = mle_rate
            std_dev = abs(average)*C
            
            # Usamos as fórmulas para a esperança e variância de Gamma para chegar a um sistema. Resolvemos o sistema.
            a_0 = (average / std_dev)**2
            b_0 = average/(std_dev**2)
            
            prior = GammaModel(k=a_0, beta=b_0)
            return prior
        
    else:
        # A média e o desvio-padrão da prior serão 'chutados' (não levarão em consideração o MLE)
        # A média será um valor positivo pequeno, enquanto o desvio-padrão será um valor grande
        average = 0.05
        std_dev = 100
        
        if isinstance(mle_model, NormalModel):            
            prior = NormalModel(average=average, std_dev=std_dev)
            return prior
        
        if isinstance(mle_model, GammaModel):
            a_0 = (average / std_dev)**2
            b_0 = average/(std_dev**2)
            
            prior = GammaModel(k=a_0, beta=b_0)
            return prior
        
def read_data(in_path: str):
    '''
    Lê o CSV.
    Calcula quantos servidores há e quantos clientes há.
    
    Retorna uma tupla clients, servers, com dois dicionários. Cada dicionário associa o nome da entidade a um objeto NetworkEntity
    '''

    clients = {}
    servers = {}

    with open(in_path, 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader, None) # Pula o cabeçalho

        for row in reader:
            # Extrai e converte os dados da linha
            download_throughput = float(row[1])
            rtt_download = float(row[2])
            upload_throughput = float(row[3])
            rtt_upload = float(row[4])
            packet_loss = float(row[5])
            client_name = row[6]
            server_name = row[7]

            # Packet_loss é uma porcentagem. Transformaremos em uma quantidade de pacotes perdidos, multiplicando por fixed_n
            packet_loss = (packet_loss/100)*BINOMIAL_FIXED_N

            invalid = False
            for x in download_throughput, rtt_download, upload_throughput, rtt_upload, packet_loss:
                if x < 0: invalid = True # Ignorar linhas com valor negativo
            if invalid: continue

            # Cria um novo objeto Client se for a primeira vez que o vemos
            if client_name not in clients:
                clients[client_name] = NetworkEntity()
            
            # Cria um novo objeto Server se for a primeira vez que o vemos
            if server_name not in servers:
                servers[server_name] = NetworkEntity()

            # Obtém os objetos correspondentes para adicionar os novos dados
            current_client = clients[client_name]
            current_server = servers[server_name]

            assert isinstance(current_client, NetworkEntity)
            assert isinstance(current_server, NetworkEntity)

            # Adiciona os dados ao cliente
            current_client.throughput_down.data.append(download_throughput)
            current_client.throughput_up.data.append(upload_throughput)
            current_client.rtt_down.data.append(rtt_download)
            current_client.rtt_up.data.append(rtt_upload)
            current_client.packet_loss.data.append(packet_loss)

            # Adiciona os dados ao servidor
            current_server.throughput_down.data.append(download_throughput)
            current_server.throughput_up.data.append(upload_throughput)
            current_server.rtt_down.data.append(rtt_download)
            current_server.rtt_up.data.append(rtt_upload)
            current_server.packet_loss.data.append(packet_loss)

        return clients, servers
    
def chi2_1df_pvalue(x):
    '''
    Se X segue chi-square com df=1, retorna a probabilidade de X > x
    '''
    # Quanto maior x, então menor a razão, então menor a likelihood sob H0, então pior é H0. O p-valor é a probabilidade de termos um resultado pelo menos tão desfavorável a H0 quanto o obtido. Logo, é a probabilidade de X > x
    return 1 - chi2.cdf(x, df=1)