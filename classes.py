import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
import os
from scipy.stats import gamma
from math import isclose
from settings import *

class Results:
    'Para guardar os resultados e escrever em um arquivo'
    def __init__(self): self.string = ''

    def write(self, a): self.string += f'{a}\n'

    def skipline(self): self.string += '\n'

    def generate_file(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(self.string)

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

    def mle_normal_avg(self):
        return self.average()
    
    def mle_normal_variance(self):
        return self.variance()

    def mle_binomial_p(self, fixed_n):
        '''
        Assumimos que cada dado é uma taxa percentual x. Interpretamos cada dado como sendo uma quantidade de sucessos, igual a (x %)(fixed_n).
        '''
        assert fixed_n >= 1
        # O MLE binomial é a soma de todas as quantidades de sucessos dividido pela soma de todas as quantidades de experimentos
        successes = 0
        total_experiments = 0
        for x in self.data:
            relative = x/100 # Pois a taxa vem percentual
            successes += relative*fixed_n
            total_experiments += fixed_n # Em cada ponto, temos mais n experimentos

        return successes/total_experiments
    
    def mle_gamma_k(self):
        shape, loc_result, scale = gamma.fit(self.data, floc=0)
        assert isclose(loc_result, 0)
        return shape
    
    def mle_gamma_beta(self):
        shape, loc_result, scale = gamma.fit(self.data, floc=0)
        assert isclose(loc_result, 0)
        return scale

class Server:
    'Representa um servidor'
    def __init__(self):
        # Em bps
        self.throughput_down = VariableData()
        self.throughput_up = VariableData()

        # Em segundos
        self.rtt_down = VariableData()
        self.rtt_up = VariableData()

        # Em %
        self.packet_loss = VariableData()

class Client:
    'Representa um cliente'
    def __init__(self):
        # Em bps
        self.throughput_down = VariableData()
        self.throughput_up = VariableData()

        # Em segundos
        self.rtt_down = VariableData()
        self.rtt_up = VariableData()

        # Em %
        self.packet_loss = VariableData()

class GraphGenerator:
    '''
    Gera todos os gráficos para um par de cliente e servidor selecionados
    '''
    def __init__(self, client_obj, client_name, server_obj, server_name, output_dir):
        assert isinstance(client_obj, Client)
        assert isinstance(server_obj, Server)

        self.client = client_obj
        self.client_name = client_name
        self.server = server_obj
        self.server_name = server_name

        self.output_dir = output_dir

        # Dicionário auxiliar para nomes e unidades dos gráficos
        self.var_details = {
            'throughput_down': {'name': 'Throughput de Download', 'unit': 'bps'},
            'throughput_up':   {'name': 'Throughput de Upload', 'unit': 'bps'},
            'rtt_down':        {'name': 'RTT de Download', 'unit': 's'},
            'rtt_up':          {'name': 'RTT de Upload', 'unit': 's'},
            'packet_loss':     {'name': 'Perda de Pacotes', 'unit': '%'}
        }

    def _plot_boxplot(self, client_data, server_data, var_name_key):
        """
        Gera um boxplot pareado para os dados de cliente e servidor.

        Especificação do Boxplot (default do matplotlib, whis=1.5):
        - Linha central: Mediana (50-percentil)
        - Caixa (Box): Do 25-percentil (Q1) ao 75-percentil (Q3)
        - IQR (Interquartile Range): Q3 - Q1
        - Bigodes (Whiskers): Estendem-se até o último dado que não é 
          considerado outlier.
          - Bigode superior: max(dado <= Q3 + 1.5 * IQR)
          - Bigode inferior: min(dado >= Q1 - 1.5 * IQR)
        - Outliers: Dados além dos bigodes (Q3 + 1.5*IQR ou Q1 - 1.5*IQR).
          Plotados como círculos ('o').
        """
        details = self.var_details[var_name_key]
        title_name = details['name']
        unit = details['unit']
        
        # Cria uma nova figura
        plt.figure(figsize=BOXPLOT_FIGSIZE)
        
        # Plota os dois boxplots lado a lado
        plt.boxplot(
            [client_data, server_data],

            labels=[self.client_name, self.server_name]
        )
        
        plt.title(f'Boxplot de {title_name}')
        plt.ylabel(f'{title_name} ({unit})')
        plt.grid(axis='y', linestyle='--', alpha=BOXPLOT_GRID_ALPHA) # Adiciona grade horizontal
        
        # Salva o arquivo
        filename = os.path.join(self.output_dir, f'boxplot_{var_name_key}.png')
        plt.savefig(filename)
        plt.close() # Fecha a figura para liberar memória

    def plot_all_boxplots(self):
        """Gera os 5 boxplots pareados."""
        for var_key in self.var_details.keys():
            var_data = getattr(self.client, var_key)
            assert isinstance(var_data, VariableData)
            client_data = var_data.data

            var_data = getattr(self.server, var_key)
            assert isinstance(var_data, VariableData)
            server_data = var_data.data
            self._plot_boxplot(client_data, server_data, var_key)

    def plot_scatter_plots(self):
        """Gera os 2 scatter plots de correlação."""
        # --- Gráfico 1: Cliente RTT Down vs Packet Loss ---
        plt.figure(figsize=SCATTER_PLOT_FIGSIZE)
        plt.scatter(
            self.client.rtt_down.data, 
            self.client.packet_loss.data, 
            alpha=SCATTER_PLOT_ALPHA # Transparência para ver pontos sobrepostos
        )
        plt.title(f'Correlação RTT Download vs Perda de Pacote ({self.client_name})')
        plt.xlabel(f"{self.var_details['rtt_down']['name']} ({self.var_details['rtt_down']['unit']})")
        plt.ylabel(f"{self.var_details['packet_loss']['name']} ({self.var_details['packet_loss']['unit']})")
        plt.grid(True, linestyle='--', alpha=SCATTER_PLOT_GRID_ALPHA)
        
        filename = os.path.join(self.output_dir, 'scatter_client_rtt_loss.png')
        plt.savefig(filename)
        plt.close()

        # --- Gráfico 2: Servidor RTT Up vs Packet Loss ---
        plt.figure(figsize=SCATTER_PLOT_FIGSIZE)
        plt.scatter(
            self.server.rtt_up.data, 
            self.server.packet_loss.data, 
            alpha=SCATTER_PLOT_ALPHA,
            color='orange' # Cor diferente para diferenciar
        )
        plt.title(f'Correlação RTT Upload vs Perda de Pacote ({self.server_name})')
        plt.xlabel(f"{self.var_details['rtt_up']['name']} ({self.var_details['rtt_up']['unit']})")
        plt.ylabel(f"{self.var_details['packet_loss']['name']} ({self.var_details['packet_loss']['unit']})")
        plt.grid(True, linestyle='--', alpha=SCATTER_PLOT_GRID_ALPHA)
        
        filename = os.path.join(self.output_dir, 'scatter_server_rtt_loss.png')
        plt.savefig(filename)
        plt.close()


    def _plot_histogram(self, data, entity_name, var_name_key):
        """Gera um único histograma."""
        details = self.var_details[var_name_key]
        title_name = details['name']
        unit = details['unit']

        plt.figure(figsize=HISTOGRAM_FIGSIZE)
        
        # 'bins="auto"' usa um estimador inteligente (ex: Freedman-Diaconis)
        # para decidir o número de barras.
        plt.hist(data, bins='auto', edgecolor='black', alpha=HISTOGRAM_ALPHA)
        
        plt.title(f'Histograma de {title_name} ({entity_name})')
        plt.xlabel(f'{title_name} ({unit})')
        plt.ylabel('Frequência')
        plt.grid(axis='y', linestyle='--', alpha=HISTOGRAM_GRID_ALPHA)
        
        filename = os.path.join(self.output_dir, f'hist_{entity_name}_{var_name_key}.png')
        plt.savefig(filename)
        plt.close()

        
    def plot_all_histograms(self):
        """Gera os 10 histogramas (5 para cliente, 5 para servidor)."""
        for var_key in self.var_details.keys():
            # Gráfico do Cliente
            var_data = getattr(self.client, var_key)
            assert isinstance(var_data, VariableData)
            self._plot_histogram(
                var_data.data, 
                self.client_name, 
                var_key
            )

            # Gráfico do Servidor
            var_data = getattr(self.client, var_key)
            assert isinstance(var_data, VariableData)
            self._plot_histogram(
                var_data.data, 
                self.server_name, 
                var_key
            )

class AllData:
    def __init__(self, path):
        '''
        Lê o CSV.
        Calcula quantos servidores há e quantos clientes há.
        Escreve em this.clients uma lista de objetos Client com os dados
        Escreve em this.servers uma lista de objetos Server com os dados
        '''
        self.clients = {}
        self.servers = {}

        with open(path, 'r', encoding='utf-8', newline='') as file:
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

                invalid = False
                for x in download_throughput, rtt_download, upload_throughput, rtt_upload, packet_loss:
                    if x < 0: invalid = True # Ignorar linhas com valor negativo
                if invalid: continue

                # Cria um novo objeto Client se for a primeira vez que o vemos
                if client_name not in self.clients:
                    self.clients[client_name] = Client()
                
                # Cria um novo objeto Server se for a primeira vez que o vemos
                if server_name not in self.servers:
                    self.servers[server_name] = Server()

                # Obtém os objetos correspondentes para adicionar os novos dados
                current_client = self.clients[client_name]
                current_server = self.servers[server_name]

                assert isinstance(current_client, Client)
                assert isinstance(current_server, Server)

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

    def __format_latex_number(self, x):
        """
        Formata um número para LaTeX.
        Usa notação científica para números muito grandes ou muito pequenos.
        """
        if pd.isna(x):
            return "NaN" # Trata valores ausentes

        # Define os limites: maior que 100.000 ou menor que 0.001 (e não zero)
        if abs(x) > 1e5 or (abs(x) < 1e-3 and abs(x) != 0):
            # Formata como a.bcde e+exp (ex: 1.2345e+08)
            s = f"{x:.4e}" 
            parts = s.split('e')
            mantissa = parts[0]
            exponent = int(parts[1])
            
            # Retorna a string em formato LaTeX: $1.2345 \times 10^{8}$
            # Precisamos de \\times (para o \), e {{ }} (para as chaves do expoente)
            return f"${mantissa} \\times 10^{{{exponent}}}$"
        else:
            # Para números "normais", usa 4 casas decimais
            return f"{x:.4f}"

    def __create_statistics_dataframe(self, data_dict, variable_name, percentile_list):
        """
        Função auxiliar para criar um DataFrame do Pandas com as estatísticas
        para uma dada variável e um dicionário de dados (clientes ou servidores).
        """

        assert isinstance(data_dict, dict)

        # Nomes das linhas da nossa tabela final
        index_names = [
            'Média', 'Variância', 'Desvio-padrão', 
            '0-percentil (mínimo)', '25-percentil', '50-percentil (mediana)',
            '75-percentil', '100-percentil (máximo)'
        ]
        
        # Dicionário para guardar os dados da tabela
        table_data = {}

        #Função auxiliar para "ordenação natural" (ex: client1, client2, client10)
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() 
                    for text in re.split('([0-9]+)', s)]
        
        # Ordena os nomes dos clientes/servidores usando a chave de ordenação
        sorted_names = sorted(data_dict.keys(), key=natural_sort_key)

        # Itera sobre cada cliente ou servidor
        for name in sorted_names:
            obj = data_dict[name]
            # Pega o atributo de dados da variável correta (ex: obj.throughput_down)
            variable_data = getattr(obj, variable_name)
            assert isinstance(variable_data, VariableData)

            # Calcula todas as estatísticas para a coluna atual
            stats = [
                variable_data.average(),
                variable_data.variance(),
                variable_data.std_deviation()
            ]

            for p in percentile_list:
                stats.append(variable_data.percentile(p))

            table_data[name] = stats

        # Cria o DataFrame do Pandas
        df = pd.DataFrame(table_data, index=index_names)
        #return df.to_string()

        '''
        return df.to_latex(
            #booktabs=True, # Usa \\usepackage{booktabs} para tabelas mais bonitas
            caption=var_title, # SOLICITAÇÃO 1: Adiciona o título (caption)
            float_format=self.__format_latex_number, # SOLICITAÇÃO 3: Formatação científica
            escape=False # Permite que o LaTeX da notação científica ($ \\times $) funcione
        )
        '''
    
        return df.to_latex(
            #booktabs=True,
            float_format=self.__format_latex_number, 
            escape=False
        )

    def __write_descriptive_statistics(self, out_path, percentile_list):
        """
        Gera as tabelas de estatísticas descritivas para todas as variáveis
        e as salva em um arquivo de texto.
        """
        # Usando a sua classe Results para construir a string de saída
        results = Results()

        results.write("\\documentclass{article}")
        results.write("\\usepackage[utf8]{inputenc}")
        results.write("\\usepackage[T1]{fontenc}")
        results.write("\\usepackage{booktabs} % Para tabelas mais bonitas")
        results.write("\\usepackage{amsmath}  % Para o comando \\times")

        results.write("\\usepackage{caption}")
        results.write("\\captionsetup{labelformat=empty}")

        results.write("\\usepackage{graphicx} % NOVO: Para usar \\resizebox")
        results.write("\\usepackage{float}    % NOVO: Para forçar a posição com [H]")
        # NOVO: Página em paisagem (landscape) e margens de 1cm
        results.write("\\usepackage[landscape, margin=1cm]{geometry}")

        results.write("\\begin{document}")
        results.skipline()

        # Lista das variáveis que queremos analisar
        variables = {
            'throughput_down': 'Throughput de Download (bps)',
            'throughput_up': 'Throughput de Upload (bps)',
            'rtt_down': 'RTT de Download (s)',
            'rtt_up': 'RTT de Upload (s)',
            'packet_loss': 'Perda de Pacotes (por cento)'
        }

        for var_name, var_title in variables.items():
            results.write(f'%{"="*80}')
            results.write(f'%Estatísticas para: {var_title}')
            results.write(f'%{"="*80}')
            results.skipline()

            # Tabela de Clientes
            results.write('%--- Tabela de Clientes ---')
            results.write("\\begin{table}[H]") # [H] força a tabela a ficar aqui
            results.write("\\centering")       # Centraliza a tabela
            
            client_title = f"Clientes: {var_title}"
            results.write(f"\\caption{{{client_title}}}") # Adiciona o caption
            
            # Comando para redimensionar:
            # \textwidth = largura total do texto na página
            # ! = manter a proporção (altura automática)
            results.write("\\resizebox{\\textwidth}{!}{") 
            
            # Gera APENAS o \begin{tabular} ... \end{tabular}
            client_table_string = self.__create_statistics_dataframe(
                self.clients, var_name, percentile_list
            )
            results.write(client_table_string)
            
            results.write("}") # Fecha o \resizebox
            results.write("\\end{table}") # Fecha o ambiente da tabela
            results.write("\\clearpage") # Força uma nova página para a próxima tabela

            # Tabela de Servidores
            results.write('%--- Tabela de Servidores ---')
            results.write("\\begin{table}[H]")
            results.write("\\centering")
            
            server_title = f"Servidores: {var_title}"
            results.write(f"\\caption{{{server_title}}}")
            
            results.write("\\resizebox{\\textwidth}{!}{")
            
            server_table_string = self.__create_statistics_dataframe(
                self.servers, var_name, percentile_list
            )
            results.write(server_table_string)
            
            results.write("}")
            results.write("\\end{table}")
            results.write("\\clearpage") # Força uma nova página
            results.skipline()

        results.write('\\end{document}')
        # Gera o arquivo final
        results.generate_file(out_path)

    def write_latex_tables(self):
        out_path = 'Output/descriptive_statistic.tex'
        percentile_list = [0, 25, 50, 75, 100]

        self.__write_descriptive_statistics(out_path, percentile_list)

    def write_amount_of_points(self):
        '''
        Escreve em um arquivo a quantidade de linhas de dados para cada cliente/servidor, para ajudar a selecionar os melhores
        '''

        out_path = 'Output/amount_of_points.txt'

        result = Results()
        for name, obj in self.clients.items():
            assert isinstance(obj, Client)
            n = len(obj.packet_loss.data)
            result.write(f'{name}: {n} pontos')
        for name, obj in self.servers.items():
            assert isinstance(obj, Server)
            n = len(obj.packet_loss.data)
            result.write(f'{name}: {n} pontos')

        result.generate_file(out_path)

    def generate_plots_for_selection(self):
        """
        Seleciona um cliente e um servidor específicos e instancia
        o GraphGenerator para criar todos os gráficos.
        """
        # --- 1. Selecionar os objetos ---
        client_name_to_select = 'client13'
        server_name_to_select = 'server07'
        output_dir = 'Output/Graphs'

        client_obj = self.clients[client_name_to_select]

        server_obj = self.servers[server_name_to_select]

        # --- 2. Instanciar o Gerador ---
        plotter = GraphGenerator(
            client_obj, 
            client_name_to_select, 
            server_obj, 
            server_name_to_select,
            output_dir
        )

        # --- 3. Gerar todos os gráficos ---
        plotter.plot_all_boxplots()
        plotter.plot_scatter_plots()
        plotter.plot_all_histograms()