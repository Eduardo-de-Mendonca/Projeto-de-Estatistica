import csv
import pandas as pd
import re
import os

from utilities import *

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

class Results:
    'Para guardar os resultados e escrever em um arquivo'
    def __init__(self): self.string = ''

    def write(self, a): self.string += f'{a}\n'

    def skipline(self): self.string += '\n'

    def generate_file(self, file_name):
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(self.string)
    
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

class GraphGenerator:
    '''
    Gera todos os gráficos para um par de cliente e servidor selecionados
    '''
    def __init__(self, client_obj, client_name, server_obj, server_name, output_dir):
        assert isinstance(client_obj, NetworkEntity)
        assert isinstance(server_obj, NetworkEntity)

        self.client = client_obj
        self.client_name = client_name
        self.server = server_obj
        self.server_name = server_name

        self.output_dir = output_dir

        # Dicionário auxiliar para nomes e unidades dos gráficos
        self.var_details = VAR_DETAILS

    def __plot_boxplot(self, client_data, server_data, var_name_key):
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
            self.__plot_boxplot(client_data, server_data, var_key)

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

    def __plot_histogram(self, data, entity_name, var_name_key, model = None):
        """Gera um único histograma."""
        details = self.var_details[var_name_key]
        title_name = details['name']
        unit = details['unit']

        plt.figure(figsize=HISTOGRAM_FIGSIZE)
        
        # 'bins="auto"' usa um estimador inteligente (ex: Freedman-Diaconis) para decidir o número de barras.
        # Esta linha desenha o histograma de fato
        counts, bin_edges, patches = plt.hist(data, bins='auto', edgecolor='black', alpha=HISTOGRAM_ALPHA)
        

        if model != None:
            assert isinstance(model, Model)
            bin_width = bin_edges[1] - bin_edges[0]
            scale_factor = bin_width*len(data)
            model.draw(scale_factor)
        
        # Esse pedaço escreve textos/legendas
        plt.title(f'Histograma de {title_name} ({entity_name})')
        plt.xlabel(f'{title_name} ({unit})')
        plt.ylabel('Frequência')
        plt.legend() # Põe legenda em tudo que tem label (nesse caso, na pdf do modelo)
        filename = os.path.join(self.output_dir, f'hist_{entity_name}_{var_name_key}.png')
        plt.savefig(filename)
        plt.close()
        
    def plot_all_histograms(self, draw_mle_models = False):
        """Gera os 10 histogramas (5 para cliente, 5 para servidor)."""
        for var_key in self.var_details.keys():
            for entity, name in ((self.client, self.client_name), (self.server, self.server_name)):
                var_data = getattr(entity, var_key)
                assert isinstance(var_data, VariableData)

                model = None
                if draw_mle_models:
                    model_class = self.var_details[var_key]['model_mle']
                    assert issubclass(model_class, Model)
                    model = model_class.from_mle(var_data)

                # Gráficos
                self.__plot_histogram(
                    var_data.data, 
                    name, 
                    var_key,
                    model
                )
    
    def __plot_qq(self, data, entity_name, var_name_key, model):
        """Chama o método de plotagem QQ do modelo."""
        assert isinstance(model, Model)
        
        filename = os.path.join(self.output_dir, f'qq_{entity_name}_{var_name_key}.png')
        
        model.draw_qq_plot(data, filename)

    def plot_all_qq_plots(self):
        """Gera os 10 QQ plots (5 para cliente, 5 para servidor)."""
        for var_key in self.var_details.keys():
            for entity, name in ((self.client, self.client_name), (self.server, self.server_name)):
                var_data = getattr(entity, var_key)
                assert isinstance(var_data, VariableData)

                model_mle = None
                model_class = self.var_details[var_key]['model_mle']
                assert issubclass(model_class, Model)
                
                model_mle = model_class.from_mle(var_data)

                self.__plot_qq(
                    var_data.data,
                    name,
                    var_key,
                    model_mle
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

                # Packet_loss é uma porcentagem. Transformaremos em uma quantidade de pacotes perdidos, multiplicando por fixed_n
                packet_loss = (packet_loss/100)*BINOMIAL_FIXED_N

                invalid = False
                for x in download_throughput, rtt_download, upload_throughput, rtt_upload, packet_loss:
                    if x < 0: invalid = True # Ignorar linhas com valor negativo
                if invalid: continue

                # Cria um novo objeto Client se for a primeira vez que o vemos
                if client_name not in self.clients:
                    self.clients[client_name] = NetworkEntity()
                
                # Cria um novo objeto Server se for a primeira vez que o vemos
                if server_name not in self.servers:
                    self.servers[server_name] = NetworkEntity()

                # Obtém os objetos correspondentes para adicionar os novos dados
                current_client = self.clients[client_name]
                current_server = self.servers[server_name]

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
            assert isinstance(obj, NetworkEntity)
            n = len(obj.packet_loss.data)
            result.write(f'{name}: {n} pontos')
        for name, obj in self.servers.items():
            assert isinstance(obj, NetworkEntity)
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

    def write_mle_models_to_file(self):
        """
        Calcula e escreve todos os modelos MLE (usando __repr__) para 
        os clientes e servidores selecionados em um arquivo.
        """
        client_name = 'client13'
        server_name = 'server07'
        entities = {
            client_name: self.clients[client_name],
            server_name: self.servers[server_name]
        }
        
        results = Results()
        details = VAR_DETAILS

        for name, entity in entities.items():
            results.write(f"--- Modelos MLE para: {name} ---")
            for var_key, var_info in details.items():
                var_data = getattr(entity, var_key)
                model_class = var_info['model_mle']
                
                assert issubclass(model_class, Model)
                if model_class == BinomialModel:
                    model_mle = model_class.from_mle(var_data)
                else:
                    model_mle = model_class.from_mle(var_data)
                
                results.write(f"{var_key}: {model_mle}")
            results.skipline()
        
        results.generate_file('Output/mle_models.txt')

    def plot_mle_models(self):
        # --- 1. Selecionar os objetos ---
        client_name_to_select = 'client13'
        server_name_to_select = 'server07'
        output_dir = 'Output/MLEgraphs'

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

        plotter.plot_all_histograms(True)
        plotter.plot_all_qq_plots()

    def run_bayesian_inference_100(self):
        """
        Roda a inferência Bayesiana com 100% dos dados e compara 
        E[theta|data] com o MLE.
        """

        client_name = 'client13'
        server_name = 'server07'
        entities = {
            client_name: self.clients[client_name],
            server_name: self.servers[server_name]
        }
        results = Results()
        results.write(f'USE_MLE_AS_REFERENCE_FOR_PRIORS = {USE_MLE_AS_REFERENCE_FOR_PRIORS}')

        for name, entity in entities.items():
            results.write(f"--- Inferência Bayesiana (100% Dados) para: {name} ---")
            for var_key, var_info in VAR_DETAILS.items():
                results.skipline()
                results.write(f"Variável: {var_key}")
                
                var_data = getattr(entity, var_key)
                model_mle_class = var_info['model_mle']
                
                # Calcular MLE
                assert issubclass(model_mle_class, Model)
                model_mle = model_mle_class.from_mle(var_data)
                
                # Calcular Prior (Lógica Empírica) e Posterior
                if var_key == 'packet_loss':
                    # Likelihood Binomial, Prior Beta (Fixa)
                    prior = choose_prior(model_mle)
                    posterior = prior.calculate_posterior(var_data, fixed_n=BINOMIAL_FIXED_N)
                    
                    results.write(f'Assumindo n fixo igual a {BINOMIAL_FIXED_N}...')
                    results.write(f'Prior para p: {prior}')
                    results.write(f'Posterior para p: {posterior}')
                    results.write(f"Esperança da posterior (MAP para p): {posterior.expected_value()}")

                    results.write(f"Estimativa MLE para p: {model_mle.p}")

                elif 'rtt' in var_key:
                    prior = choose_prior(model_mle)
                    
                    sigma_sq_known = model_mle.std_dev**2
                    posterior = prior.calculate_posterior(var_data, sigma_sq_known)
                    
                    results.write(f'Assumindo variância fixa igual a {sigma_sq_known}...')
                    results.write(f'Prior para a média: {prior}')
                    results.write(f'Posterior para a média: {posterior}')
                    results.write(f"Esperança da posterior (MAP para a média): {posterior.expected_value()}")
                    results.write(f"Estimativa MLE para a média: {model_mle.average}")
            
                elif 'throughput' in var_key:
                    k_fixed = model_mle.k
                    
                    prior = choose_prior(model_mle)
                    
                    posterior = prior.calculate_posterior(var_data, k_fixed)
                    
                    results.write(f'Assumindo k fixo igual a {k_fixed}...')
                    results.write(f'Prior para beta: {prior}')
                    results.write(f'Posterior para beta: {posterior}')
                    results.write(f'Esperança da posterior (MAP para beta): {posterior.expected_value()}')
                    results.write(f"Estimativa MLE para beta: {model_mle.beta}")
                
            results.skipline()

        results.generate_file('Output/bayes_100_report.txt')

    def run_bayesian_inference_70_30(self):
        """
        Roda a inferência com 70% dos dados, calcula o preditivo e compara com os 30% de teste.
        """
        client_name = 'client13'
        server_name = 'server07'
        entities = {
            client_name: self.clients[client_name],
            server_name: self.servers[server_name]
        }
        results = Results()
        results.write(f'USE_MLE_AS_REFERENCE_FOR_PRIORS = {USE_MLE_AS_REFERENCE_FOR_PRIORS}\n')

        for name, entity in entities.items():
            results.write(f"--- Predição Bayesiana (70/30 Split) para: {name} ---")
            for var_key, var_info in VAR_DETAILS.items():
                results.skipline()
                results.write(f"Variável: {var_key}")
                
                var_data_full = getattr(entity, var_key)
                assert isinstance(var_data_full, VariableData)
                vd_train, vd_test = var_data_full.split_data(train_frac=0.7)
                
                # Calcular estatísticas reais dos dados de teste (30%)
                test_avg = vd_test.average()
                test_var = vd_test.variance()

                # Calcular Posterior (usando 70% de treino)
                model_mle_class = var_info['model_mle']
                
                # Calcular MLE dos dados de treino (para priors empíricas e params. fixos)
                assert issubclass(model_mle_class, Model)
                model_mle_train = model_mle_class.from_mle(vd_train)
                prior = choose_prior(model_mle_train) # Usa o MLE de treino para a prior (quando relevante)
                
                # Calcular Posterior e Modelo Preditivo
                if 'rtt' in var_key:
                    assert isinstance(model_mle_train, NormalModel)
                    sigma_sq_known = model_mle_train.std_dev**2
                    posterior = prior.calculate_posterior(vd_train, sigma_sq_known)
                    pred_model = NormalModel.from_posterior(posterior, sigma_sq_known)
                
                elif var_key == 'packet_loss':
                    posterior = prior.calculate_posterior(vd_train, fixed_n=BINOMIAL_FIXED_N)
                    pred_model = BetaBinomialPredictiveModel.from_posterior(posterior, n_star=BINOMIAL_FIXED_N)
                
                elif 'throughput' in var_key:
                    assert isinstance(model_mle_train, GammaModel)
                    k_known = model_mle_train.k
                    posterior = prior.calculate_posterior(vd_train, k_known)
                    pred_model = ScaledBetaPrimePredictiveModel.from_posterior(posterior, k_known)
                
                # Obter estatísticas preditivas
                pred_avg = pred_model.expected_value()
                pred_var = pred_model.variance()
                
                # Escrever resultados
                results.write(f'Prior: {prior}')
                results.write(f"Posterior (de 70%): {posterior}")
                results.write(f"Modelo Preditivo: {pred_model}")
                results.write(f"Esperança do modelo preditivo: {pred_avg}")
                results.write(f'Variância do modelo preditivo: {pred_var}')
                results.write(f"Média real dos dados de teste (30%): {test_avg}")
                results.write(f'Variância real dos dados de teste (30%): {test_var}')
                results.write(f"COMPARAÇÃO (Média): {pred_avg:.4e} (Predito) vs {test_avg:.4e} (Real)")
                results.write(f"COMPARAÇÃO (Variância): {pred_var:.4e} (Predito) vs {test_var:.4e} (Real)")

            results.skipline()
        
        results.generate_file('Output/bayes_70_30_report.txt')