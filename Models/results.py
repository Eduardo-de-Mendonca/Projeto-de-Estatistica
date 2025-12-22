import os

class Results:
    'Para guardar os resultados e escrever em um arquivo'
    def __init__(self): self.string = ''

    def write(self, a): self.string += f'{a}\n'

    def skipline(self): self.string += '\n'

    def generate_file(self, file_name):
        # Se o path não existe, cria as pastas correspondentes
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(self.string)