
import builtins
import cv2
import os

import pandas as pd

input_folder = "images"
output_folder = "sub_images"

# Criar o diretório de saída se não existir
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Carregar o arquivo CSV com as coordenadas dos núcleos
csv_file = "classifications.csv"
df = pd.read_csv(csv_file)

#Itera sobre "characters.csv" e direciona cada núcleo recortado para o sub-diretório adequado
for index, row in df.iterrows():
        # Obter o caminho completo da imagem
        image_path = os.path.join(input_folder, row['image_filename'])

        # Verificar se a imagem existe
        if not os.path.exists(image_path):
            builtins.print(f"Imagem não encontrada: {image_path}") # type: ignore
            continue

        # Carregar a imagem
        image = cv2.imread(image_path)

        # Verificar se a imagem foi carregada com sucesso
        if image is None:
            print(f"Erro ao carregar a imagem: {image_path}")
            continue

        # Obter as coordenadas do núcleo
        x, y = int(row['nucleus_x']), int(row['nucleus_y'])

        # Verificar se as coordenadas estão dentro dos limites da imagem
        if x - 50 < 0 or y - 50 < 0 or x + 50 >= image.shape[1] or y + 50 >= image.shape[0]:
            print(f"Coordenadas inválidas para a imagem: {row['image_id']} na célula {row['cell_id']}")
            continue

        # Definir a região de interesse (ROI) para cortar a imagem
        roi = image[y-50:y+50, x-50:x+50]

        # Criar o diretório da classe (bethesda_system) se não existir
        class_folder = os.path.join(output_folder, str(row['bethesda_system']))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Construir o nome do arquivo para a subimagem
        file_name = f"{row['cell_id']}.png"

        # Construir o caminho completo de saída
        output_path = os.path.join(class_folder, file_name)

        # Redimensionar a imagem para 100x100 pixels
        roi_resized = cv2.resize(roi, (100, 100))

        # Salvar a imagem cortada no diretório de saída
        cv2.imwrite(output_path, roi_resized)
