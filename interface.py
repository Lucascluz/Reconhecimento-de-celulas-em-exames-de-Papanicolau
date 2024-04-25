import os
import cv2
import io
import mahotas
import tempfile

import numpy as np
import tkinter as tk

from tkinter import filedialog
from tkinter import ttk
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from PIL import Image, ImageTk
from skimage import color
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops

global zoom_factor

class ImageViewerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True,)

        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,)
        
        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW, )

        self.image_label = tk.Label(self.image_frame, )
        self.image_label.pack()
            
        self.scrollbarVertical = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbarVertical.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbarVertical.set)

        self.scrollbarHorizontal = ttk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.canvas.xview, )
        self.scrollbarHorizontal.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=self.scrollbarHorizontal.set)

        # Criando um frame para os botões adicionais
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()
        
        self.open_button = tk.Button(self.button_frame, text="Open Image", command=self.open_image, width=20, height=2)
        self.open_button.pack(side=tk.LEFT)
        
        self.zoom_plus_button = tk.Button(self.button_frame, text="+", width=6, height=2, state="disabled")
        self.zoom_plus_button.pack(side=tk.LEFT)
        
        self.zoom_minus_button = tk.Button(self.button_frame, text="-", width=6, height=2, state="disabled")
        self.zoom_minus_button.pack(side=tk.LEFT)

        # Organizando os botões adicionais lado a lado
        
        # Botão para converter a imagem para tons de cinza
        self.gray_scale_button = tk.Button(self.button_frame, text="Gray Scale", width=20, height=2, state="disabled")
        self.gray_scale_button.pack(side=tk.LEFT)
        
        self.colored_button = tk.Button(self.button_frame, text="Color (RGB)", width=20, height=2, state="disabled")
        self.colored_button.pack(side=tk.LEFT)
        
        self.histograms_button = tk.Button(self.button_frame, text="Gray Hist.", width=20, height=2, state = "disabled")
        self.histograms_button.pack(side=tk.LEFT)
        
        self.hsv_space_button = tk.Button(self.button_frame, text="HSV Hist.", width=20, height=2, state = "disabled")
        self.hsv_space_button.pack(side=tk.LEFT)
        
        self.haralick_button = tk.Button(self.button_frame, text="Haralick", width=20, height=2, state = "disabled")
        self.haralick_button.pack(side=tk.LEFT)
        
        self.hu_invariants_button = tk.Button(self.button_frame, text="Hu Invariants", width=20, height=2, state = "disabled")
        self.hu_invariants_button.pack(side=tk.LEFT)
        
        self.classify_button = tk.Button(self.button_frame, text="Classify", width=20, height=2, state = "disabled")
        self.classify_button.pack(side=tk.LEFT)
        
    def place_graph(self, photo):
        global zoom_factor
        
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        # Calcula as coordenadas para centralizar a imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo.width()
        image_height = photo.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Adiciona a imagem ao canvas centralizada
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image = photo)
        self.canvas.image = photo
        
        self.canvas.config(scrollregion=self.canvas.bbox("all"))    
        
    def place_image(self, image_pil):
        global zoom_factor
        
        # Atualizando o tamanho da imagem
        img_width = int(image_pil.width * zoom_factor)
        img_height = int(image_pil.height * zoom_factor)

        # Redimensionando a imagem
        image_resized = image_pil.resize((img_width, img_height))
        photo = ImageTk.PhotoImage(image_resized)
        
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        # Calcula as coordenadas para centralizar a imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo.width()
        image_height = photo.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Adiciona a imagem ao canvas centralizada
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image = photo)
        self.canvas.image = photo
        
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
            
    def open_image(self):
        file_path = filedialog.askopenfilename()
        
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil_base = Image.fromarray(image_rgb)

            # Define o fator de zoom para o valor padrão
            global zoom_factor
            zoom_factor = 1.0

            self.place_image(image_pil_base)
            
            self.open_button.config(text="Change Image")
            self.zoom_plus_button.config(state="active", command = lambda: self.zoom_plus(image_pil_base))
            self.zoom_minus_button.config(state="active", command = lambda: self.zoom_minus(image_pil_base))
            self.gray_scale_button.config(state="active", command= lambda:self.convert_to_grayscale(image_pil_base))
            self.colored_button.config(state="active", command= lambda: self.revert_color(image_pil_base))
            self.histograms_button.config(state="active", command = lambda: self.convert_to_histogram_gray(image_pil_base))
            self.hsv_space_button.config(state="active", command= lambda: self.convert_to_histogram_hsv(image_pil_base))
            self.haralick_button.config(state="active", command = lambda: self.get_haralick_descriptors(image_pil_base))
            self.hu_invariants_button.config(state="active", command= lambda: self.hu_invariants(image_pil_base))
            self.classify_button.config(state="active")
            
    def zoom_plus(self, image_pil_base):
        # Aumenta o fator de zoom em 10%
        global zoom_factor
        zoom_factor = zoom_factor + 0.1

        self.place_image(image_pil_base)

        
    def zoom_minus(self, image_pil_base):
        # Aumenta o fator de zoom em 10%
        global zoom_factor
        zoom_factor = zoom_factor - 0.1

        self.place_image(image_pil_base)
            
    def convert_to_grayscale(self, image_pil_color):
        # Converta a imagem PIL para uma imagem RGB e depois para uma matriz numpy
        image_pil_rgb = image_pil_color.convert('RGB')
        matrix_rgb = np.array(image_pil_rgb)

        # Use o OpenCV para converter a imagem para tons de cinza
        matrix_gray = cv2.cvtColor(matrix_rgb, cv2.COLOR_RGB2GRAY)

        # Converta a imagem em tons de cinza de volta para o formato PIL
        image_pil_gray = Image.fromarray(matrix_gray)
        
        # Coloque a imagem na tela
        self.place_image(image_pil_gray)
        
        self.zoom_plus_button.config(state="active")
        self.zoom_minus_button.config(state="active")
        self.gray_scale_button.config(state="disabled")
        self.colored_button.config(state="active")
        self.histograms_button.config(state="active") 
        self.hsv_space_button.config(state="active") 
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")
        
    def revert_color(self, image_pil_color):
                
        self.place_image(image_pil_color)
        
        self.zoom_plus_button.config(state="active")
        self.zoom_minus_button.config(state="active")
        self.gray_scale_button.config(state="active")
        self.colored_button.config(state="disabled")
        self.histograms_button.config(state="active") 
        self.hsv_space_button.config(state="active") 
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")
        
    def convert_to_histogram_gray(self, image_pil_color):
        # Converte a imagem para tons de cinza
        gray_img = image_pil_color.convert('L')

        # Converte a imagem em tons de cinza para um array NumPy
        gray_array = np.array(gray_img)

        # Reduz a imagem para 16 tons de cinza
        image = cv2.convertScaleAbs(gray_array, alpha=(15/255))

        # Calcula o histograma
        histogram = cv2.calcHist([image], [0], None, [16], [0, 16])

        # Plota o histograma
        # Define o tamanho da figura
        plt.figure(figsize=(12, 7))  # Você pode ajustar esses valores conforme necessário
        plt.title("Histograma de Tons de Cinza")
        plt.xlabel("Tons de Cinza")
        plt.ylabel("Quantidade de Ocorrências")
        plt.plot(histogram)
        plt.xlim([0, 15])

        # Cria um objeto BytesIO para salvar o gráfico
        buf = io.BytesIO()

        # Salva o gráfico no objeto BytesIO
        plt.savefig(buf, format='png')

        # Move o cursor do objeto BytesIO para o início
        buf.seek(0)

        # Carrega o objeto BytesIO como uma imagem PIL
        img = Image.open(buf)

        tk_img = ImageTk.PhotoImage(img)
        
        self.place_graph(tk_img)
        
        self.zoom_plus_button.config(state="disabled")
        self.zoom_minus_button.config(state="disabled")
        self.gray_scale_button.config(state="active")
        self.colored_button.config(state="active")
        self.histograms_button.config(state="disabled") 
        self.hsv_space_button.config(state="active") 
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")
        

    def convert_to_histogram_hsv(self, image_pil_color):
        # Converte a imagem para um array NumPy
        array = np.array(image_pil_color)
        
        # Convert the image to HSV
        hsv = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)

        # Split the HSV image into H, S and V channels
        h, s, v = cv2.split(hsv)

        # Calculate the 2D histogram for the H and V channels
        hist = cv2.calcHist([v, h], [0, 1], None, [8, 16], [0, 180, 0, 256])

        # Normalize the histogram
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Define o tamanho da figura
        plt.figure(figsize=(12, 7))  # Você pode ajustar esses valores conforme necessário

        # Plot the 2D histogram
        plt.imshow(hist, interpolation='nearest')
        plt.title('2D Color Histogram')
        plt.xlabel('V')
        plt.ylabel('H')

        # Cria um objeto BytesIO para salvar o gráfico
        buf = io.BytesIO()

        # Salva o gráfico no objeto BytesIO
        plt.savefig(buf, format='png', bbox_inches='tight')  # Adiciona bbox_inches='tight' para remover espaços em branco

        # Move o cursor do objeto BytesIO para o início
        buf.seek(0)

        # Carrega o objeto BytesIO como uma imagem PIL
        img = Image.open(buf)

        tk_img = ImageTk.PhotoImage(img)
        
        self.place_graph(tk_img)
        
        self.zoom_plus_button.config(state="disabled")
        self.zoom_minus_button.config(state="disabled")
        self.gray_scale_button.config(state="active")
        self.colored_button.config(state="active")
        self.histograms_button.config(state="active") 
        self.hsv_space_button.config(state="disabled") 
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")


    def get_haralick_descriptors(self, image_pil_color):
        # Converte a imagem para um array NumPy
        array = np.array(image_pil_color)
        
        image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

        # Convertendo a imagem RGB para tons de cinza
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Reduzindo a imagem para 16 tons de cinza
        image_gray //= 16

        # Convertendo a imagem para um array numpy
        image = np.array(image_gray)

        distances = [1, 2, 4, 8, 16, 31]
        angle = 0  # Ângulo constante

        fig, axs = plt.subplots(2, 3, figsize=(12, 7))  # Altere para 2 linhas e 3 colunas

        for i, d in enumerate(distances):
            row = i // 3  # Determina a linha do subplot
            col = i % 3   # Determina a coluna do subplot
            glcm = graycomatrix(image, distances=[d], angles=[angle], levels=16, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            entropy = -np.sum(glcm*np.log2(glcm + np.finfo(float).eps))
            axs[row, col].imshow(glcm[:, :, 0, 0], cmap="gray")  # Adicione os parâmetros vmin e vmax
            axs[row, col].set_title(f'Distância: {d}\nContraste: {contrast:.2f}\nHomogeneidade: {homogeneity:.2f}\nEntropia: {entropy:.2f}')

        plt.tight_layout()

        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Load the BytesIO object as a PIL image
        img = Image.open(buf)

        tk_img = ImageTk.PhotoImage(img)
        
        self.place_graph(tk_img)
        
        self.zoom_plus_button.config(state="disabled")
        self.zoom_minus_button.config(state="disabled")
        self.gray_scale_button.config(state="active")
        self.colored_button.config(state="active")
        self.histograms_button.config(state="active") 
        self.hsv_space_button.config(state="active") 
        self.haralick_button.config(state="disabled")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")
        
    def hu_invariants(self, image_pil_base):
        # Converte a imagem para um array NumPy
        array = np.array(image_pil_base)
        
        # Converter a imagem para tons de cinza
        imagem_cinza = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

        # Calcular os momentos invariantes de Hu para a imagem em tons de cinza
        momentos_hu_cinza = cv2.HuMoments(cv2.moments(imagem_cinza)).flatten()

        # Converter a imagem para o espaço de cores HSV
        imagem_hsv = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)

        # Calcular os momentos invariantes de Hu para cada canal do modelo HSV
        momentos_hu_hsv = [cv2.HuMoments(cv2.moments(imagem_hsv[:,:,i])).flatten() for i in range(3)]

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))

        # Definir as cores para as caixas de texto
        cores = ['gray', 'red', 'green', 'blue']

        # Exibir os momentos invariantes de Hu para a imagem em tons de cinza
        axs[0, 0].text(0.5, 0.0, '\n'.join(map(str, momentos_hu_cinza)), ha='center', va='top', size=25, bbox=dict(boxstyle='round', facecolor=cores[0], alpha=0.2))
        axs[0, 0].axis('off')
        axs[0, 0].set_title(f'Canal em tons de Cinza')

        # Exibir os momentos invariantes de Hu para os 3 canais do modelo HSV
        for i, momentos_hu in enumerate(momentos_hu_hsv, start=1):
            row = i // 2
            col = i % 2
            axs[row, col].text(0.5, 0.0, '\n'.join(map(str, momentos_hu)), ha='center', va='top', size=25, bbox=dict(boxstyle='round', facecolor=cores[i], alpha=0.2))
            axs[row, col].axis('off')
            axs[row, col].set_title(f'Canal {i} do modelo HSV')
            
        plt.subplots_adjust(hspace=0.2, wspace=0.5)
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Load the BytesIO object as a PIL image
        img = Image.open(buf)

        tk_img = ImageTk.PhotoImage(img)
        
        self.place_graph(tk_img)
        
        self.zoom_plus_button.config(state="disabled")
        self.zoom_minus_button.config(state="disabled")
        self.gray_scale_button.config(state="active")
        self.colored_button.config(state="active")
        self.histograms_button.config(state="active") 
        self.hsv_space_button.config(state="active") 
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="disabled")
        self.classify_button.config(state="active")
        
def main():
    # Cria a janela principal da aplicação
    root = tk.Tk()
    
    # Define o estado inicial da janela como maximizado
    root.state('zoomed')  # 'zoomed' maximiza a janela na maioria dos sistemas
    
    # Inicializa a aplicação e inicia o loop principal da interface gráfica
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
