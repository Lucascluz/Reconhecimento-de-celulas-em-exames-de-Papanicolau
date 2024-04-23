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
        
        self.open_image()
        
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
        
    def place_graph(self, photo):
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        global zoom_factor

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
        # file_path = filedialog.askopenfilename()
        file_path = "380936485_726263259498711_6168372829584127727_n.jpg"
        
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
            self.hu_invariants_button.config(state="active")
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
        
        self.zoom_plus_button.config(state="active", command = lambda: self.zoom_plus(image_pil_gray))
        self.zoom_minus_button.config(state="active", command = lambda: self.zoom_minus(image_pil_gray))
        
    def revert_color(self, image_pil_color):
                
        self.place_image(image_pil_color)
        
        self.zoom_plus_button.config(state="active", command = lambda: self.zoom_plus(image_pil_color))
        self.zoom_minus_button.config(state="active", command = lambda: self.zoom_minus(image_pil_color))
        
    def convert_to_histogram_gray(self, image_pil_color):
        # Converte a imagem para tons de cinza
        gray_img = image_pil_color.convert('L')

        # Converte a imagem em tons de cinza para um array NumPy
        gray_array = np.array(gray_img)

        # Plota o histograma
        plt.figure(figsize=(15.5, 7.5))
        plt.hist(gray_array.ravel(), bins=16, color='gray')
        plt.title('Histograma de tons de cinza')

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
        

    def convert_to_histogram_hsv(self, image_pil_color):
        # Converte a imagem para o espaço de cores HSV
        hsv_img = image_pil_color.convert('HSV')

        # Converte a imagem HSV para um array NumPy
        hsv_array = np.array(hsv_img)

        # Plota o histograma para cada canal H, S e V
        plt.figure(figsize=(15.5, 7.5))

        for i, (channel, color) in enumerate(zip('HSV', 'bgr')):
            histogram, bins = np.histogram(hsv_array[:, :, i], bins=16, range=(0, 256))
            plt.plot(bins[:-1], histogram, color=color)

        plt.title('Histograma do espaço de cores HSV')

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
    

    def calculate_glcm(self, image):
        # Convertendo a imagem para escala de cinza
        image = color.rgb2gray(image)

        # Reduzindo a escala de cinza para 16 tons
        image = img_as_ubyte(image)
        image = image // 16

        # Calculando a matriz de co-ocorrência para diferentes distâncias
        distances = [1, 2, 4, 8, 16, 32]
        glcm = graycomatrix(image, distances, [0], 16, symmetric=True, normed=True)

        return glcm

    def get_haralick_descriptors(self, image_pil_color):
        # Calculating matrix of coocurrence
        glcm = self.calculate_glcm(image_pil_color)
            
        # Calculating Haralick descriptors
        contrast = graycoprops(glcm, 'contrast')
        homogeneity = graycoprops(glcm, 'homogeneity')

        # greycoprops function does not calculate entropy, so we will calculate it manually
        entropy = -np.sum(glcm*np.log2(glcm + np.finfo(float).eps))

        glcm_2d = np.sum(glcm, axis=2)

        # Create a 2x2 grid for plots
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(3, 2)

        # Plot co-occurrence matrix
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(glcm_2d, cmap='hot', interpolation='nearest')
        ax1.set_title('Matriz de Coocorrência')

        # Plot descriptors
        ax2 = fig.add_subplot(gs[0, 0])
        ax2.axis('off')  # Hide axes
        ax2.text(0, 0.7, f"Contraste: {contrast[0][0]}")
        ax2.text(0, 0.5, f"Homogeneidade: {homogeneity[0][0]}")
        ax2.text(0, 0.3, f"Entropia: {entropy}")

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

# Load the image
image = cv2.imread('image.jpg')

# Calculate the Hu moments
gray_hu_moments = calculate_hu_moments(image)
h_hu_moments, s_hu_moments, v_hu_moments = calculate_hsv_hu_moments(image)

# Create a new PIL image
img = Image.new('RGB', (500, 500), color = (73, 109, 137))

d = ImageDraw.Draw(img)

# Add the Hu moments to the image
d.text((10,10), f"Gray Hu Moments: {np.squeeze(gray_hu_moments)}")
d.text((10,30), f"H Channel Hu Moments: {np.squeeze(h_hu_moments)}")
d.text((10,50), f"S Channel Hu Moments: {np.squeeze(s_hu_moments)}")
d.text((10,70), f"V Channel Hu Moments: {np.squeeze(v_hu_moments)}")

img.show()

        
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
