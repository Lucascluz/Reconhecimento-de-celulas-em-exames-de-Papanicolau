import cv2
import io
import mahotas

import numpy as np
import tkinter as tk

from tkinter import filedialog
from tkinter import ttk
from matplotlib import pyplot as plt
from PIL import Image, ImageTk, ImageDraw
from skimage.feature import graycomatrix, graycoprops

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
        
        self.histograms_button = tk.Button(self.button_frame, text="Histogram", width=20, height=2, state = "disabled")
        self.histograms_button.pack(side=tk.LEFT)
        
        self.hsv_space_button = tk.Button(self.button_frame, text="HSV Space", width=20, height=2, state = "disabled")
        self.hsv_space_button.pack(side=tk.LEFT)
        
        self.haralick_button = tk.Button(self.button_frame, text="Haralick", width=20, height=2, state = "disabled")
        self.haralick_button.pack(side=tk.LEFT)
        
        self.hu_invariants_button = tk.Button(self.button_frame, text="Hu Invariants", width=20, height=2, state = "disabled")
        self.hu_invariants_button.pack(side=tk.LEFT)
        
        self.classify_button = tk.Button(self.button_frame, text="Classify", width=20, height=2, state = "disabled")
        self.classify_button.pack(side=tk.LEFT)
        
        self.open_image()
        
    def place_image(self, photo):
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
        
        # file_path = filedialog.askopenfilename()
        file_path = "380936485_726263259498711_6168372829584127727_n.jpg"
        
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil_base = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image_pil_base)

            self.place_image(photo)
            
            self.open_button.config(text="Change Image")
            self.gray_scale_button.config(state="active", command= lambda:self.convert_to_grayscale(image_pil_base))
            self.colored_button.config(state="active", command= lambda: self.revert_color(image_pil_base))
            self.histograms_button.config(state="active", command = lambda: self.convert_to_histogram_gray(image_pil_base))
            self.hsv_space_button.config(state="active", command= lambda: self.convert_to_histogram_hsv(image_pil_base))
            self.haralick_button.config(state="active", command = lambda: self.get_haralick_descriptors(image_pil_base))
            self.hu_invariants_button.config(state="active")
            self.classify_button.config(state="active")
            
    def zoom_plus(self, image_pil_base):
        print("ainda não implementado")
        
    def zoom_minus(self, image_pil_base):
        print("ainda não implementado")
            
    def convert_to_grayscale(self, image_pil_color):
        # Converta a imagem PIL para uma imagem RGB e depois para uma matriz numpy
        image_pil_rgb = image_pil_color.convert('RGB')
        matrix_rgb = np.array(image_pil_rgb)

        # Use o OpenCV para converter a imagem para tons de cinza
        matrix_gray = cv2.cvtColor(matrix_rgb, cv2.COLOR_RGB2GRAY)

        # Converta a imagem em tons de cinza de volta para o formato PIL
        image_pil_gray = Image.fromarray(matrix_gray)
        
        # Converta a imagem do formato PIL para o formato ImageTK
        photo_gray = ImageTk.PhotoImage(image_pil_gray)
        
        # Coloque a imagem na tela
        self.place_image(photo_gray)
        
    def revert_color(self, image_pil_color):
        photo_color = ImageTk.PhotoImage(image_pil_color)
        
        self.place_image(photo_color)
        
    def convert_to_histogram_gray(self, image_pil_color):
        # Converte a imagem para tons de cinza
        gray_img = image_pil_color.convert('L')

        # Converte a imagem em tons de cinza para um array NumPy
        gray_array = np.array(gray_img)

        # Plota o histograma
        plt.figure(figsize=(15.5, 7.5))
        plt.hist(gray_array.ravel(), bins=256, color='gray')
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
        
        self.place_image(tk_img)
        

    def convert_to_histogram_hsv(self, image_pil_color):
        # Converte a imagem para o espaço de cores HSV
        hsv_img = image_pil_color.convert('HSV')

        # Converte a imagem HSV para um array NumPy
        hsv_array = np.array(hsv_img)

        # Plota o histograma para cada canal H, S e V
        plt.figure(figsize=(15.5, 7.5))

        for i, (channel, color) in enumerate(zip('HSV', 'bgr')):
            histogram, bins = np.histogram(hsv_array[:, :, i], bins=256, range=(0, 256))
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
        
        self.place_image(tk_img)
        
    def get_haralick_descriptors(self, image_pil_color):
        # Convertendo a imagem em cores para tons de cinza
        gray_image = image_pil_color.convert('L')
                
        # Cria uma cópia da imagem
        gray_image_copy = np.copy(gray_image)

        # Usa a cópia da imagem na função graycomatrix
        glcm = graycomatrix(gray_image_copy, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

        # Calcule a homogeneidade a partir da GLCM
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        
        contrast = graycoprops(glcm, "contrast")[0,0]
        
        print("Homogeniade: " + str(homogeneity))
        
        print("Contraste : " + str(contrast))

        
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
