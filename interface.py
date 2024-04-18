from tkinter import filedialog
from tkinter import ttk

from PIL import Image, ImageTk, ImageDraw

import cv2

import tkinter as tk
import numpy as np



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
            
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil_base = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image_pil_base)

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
            self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo)
            self.canvas.image = photo

            self.open_button.config(text="Change Image")
            self.gray_scale_button.config(state="active", command= lambda:self.convert_to_grayscale(image_pil_base))
            self.colored_button.config(state="disabled")
            self.histograms_button.config(state="disabled")
            self.hsv_space_button.config(state="active", command= lambda: self.convert_to_hsv_space(image_pil_base))
            self.haralick_button.config(state="disabled")
            self.hu_invariants_button.config(state="active")
            self.classify_button.config(state="active")

            self.canvas.config(scrollregion=self.canvas.bbox("all"))
            
    def convert_to_grayscale(self, image_pil_color):
        # Converta a imagem PIL para uma imagem RGB e depois para uma matriz numpy
        image_pil_rgb = image_pil_color.convert('RGB')
        image_np = np.array(image_pil_rgb)

        # Use o OpenCV para converter a imagem para tons de cinza
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Converta a imagem em tons de cinza de volta para o formato PIL
        image_pil_gray = Image.fromarray(image_gray)
        
        photo_gray = ImageTk.PhotoImage(image_pil_gray)
        
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        # Calcula as coordenadas para centralizar a imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo_gray.width()
        image_height = photo_gray.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Adiciona a imagem ao canvas centralizada
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo_gray)
        self.canvas.image = photo_gray

        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.gray_scale_button.config(state="disabled")
        self.colored_button.config(state="active", command= lambda: self.revert_color(image_pil_color))
        self.histograms_button.config(state="active", command= lambda: self.convert_to_histograms(image_pil_color))
        self.hsv_space_button.config(state="disabled")
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="disabled")
        self.classify_button.config(state="active")
        
    def revert_color(self, image_pil_color):
        photo_color = ImageTk.PhotoImage(image_pil_color)
        
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        # Calcula as coordenadas para centralizar a imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo_color.width()
        image_height = photo_color.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Adiciona a imagem ao canvas centralizada
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo_color)
        self.canvas.image = photo_color

        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        self.gray_scale_button.config(state="active", command= lambda:self.convert_to_grayscale(image_pil_color))
        self.colored_button.config(state="di")
        self.histograms_button.config(state="disabled")
        self.hsv_space_button.config(state="active", command= lambda: self.convert_to_hsv_space(image_pil_color))
        self.haralick_button.config(state="disabled")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")
        
    def convert_to_histograms(self, image_pil_color):
        # Converta a imagem PIL para uma imagem RGB e depois para uma matriz numpy
        image_pil_rgb = image_pil_color.convert('RGB')
        image_np = np.array(image_pil_rgb)

        # Use o OpenCV para converter a imagem para tons de cinza
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Converta a imagem em tons de cinza de volta para o formato PIL
        image_pil_gray = Image.fromarray(image_gray)
        
        histogram = image_pil_gray.histogram()

        # Crie uma nova imagem PIL para o histograma
        hist_height = 256
        hist_image = Image.new('L', (512, hist_height), "white")
        draw = ImageDraw.Draw(hist_image)

        # Desenhe as barras do histograma na nova imagem
        highest_freq = max(histogram)
        for i, freq in enumerate(histogram):
            # Normalize a frequência para que caiba na altura da imagem
            bar_height = int((freq / highest_freq) * hist_height)
            draw.line([(i*2, hist_height - 10), (i*2, hist_height - 10 - bar_height)], fill="black")

                # Adicione rótulos ao eixo x
        labels = [0, 32, 64, 96, 128, 160, 192, 224, 255]
        for i, label in enumerate(labels):
            draw.text((label, hist_height - 10), str(label), fill="red")
        
        photo_histogram = ImageTk.PhotoImage(hist_image)
        
        # Limpa o canvas antes de adicionar a nova imagem
        self.canvas.delete("all")

        # Calcula as coordenadas para centralizar a imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo_histogram.width()
        image_height = photo_histogram.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Adiciona a imagem ao canvas centralizada
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo_histogram)
        self.canvas.image = photo_histogram
        
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        self.gray_scale_button.config(text="Gray Scale", state="active", command=lambda: self.convert_to_grayscale(image_pil_color))
        self.histograms_button.config(state="disabled")
        self.hsv_space_button.config(state="disabled")
        self.haralick_button.config(state="disabled")
        self.hu_invariants_button.config(state="disabled")
        self.classify_button.config(state="disabled")

    def convert_to_hsv_space(self, image_pill_color):
        print("não fiz ainda")
        
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
