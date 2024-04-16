import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

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

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        file_path = "InterfaceBackground.png"
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
        
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

        # Organizando os botões adicionais lado a lado
        
        # Botão para converter a imagem para tons de cinza
        self.gray_scale_button = tk.Button(self.button_frame, text="Gray Scale", width=20, height=2, state="disabled")
        self.gray_scale_button.pack(side=tk.LEFT)
        
        self.histograms_button = tk.Button(self.button_frame, text="Histograms", width=20, height=2, state = "disabled")
        self.histograms_button.pack(side=tk.LEFT)
        
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
            image_pil = Image.fromarray(image_rgb)
            photo = ImageTk.PhotoImage(image_pil)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.open_button.config(text="Change Image")
            self.gray_scale_button.config(state="active", command=self.convert_to_grayscale)
            self.histograms_button.config(state="active")
            self.haralick_button.config(state="active")
            self.hu_invariants_button.config(state="active")
            self.classify_button.config(state="active")
            
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
            
    def convert_to_grayscale(self):
        # Obtém a imagem atualmente exibida na label
        current_photo = self.image_label.image
        print("pegou a imagem deu bom")

        if current_photo:
            print("pegou a imagem deu bom e não é nula")
            # Converte a imagem para o formato PIL
            pil_image = current_photo.pilimage
            # Converte a imagem para o formato OpenCV (BGR)
            bgr_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
            # Converte a imagem para tons de cinza usando OpenCV
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            # Converte a imagem para o formato PIL (tamanho invertido porque o OpenCV usa altura x largura)
            gray_pil_image = Image.fromarray(gray_image)
            # Converte a imagem de volta para o formato ImageTk
            photo = ImageTk.PhotoImage(image=gray_pil_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

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
