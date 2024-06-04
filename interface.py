# Built-in libraries
import io
import tkinter as tk
from tkinter import filedialog, ttk

# Third-party libraries
import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.widgets import Button
from PIL import Image, ImageTk
import skimage
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

global zoom_factor

class ImageViewerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")

        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=tk.NW)

        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
            
        self.scrollbarVertical = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbarVertical.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbarVertical.set)

        self.scrollbarHorizontal = ttk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbarHorizontal.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=self.scrollbarHorizontal.set)

        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack()

        self.create_buttons()
        self.image_pil_base = None
        self.co_matrices = None
        self.current_matrix_index = 0

    def create_buttons(self):
        self.open_button = tk.Button(self.button_frame, text="Open Image", command=self.open_image, width=10, height=2)
        self.open_button.pack(side=tk.LEFT)
        
        self.zoom_plus_button = tk.Button(self.button_frame, text="+", width=5, height=2, state="disabled", command=self.zoom_plus)
        self.zoom_plus_button.pack(side=tk.LEFT)
        
        self.zoom_minus_button = tk.Button(self.button_frame, text="-", width=5, height=2, state="disabled", command=self.zoom_minus)
        self.zoom_minus_button.pack(side=tk.LEFT)

        self.gray_scale_button = tk.Button(self.button_frame, text="Gray Scale", width=10, height=2, state="disabled", command=self.convert_to_grayscale)
        self.gray_scale_button.pack(side=tk.LEFT)
        
        self.colored_button = tk.Button(self.button_frame, text="Color (RGB)", width=10, height=2, state="disabled", command=self.revert_color)
        self.colored_button.pack(side=tk.LEFT)
        
        self.histograms_button = tk.Button(self.button_frame, text="Gray Hist.", width=10, height=2, state="disabled", command=self.convert_to_histogram_gray)
        self.histograms_button.pack(side=tk.LEFT)
        
        self.hsv_space_button = tk.Button(self.button_frame, text="HSV Hist.", width=10, height=2, state="disabled", command=self.convert_to_histogram_hsv)
        self.hsv_space_button.pack(side=tk.LEFT)
        
        self.co_button = tk.Button(self.button_frame, text="Co-occurrence Matrices",width=10, height=2, state="disabled", command=self.co_occurrence_matrices)
        self.co_button.pack()
        
        self.haralick_button = tk.Button(self.button_frame, text="Haralick", width=10, height=2, state="disabled", command=self.get_haralick_descriptors)
        self.haralick_button.pack(side=tk.LEFT)
        
        self.hu_invariants_button = tk.Button(self.button_frame, text="Hu Invariants", width=10, height=2, state="disabled", command=self.hu_invariants)
        self.hu_invariants_button.pack(side=tk.LEFT)
        
        self.classify_button = tk.Button(self.button_frame, text="Classify", width=10, height=2, state="disabled", command=self.classify)
        self.classify_button.pack(side=tk.LEFT)

    def place_graph(self, photo):
        self.canvas.delete("all")

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo.width()
        image_height = photo.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo)
        self.canvas.image = photo
        self.canvas.config(scrollregion=self.canvas.bbox("all"))    

    def place_image(self, image_pil):
        img_width = int(image_pil.width * zoom_factor)
        img_height = int(image_pil.height * zoom_factor)
        image_resized = image_pil.resize((img_width, img_height))
        photo = ImageTk.PhotoImage(image_resized)

        self.canvas.delete("all")

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = photo.width()
        image_height = photo.height()
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=photo)
        self.canvas.image = photo
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
            
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_pil_base = Image.fromarray(image_rgb)

            global zoom_factor
            zoom_factor = 1.0

            self.place_image(self.image_pil_base)
            
            self.open_button.config(text="Change Image")
            self.enable_buttons()

    def enable_buttons(self):
        self.zoom_plus_button.config(state="active")
        self.zoom_minus_button.config(state="active")
        self.gray_scale_button.config(state="active")
        self.colored_button.config(state="active")
        self.histograms_button.config(state="active")
        self.hsv_space_button.config(state="active")
        self.co_button.config(state="active")
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")

    def zoom_plus(self):
        global zoom_factor
        zoom_factor += 0.1
        self.place_image(self.image_pil_base)

    def zoom_minus(self):
        global zoom_factor
        zoom_factor -= 0.1
        self.place_image(self.image_pil_base)
            
    def convert_to_grayscale(self):
        if self.image_pil_base:
            image_pil_rgb = self.image_pil_base.convert('RGB')
            matrix_rgb = np.array(image_pil_rgb)
            matrix_gray = cv2.cvtColor(matrix_rgb, cv2.COLOR_RGB2GRAY)
            image_pil_gray = Image.fromarray(matrix_gray)
            self.place_image(image_pil_gray)
            self.update_button_states(gray=True)

    def revert_color(self):
        if self.image_pil_base:
            self.place_image(self.image_pil_base)
            self.update_button_states(gray=False)

    def update_button_states(self, gray=False):
        self.gray_scale_button.config(state="disabled" if gray else "active")
        self.colored_button.config(state="active" if gray else "disabled")
        self.histograms_button.config(state="active")
        self.hsv_space_button.config(state="active")
        self.haralick_button.config(state="active")
        self.hu_invariants_button.config(state="active")
        self.classify_button.config(state="active")
        self.zoom_plus_button.config(state="active")
        self.zoom_minus_button.config(state="active")

    def convert_to_histogram_gray(self):
        if self.image_pil_base:
            gray_img = self.image_pil_base.convert('L')
            gray_array = np.array(gray_img)
            image = cv2.convertScaleAbs(gray_array, alpha=(15/255))
            histogram = cv2.calcHist([image], [0], None, [16], [0, 16])

            plt.figure(figsize=(12, 7))
            plt.title("Histograma de Tons de Cinza")
            plt.xlabel("Tons de Cinza")
            plt.ylabel("Quantidade de Ocorrências")
            plt.plot(histogram)
            plt.xlim([0, 15])

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            tk_img = ImageTk.PhotoImage(img)
            self.place_graph(tk_img)
            self.update_button_states(histogram_gray=True)

    def convert_to_histogram_hsv(self):
        if self.image_pil_base:
            array = np.array(self.image_pil_base)
            hsv = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            hist = cv2.calcHist([v, h], [0, 1], None, [8, 8], [0, 256, 0, 256])

            plt.figure(figsize=(12, 7))
            plt.imshow(hist, interpolation="nearest")
            plt.title("Histograma HSV")
            plt.xlabel("Hue")
            plt.ylabel("Saturation")

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            tk_img = ImageTk.PhotoImage(img)
            self.place_graph(tk_img)
            self.update_button_states(histogram_hsv=True)
            
    def co_occurrence_matrices(self):
        if self.image_pil_base:
            # Converter a imagem para 16 tons de cinza
            gray_img = self.image_pil_base.convert('L')
            gray_img = gray_img.resize((256, 256))  # Redimensiona para um tamanho fixo, se necessário
            gray_array = np.array(gray_img)
            gray_array = (gray_array / 16).astype(int)  # Converter para 16 tons de cinza

            # Definir distâncias
            distances = [1, 2, 4, 8, 16, 32]

            # Calcular as matrizes de co-ocorrência
            self.co_matrices = []
            for dist in distances:
                co_matrix = skimage.feature.graycomatrix(gray_array, [dist], [0], levels=16, symmetric=True, normed=True)
                self.co_matrices.append(co_matrix[:, :, 0, 0])

            # Mostrar a primeira matriz
            self.show_matrix()

    def show_matrix(self):
        plt.clf()
        plt.imshow(self.co_matrices[self.current_matrix_index], cmap='gray')
        plt.title(f"Co-occurrence Matrix (Distance {self.current_matrix_index + 1})")
        plt.colorbar()
        plt.connect('key_press_event', self.key_press_event)
        plt.show()

    def key_press_event(self, event):
        if event.key == 'right':
            self.current_matrix_index = (self.current_matrix_index + 1) % len(self.co_matrices)
            self.show_matrix()
        elif event.key == 'left':
            self.current_matrix_index = (self.current_matrix_index - 1) % len(self.co_matrices)
            self.show_matrix()
        
    #########################################################################  
    def haralick_descriptors(self, co_matrix):
        # Calcular os descritores de Haralick
        contrast = graycoprops(co_matrix, 'contrast')[0, 0]
        homogeneity = graycoprops(co_matrix, 'homogeneity')[0, 0]

        # Calcular a entropia
        co_matrix_normed = co_matrix / np.sum(co_matrix)
        entropia = -np.sum(co_matrix_normed * np.log2(co_matrix_normed + (co_matrix_normed == 0)))

        return contrast, homogeneity, entropia

    def get_haralick_descriptors(self):
        if self.image_pil_base:
            # Converter a imagem para 16 tons de cinza
            gray_img = self.image_pil_base.convert('L')
            gray_img = gray_img.resize((256, 256))  # Redimensiona para um tamanho fixo, se necessário
            gray_array = np.array(gray_img)
            gray_array = (gray_array / 16).astype(int)  # Converter para 16 tons de cinza

            # Definir distâncias
            distances = [1, 2, 4, 8, 16, 32]

            # Calcular as matrizes de co-ocorrência e os descritores de Haralick
            haralick_features_str = ""
            for dist in distances:
                co_matrix = graycomatrix(gray_array, [dist], [0], levels=16, symmetric=True, normed=True)
                contrast, homogeneity, entropia = self.haralick_descriptors(co_matrix)
                haralick_features_str += (f"Distância {dist}:\n"f"  Contraste: {contrast:.4f}\n"f"  Homogeneidade: {homogeneity:.4f}\n"f"  Entropia: {entropia:.4f}\n\n")

            # Exibir os descritores
            tk.messagebox.showinfo("Haralick Descriptors", haralick_features_str)
            
    def hu_invariants(self):
        if self.image_pil_base:
            # Converter a imagem para tons de cinza
            gray_img = self.image_pil_base.convert('L')
            gray_array = np.array(gray_img)
            hu_moments_gray = cv2.HuMoments(cv2.moments(gray_array)).flatten()

            # Converter a imagem para o modelo HSV
            hsv_img = cv2.cvtColor(np.array(self.image_pil_base), cv2.COLOR_RGB2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv_img)

            hu_moments_hue = cv2.HuMoments(cv2.moments(h_channel)).flatten()
            hu_moments_saturation = cv2.HuMoments(cv2.moments(s_channel)).flatten()
            hu_moments_value = cv2.HuMoments(cv2.moments(v_channel)).flatten()

            # Preparar a mensagem com todos os momentos de Hu
            hu_moments_str = "\n".join(
                [f"Hu Moment {i+1} (Gray): {moment:.4e}" for i, moment in enumerate(hu_moments_gray)] +
                [f"Hu Moment {i+1} (Hue): {moment:.4e}" for i, moment in enumerate(hu_moments_hue)] +
                [f"Hu Moment {i+1} (Saturation): {moment:.4e}" for i, moment in enumerate(hu_moments_saturation)] +
                [f"Hu Moment {i+1} (Value): {moment:.4e}" for i, moment in enumerate(hu_moments_value)]
            )

            tk.messagebox.showinfo("Hu Invariants", hu_moments_str)
            
    def classify(self):
        if self.image_pil_base:
            image = self.image_pil_base.convert('L')
            image_array = np.array(image).reshape(-1, 1)
            scaler = StandardScaler()
            scaled_image = scaler.fit_transform(image_array)

            model = joblib.load("mlp_model.joblib")
            prediction = model.predict(scaled_image.T)
            class_prediction = "Benign" if prediction == 0 else "Malignant"

            tk.messagebox.showinfo("Classification", f"Prediction: {class_prediction}")

def main():
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
