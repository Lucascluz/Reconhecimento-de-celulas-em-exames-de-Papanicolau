# Built-in libraries
import io
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# Third-party libraries
import cv2
import joblib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from PIL import Image, ImageTk
import skimage
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Compose
from efficientnet_pytorch import EfficientNet

global zoom_factor

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        
        # Get the output dimension of the EfficientNet feature extractor
        efficient_net_output_dim = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Identity()  # Remove the final classification layer

        # Define the combined fully connected layer
        self.fc_combined = nn.Linear(efficient_net_output_dim, num_classes)

    def forward(self, images):
        efficient_net_out = self.efficient_net(images)
        efficient_net_out = torch.flatten(efficient_net_out, 1)
        out = self.fc_combined(efficient_net_out)
        return out

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
        button_info = [
            ("Open Image", self.open_image),
            ("+", self.zoom_plus),
            ("-", self.zoom_minus),
            ("Gray Scale", self.convert_to_grayscale),
            ("Color (RGB)", self.revert_color),
            ("Gray Hist.", self.convert_to_histogram_gray),
            ("HSV Hist.", self.convert_to_histogram_hsv),
            ("Co-occurrence Matrices", self.co_occurrence_matrices),
            ("Haralick", self.get_haralick_descriptors),
            ("Hu Invariants", self.hu_invariants),
            ("Eficinet-2", self.eficinetBinaryClassification),
            ("Eficinet-6", self.eficinetMultiClassification)
        ]

        for text, command in button_info:
            state = "disabled" if text != "Open Image" else "normal"
            btn = tk.Button(self.button_frame, text=text, command=command, width=10, height=2, state=state)
            btn.pack(side=tk.LEFT)
            setattr(self, f"{text.replace(' ', '_').lower()}_button", btn)

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

        self.place_graph(photo)
            
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_pil_base = Image.fromarray(image_rgb)

            global zoom_factor
            zoom_factor = 1.0

            self.place_image(self.image_pil_base)
            
            self.open_image_button.config(text="Change Image")
            self.enable_buttons()

    def enable_buttons(self):
        for attr in dir(self):
            if attr.endswith("_button"):
                getattr(self, attr).config(state="active")

    def zoom_plus(self):
        global zoom_factor
        zoom_factor += 0.1
        self.place_image(self.image_pil_base)

    def zoom_minus(self):
        global zoom_factor
        zoom_factor -= 0.1
        self.place_image(self.image_pil_base)
            
    def convert_to_grayscale(self):
        self.update_image(cv2.COLOR_RGB2GRAY, "convert", "L")

    def revert_color(self):
        self.place_image(self.image_pil_base)
        self.update_button_states(gray=False)

    def update_image(self, cvt_color_code, convert_method, mode):
        if self.image_pil_base:
            image_pil_rgb = self.image_pil_base.convert('RGB')
            matrix_rgb = np.array(image_pil_rgb)
            matrix_converted = cvt_color_code == 'convert' and np.array(image_pil_rgb.convert(mode)) or cv2.cvtColor(matrix_rgb, cvt_color_code)
            image_pil_converted = Image.fromarray(matrix_converted)
            self.place_image(image_pil_converted)
            self.update_button_states(gray=True)

    def update_button_states(self, gray=False):
        state_gray = "disabled" if gray else "active"
        state_color = "active" if gray else "disabled"
        self.gray_scale_button.config(state=state_gray)
        self.colored_button.config(state=state_color)

    def convert_to_histogram_gray(self):
        self.show_histogram('L', "Histograma de Tons de Cinza", [0, 16])

    def convert_to_histogram_hsv(self):
        self.show_histogram('HSV', "Histograma HSV", [0, 256, 0, 256])

    def show_histogram(self, mode, title, hist_range):
        if self.image_pil_base:
            image = self.image_pil_base.convert(mode)
            array = np.array(image)
            hist = cv2.calcHist([array], [0], None, [16], hist_range)

            plt.figure(figsize=(12, 7))
            plt.title(title)
            plt.plot(hist)
            plt.xlim([0, 15])

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            tk_img = ImageTk.PhotoImage(img)
            self.place_graph(tk_img)

    def co_occurrence_matrices(self):
        if self.image_pil_base:
            gray_img = self.image_pil_base.convert('L')
            gray_img = gray_img.resize((256, 256))
            gray_array = np.array(gray_img)
            gray_array = (gray_array / 16).astype(int)

            distances = [1, 2, 4, 8, 16, 32]

            self.co_matrices = [graycomatrix(gray_array, [dist], [0], levels=16, symmetric=True, normed=True)[:, :, 0, 0] for dist in distances]
            self.show_matrix()

    def show_matrix(self):
        plt.clf()
        plt.imshow(self.co_matrices[self.current_matrix_index], cmap='gray')
        plt.title(f"Co-occurrence Matrix (Distance {self.current_matrix_index + 1})")
        plt.colorbar()
        plt.connect('key_press_event', self.key_press_event)
        plt.show()

    def key_press_event(self, event):
        if event.key in ['right', 'left']:
            self.current_matrix_index = (self.current_matrix_index + (1 if event.key == 'right' else -1)) % len(self.co_matrices)
            self.show_matrix()

    def haralick_descriptors(self, co_matrix):
        contrast = graycoprops(co_matrix, 'contrast')[0, 0]
        homogeneity = graycoprops(co_matrix, 'homogeneity')[0, 0]

        co_matrix_normed = co_matrix / np.sum(co_matrix)
        entropia = -np.sum(co_matrix_normed * np.log2(co_matrix_normed + (co_matrix_normed == 0)))

        return contrast, homogeneity, entropia

    def get_haralick_descriptors(self):
        if self.image_pil_base:
            gray_img = self.image_pil_base.convert('L').resize((256, 256))
            gray_array = np.array(gray_img)
            gray_array = (gray_array / 16).astype(int)

            distances = [1, 2, 4, 8, 16, 32]
            features = [self.haralick_descriptors(graycomatrix(gray_array, [dist], [0], levels=16, symmetric=True, normed=True)) for dist in distances]
            features_str = "\n\n".join([f"Distância {dist}:\n  Contraste: {f[0]:.4f}\n  Homogeneidade: {f[1]:.4f}\n  Entropia: {f[2]:.4f}" for dist, f in zip(distances, features)])
            
            messagebox.showinfo("Haralick Descriptors", features_str)
            
    def hu_invariants(self):
        if self.image_pil_base:
            gray_img = np.array(self.image_pil_base.convert('L'))
            hsv_img = cv2.cvtColor(np.array(self.image_pil_base), cv2.COLOR_RGB2HSV)
            channels = [gray_img, hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]]

            hu_moments = [cv2.HuMoments(cv2.moments(ch)).flatten() for ch in channels]
            hu_moments_str = "\n".join([f"Hu Moment {i+1} ({label}): {moment:.4e}" for label, moments in zip(["Gray", "Hue", "Saturation", "Value"], hu_moments) for i, moment in enumerate(moments)])
            
            messagebox.showinfo("Hu Invariants", hu_moments_str)
            
    def eficinetBinaryClassification(self):
        preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = self.image_pil_base.convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)

        model = CustomEfficientNet(num_classes=2)
        model.load_state_dict(torch.load('custom_eficinet_model_binary.pth'))  # Carregar o modelo treinado
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        class_labels = ["Normal", "Cancer"]
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_labels[predicted_idx.item()]

        messagebox.showinfo("Classification", f"Prediction: {predicted_label}")
            
    def eficinetMultiClassification(self):
        preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = self.image_pil_base.convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)

        model = CustomEfficientNet(num_classes=6)
        model.load_state_dict(torch.load('custom_eficinet_model_6_categories.pth'))  # Carregar o modelo treinado
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        class_labels = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'Normal', 'SCC']
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_labels[predicted_idx.item()]

        messagebox.showinfo("Classification", f"Prediction: {predicted_label}")

def main():
    root = tk.Tk()
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
