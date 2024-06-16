# Built-in libraries
import io
import tkinter as tk
from tkinter import font as tkfont
import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from skimage.feature import graycomatrix, graycoprops
import torch
from torch import nn
from torchvision.transforms import ToTensor, Normalize, Compose
from efficientnet_pytorch import EfficientNet
from torchvision import models, transforms
import torch.nn as nn
import os
import joblib
from sklearn.preprocessing import StandardScaler

global zoom_factor

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNet, self).__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        efficient_net_output_dim = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Identity()  # Remove the final classification layer
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
        self.master.geometry("800x600")
        self.master.resizable(True, True)

        self.frame = ctk.CTkFrame(self.master)
        self.frame.pack(fill=ctk.BOTH, expand=True)

        self.canvas = ctk.CTkCanvas(self.frame)
        self.canvas.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
        
        self.image_frame = ctk.CTkFrame(self.canvas)
        self.canvas.create_window((0, 0), window=self.image_frame, anchor=ctk.NW)

        self.image_label = ctk.CTkLabel(self.image_frame)
        self.image_label.pack()
            
        self.scrollbarVertical = ctk.CTkScrollbar(self.frame, orientation=ctk.VERTICAL, command=self.canvas.yview)
        self.scrollbarVertical.pack(side=ctk.RIGHT, fill=ctk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbarVertical.set)

        self.scrollbarHorizontal = ctk.CTkScrollbar(self.master, orientation=ctk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbarHorizontal.pack(side=ctk.BOTTOM, fill=ctk.X)
        self.canvas.configure(xscrollcommand=self.scrollbarHorizontal.set)

        self.button_frame = ctk.CTkFrame(self.master)
        self.button_frame.pack()

        self.button_info = self.create_buttons()
        
        self.image_pil_base = None
        self.co_matrices = None
        self.current_matrix_index = 0
        
        # # Bind events for zoom buttons
        # self.canvas.bind("<MouseWheelUp>", self.zoom_plus)
        # self.canvas.bind("<MouseWheelDown>", self.zoom_minus)

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
            ("SVM-2", self.svmBinaryClassification),
            ("SVM-6", self.svmMulticlassClassification),
            ("Eficinet-2", self.eficinetBinaryClassification),
            ("Eficinet-6", self.eficinetMultiClassification)
        ]

        for text, command in button_info:
            state = "disabled" if text != "Open Image" else "active"
            btn = ctk.CTkButton(self.button_frame, text=text, command=command, width=100, height=50, state=state)
            btn.pack(side=ctk.LEFT)
            setattr(self, f"{text.replace(' ', '_').lower()}_button", btn)
            
        return button_info

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
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))    

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
            
            self.open_image_button.configure(text="Change Image")
            self.enable_buttons()

    def enable_buttons(self):
        for attr in dir(self):
            if attr.endswith("_button"):
                getattr(self, attr).configure(state="active")

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

    def update_image(self, cvt_color_code, convert_method, mode):
        if self.image_pil_base:
            image_pil_rgb = self.image_pil_base.convert('RGB')
            matrix_rgb = np.array(image_pil_rgb)
            matrix_converted = cvt_color_code == 'convert' and np.array(image_pil_rgb.convert(mode)) or cv2.cvtColor(matrix_rgb, cvt_color_code)
            image_pil_converted = Image.fromarray(matrix_converted)
            self.place_image(image_pil_converted)

    def convert_to_histogram_gray(self):
        # Load and process the image
        image = self.image_pil_base.convert('L')
        image_16 = image.point(lambda p: p // 16 * 16)
        histogram = image_16.histogram()
        histogram_16 = [sum(histogram[i:i + 16]) for i in range(0, 256, 16)]

        # Plot the histogram
        fig, ax = plt.subplots()
        ax.bar(range(16), histogram_16, width=1, edgecolor='black')

        # Set the x and y limits to start from zero
        ax.set_xlim(left=-1, right=16)
        ax.set_ylim(bottom=0)

        # Label the axes and set the title
        ax.set_xlabel('Shade of Gray (0-15)')
        ax.set_ylabel('Frequency')
        ax.set_title('Histogram of Grayscale Image with 16 Shades')

        # Set x-ticks to be from 0 to 15
        ax.set_xticks(range(16))
        ax.set_xticklabels(range(16))

        # Show the plot
        plt.show()
    
    def convert_to_histogram_hsv(self):
        # Convert the image to HSV
        image_hsv = self.image_pil_base.convert('HSV')
        image_hsv_np = np.array(image_hsv)

        # Separate the channels
        h_channel = image_hsv_np[:, :, 0]
        s_channel = image_hsv_np[:, :, 1]
        v_channel = image_hsv_np[:, :, 2]

        # Calculate histograms for each channel
        hist_h = np.histogram(h_channel, bins=16, range=(0, 256))[0]
        hist_s = np.histogram(s_channel, bins=16, range=(0, 256))[0]
        hist_v = np.histogram(v_channel, bins=16, range=(0, 256))[0]

        # Plot the histograms
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        axs[0].bar(range(16), hist_h, width=1, edgecolor='black')
        axs[0].set_xlim(left=-1, right=16)
        axs[0].set_ylim(bottom=0)
        axs[0].set_xlabel('Hue (0-15)')
        axs[0].set_ylabel('Frequency')
        axs[0].set_title('Histogram of Hue Channel')

        axs[1].bar(range(16), hist_s, width=1, edgecolor='black')
        axs[1].set_xlim(left=-1, right=16)
        axs[1].set_ylim(bottom=0)
        axs[1].set_xlabel('Saturation (0-15)')
        axs[1].set_ylabel('Frequency')
        axs[1].set_title('Histogram of Saturation Channel')

        axs[2].bar(range(16), hist_v, width=1, edgecolor='black')
        axs[2].set_xlim(left=-1, right=16)
        axs[2].set_ylim(bottom=0)
        axs[2].set_xlabel('Value (0-15)')
        axs[2].set_ylabel('Frequency')
        axs[2].set_title('Histogram of Value Channel')

        # Set x-ticks to be from 0 to 15 for all subplots
        for ax in axs:
            ax.set_xticks(range(16))
            ax.set_xticklabels(range(16))

        # Show the plot
        plt.tight_layout()
        plt.show()

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
            channel_labels = ["Gray", "Hue", "Saturation", "Value"]
            hu_moments = [cv2.HuMoments(cv2.moments(ch)).flatten() for ch in channels]
            
            hu_moments_str = ""
            for label, moments in zip(channel_labels, hu_moments):
                hu_moments_str += f"Hu Moments for {label} Channel:\n"
                hu_moments_str += "\n".join([f"Hu Moment {i+1}: {moment:.4e}" for i, moment in enumerate(moments)])
                hu_moments_str += "\n\n"
            
            # Create a Toplevel window
            top = tk.Toplevel()
            top.title("Hu Invariants")
            
            # Create a Text widget with a larger font
            text_widget = tk.Text(top, wrap='word', padx=10, pady=10)
            large_font = tkfont.Font(size=14)  # Adjust the size as needed
            text_widget.configure(font=large_font)
            
            # Center the text
            text_widget.tag_configure("center", justify='center')
            text_widget.insert('1.0', hu_moments_str)
            text_widget.tag_add("center", "1.0", "end")
            text_widget.pack(expand=True, fill='both')
            
            # Add a scrollbar
            scrollbar = tk.Scrollbar(top, command=text_widget.yview)
            text_widget['yscrollcommand'] = scrollbar.set
            scrollbar.pack(side='right', fill='y')

            # Add an OK button to close the window
            button = tk.Button(top, text="OK", command=top.destroy)
            button.pack()

    def svmBinaryClassification(self):
        model_path = os.path.join('models', 'best_svm_model_binary.joblib')
        model = joblib.load(model_path)

        # scaler = StandardScaler()
        array = np.array(self.image_pil_base)

        # Convert RGB to grayscale
        image_gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    
        # Reduce image to 16 gray levels
        image_gray //= 16
    
        # Define distances and angle for Haralick descriptors
        distances = [1, 2, 4, 8, 16, 31]
        angle = 0
    
        features = []
    
        # Compute GLCM and extract Haralick features
        for d in distances:
            glcm = graycomatrix(image_gray, distances=[d], angles=[angle], levels=16, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        
            # Collect the features
            features.extend([contrast, homogeneity, entropy])
    
        # Convert features list to numpy array and reshape for prediction
        features = np.array(features).reshape(1, -1)  # Reshape for a single sample
    
        # Apply scaling (assuming the scaler was trained during model training)    
        #features_scaled = scaler.transform(features)  # Uncomment if scaling is necessary
    
        # Predict the category of the image
        #y_pred = model.predict(features_scaled)
        # Predict the category of the image using both models
        y_pred = model.predict(features)

        # Display predictions using matplotlib
        fig, ax = plt.subplots(figsize=(6, 3))  # Adjust size as needed
        prediction_text = f'Binary Prediction: {y_pred[0]}'
        ax.text(0.5, 0.5, prediction_text, fontsize=15, ha='center', va='center')
        ax.axis('off')
    
        plt.tight_layout()
        plt.show()

    def svmMulticlassClassification(self):
        model_path = os.path.join('models', 'best_svm_model_6_catgoties.joblib')
        model = joblib.load(model_path)

        # scaler = StandardScaler()
        array = np.array(self.image_pil_base)

        # Convert RGB to grayscale
        image_gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    
        # Reduce image to 16 gray levels
        image_gray //= 16
    
        # Define distances and angle for Haralick descriptors
        distances = [1, 2, 4, 8, 16, 31]
        angle = 0
    
        features = []
    
        # Compute GLCM and extract Haralick features
        for d in distances:
            glcm = graycomatrix(image_gray, distances=[d], angles=[angle], levels=16, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            entropy = -np.sum(glcm * np.log2(glcm + np.finfo(float).eps))
        
            # Collect the features
            features.extend([contrast, homogeneity, entropy])
    
        # Convert features list to numpy array and reshape for prediction
        features = np.array(features).reshape(1, -1)  # Reshape for a single sample
    
        # Apply scaling (assuming the scaler was trained during model training)    
        #features_scaled = scaler.transform(features)  # Uncomment if scaling is necessary
    
        # Predict the category of the image
        #y_pred = model.predict(features_scaled)
        # Predict the category of the image using both models
        y_pred = model.predict(features)

        # Display predictions using matplotlib
        fig, ax = plt.subplots(figsize=(6, 3))  # Adjust size as needed
        prediction_text = f'Binary Prediction: {y_pred[0]}'
        ax.text(0.5, 0.5, prediction_text, fontsize=15, ha='center', va='center')
        ax.axis('off')
    
        plt.tight_layout()
        plt.show()

            
    def eficinetBinaryClassification(self):
        # model_path = 'D:\dev\pai\Reconhecimento-de-celulas-em-exames-de-Papanicolau\eficinet_model_binary_fine_tuned.pth'
        model_path = os.path.join('models', 'eficinet_model_binary_fine_tuned.pth')

        preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = self.image_pil_base.convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        
        # state_dict = torch.load('eficinet_model_binary_fine_tuned.pth')
        # model = CustomEfficientNet(num_classes=2)
        # model.load_state_dict(state_dict)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # input_tensor = input_tensor.to(device)


        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.fc = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = input_tensor.to(device)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
        class_labels = ["Normal", "Cancer"]
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_labels[predicted_idx.item()]
        messagebox.showinfo("Classification", f"Possible:{class_labels}\n" + f"\nPrediction: {predicted_label}")
            
    def eficinetMultiClassification(self):
        # model_path = 'D:\dev\pai\Reconhecimento-de-celulas-em-exames-de-Papanicolau\eficinet_model_6_categories_fine_tuned.pth'
        model_path = os.path.join('models', 'eficinet_model_6_categories_fine_tuned.pth')
        
        preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = self.image_pil_base.convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        
        # model = CustomEfficientNet(num_classes=6)
        # model.load_state_dict(torch.load('custom_eficinet_model_6_categories.pth'))
        # model.eval()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # input_tensor = input_tensor.to(device)

        model = models.efficientnet_b0()
        num_features = model.classifier[1].in_features
        model.fc = nn.Linear(num_features, 2)
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_tensor = input_tensor.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
        class_labels = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'Normal', 'SCC']
        _, predicted_idx = torch.max(output, 1)
        predicted_label = class_labels[predicted_idx.item()]
        messagebox.showinfo("Classification", f"Possible:{class_labels}\n" + f"\nPrediction: {predicted_label}")

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
