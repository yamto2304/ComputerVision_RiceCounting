import cv2
import numpy as np
from skimage import morphology, measure
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Top-hat để làm nổi bật hạt gạo trên nền tối
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blurred = cv2.GaussianBlur(tophat, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    return enhanced

def segment_otsu(img):
    # Sử dụng ngưỡng Otsu không đảo ngược
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def segment_adaptive(img):
    # Adaptive threshold cho nền sáng tối không đều
    binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 2)
    return binary

def postprocess(binary):
    kernel = np.ones((3,3), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opened, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    img_color = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)
    result = np.zeros_like(opened)
    result[markers > 1] = 255
    return result

def count_rice_grains(binary, min_area=50):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    areas = [cv2.contourArea(cnt) for cnt in valid_contours]
    avg_area = np.mean(areas) if areas else 0
    max_area = np.max(areas) if areas else 0
    return len(valid_contours), avg_area, max_area, valid_contours

def draw_output(cleaned, contours):
    result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours):
        cv2.drawContours(result, [cnt], 0, (0, 255, 0), 2)
        cv2.putText(result, str(i+1), tuple(cnt[cnt[:,:,1].argmin()][0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    return result

def cv2_to_tk(img_cv2, size=(300,300)):
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)
    img_pil = img_pil.resize(size)
    return ImageTk.PhotoImage(img_pil)

class RiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rice Grain Counter")
        self.method = 'otsu'
        self.min_area = 50

        # Frame input
        self.frame_input = tk.LabelFrame(root, text="Input Image")
        self.frame_input.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_input = tk.Label(self.frame_input)
        self.canvas_input.pack()

        # Frame outputs với canvas + scrollbar
        self.frame_outputs = tk.LabelFrame(root, text="Processing Steps")
        self.frame_outputs.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.canvas_outputs = tk.Canvas(self.frame_outputs, width=950, height=320)
        self.scrollbar_outputs = tk.Scrollbar(self.frame_outputs, orient="horizontal", command=self.canvas_outputs.xview)
        self.canvas_outputs.configure(xscrollcommand=self.scrollbar_outputs.set)
        self.canvas_outputs.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scrollbar_outputs.pack(side=tk.BOTTOM, fill=tk.X)
        self.outputs_inner = tk.Frame(self.canvas_outputs)
        self.canvas_outputs.create_window((0,0), window=self.outputs_inner, anchor="nw")
        self.outputs_inner.bind("<Configure>", lambda e: self.canvas_outputs.configure(scrollregion=self.canvas_outputs.bbox("all")))
        self.output_labels = []

        # Button
        self.btn_add = tk.Button(root, text="Thêm ảnh", command=self.load_image)
        self.btn_add.grid(row=1, column=0, columnspan=2, pady=10)

        # Info label
        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.grid(row=2, column=0, columnspan=2)

        self.img = None

    def clear_outputs(self):
        for lbl in self.output_labels:
            lbl.destroy()
        self.output_labels = []

    def add_output(self, img, title):
        tk_img = cv2_to_tk(img)
        lbl = tk.Label(self.outputs_inner, text=title, compound=tk.TOP)
        lbl.config(image=tk_img)
        lbl.image = tk_img
        lbl.pack(side=tk.LEFT, padx=5)
        self.output_labels.append(lbl)

    def load_image(self):
        path = filedialog.askopenfilename(title="Chọn ảnh gạo", filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            self.info_label.config(text="Không đọc được ảnh!")
            return
        self.img = img
        self.show_input(img)
        self.process(img)

    def show_input(self, img):
        tk_img = cv2_to_tk(img)
        self.canvas_input.config(image=tk_img)
        self.canvas_input.image = tk_img

    def process(self, img):
        self.clear_outputs()
        # 1. Ảnh gốc
        self.add_output(img, "Ảnh gốc")

        # 2. Ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.add_output(gray_bgr, "Ảnh xám")

        # 3. Làm nét
        enhanced = preprocess_image(img)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        self.add_output(enhanced_bgr, "Làm nét")

        # 4. Nhị phân
        # binary = segment_otsu(enhanced)
        binary = segment_adaptive(enhanced)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.add_output(binary_bgr, "Nhị phân")

        # 5. Làm sạch & tách hạt
        cleaned = postprocess(binary)
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        self.add_output(cleaned_bgr, "Làm sạch & tách hạt")

        # 6. Kết quả cuối
        count, avg_area, max_area, contours = count_rice_grains(cleaned, self.min_area)
        result = draw_output(cleaned, contours)
        self.add_output(result, "Kết quả cuối")
        self.info_label.config(text=f"Số hạt: {count} | Diện tích TB: {avg_area:.2f} px | Diện tích lớn nhất: {max_area:.2f} px")

if __name__ == "__main__":
    root = tk.Tk()
    app = RiceApp(root)
    root.mainloop()