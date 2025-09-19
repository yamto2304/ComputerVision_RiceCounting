import cv2
import numpy as np
from skimage import morphology, measure
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from config import PreprocessingConfig, SegmentationConfig, PostprocessingConfig, CountingConfig, GeneralConfig

def analyze_image_brightness(img):
    """Phân tích độ sáng của ảnh để xác định có cần xử lý đặc biệt không"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tính các thống kê histogram
    mean_brightness = np.mean(gray)
    median_brightness = np.median(gray)
    std_brightness = np.std(gray)
    
    # Phân tích histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Tính tỷ lệ pixel tối (< 50) và pixel sáng (> 200)
    dark_pixels = np.sum(hist[:50])
    bright_pixels = np.sum(hist[200:])
    total_pixels = gray.size
    
    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels
    
    # Đánh giá độ sáng sử dụng tham số từ config
    is_dark = False
    brightness_level = "normal"
    thresholds = PreprocessingConfig.BRIGHTNESS_THRESHOLDS
    
    if mean_brightness < thresholds['very_dark']:
        is_dark = True
        brightness_level = "very_dark"
    elif mean_brightness < thresholds['dark']:
        is_dark = True
        brightness_level = "dark"
    elif mean_brightness > thresholds['very_bright']:
        brightness_level = "very_bright"
    elif mean_brightness > thresholds['bright']:
        brightness_level = "bright"
    
    return {
        'is_dark': is_dark,
        'brightness_level': brightness_level,
        'mean_brightness': mean_brightness,
        'median_brightness': median_brightness,
        'std_brightness': std_brightness,
        'dark_ratio': dark_ratio,
        'bright_ratio': bright_ratio
    }

def preprocess_image(img):
    # Phân tích độ sáng của ảnh
    brightness_info = analyze_image_brightness(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Điều chỉnh tham số CLAHE dựa trên độ sáng
    if brightness_info['is_dark']:
        clahe_clip_limit = PreprocessingConfig.CLAHE_CLIP_LIMIT * PreprocessingConfig.CLAHE_DARK_MULTIPLIER
        if brightness_info['brightness_level'] == "very_dark":
            clahe_clip_limit = PreprocessingConfig.CLAHE_CLIP_LIMIT * PreprocessingConfig.CLAHE_VERY_DARK_MULTIPLIER
    else:
        clahe_clip_limit = PreprocessingConfig.CLAHE_CLIP_LIMIT
    
    # Thử nhiều kích thước kernel top-hat để tìm kích thước phù hợp nhất
    kernels = PreprocessingConfig.TOPHAT_KERNEL_SIZES
    
    best_enhanced = None
    best_score = 0
    
    for kernel_size in kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blurred = cv2.GaussianBlur(tophat, PreprocessingConfig.GAUSSIAN_KERNEL_SIZE, PreprocessingConfig.GAUSSIAN_SIGMA)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=PreprocessingConfig.CLAHE_TILE_GRID_SIZE)
        enhanced = clahe.apply(blurred)
        
        # Đánh giá chất lượng enhancement bằng cách tính độ tương phản
        score = np.std(enhanced)
        if score > best_score:
            best_score = score
            best_enhanced = enhanced
    
    # Nếu vẫn không tốt, thử phương pháp khác: gamma correction + CLAHE
    if best_score < 30:  # Nếu độ tương phản vẫn thấp
        # Gamma correction để làm sáng ảnh
        gamma = 0.5  # Giá trị < 1 để làm sáng
        gamma_corrected = np.power(gray / 255.0, gamma) * 255.0
        gamma_corrected = np.uint8(gamma_corrected)
        
        # Áp dụng CLAHE mạnh hơn
        clahe_strong = cv2.createCLAHE(clipLimit=clahe_clip_limit * 1.5, tileGridSize=(4, 4))
        best_enhanced = clahe_strong.apply(gamma_corrected)
    
    return best_enhanced, brightness_info

def segment_otsu(img):
    # Sử dụng ngưỡng Otsu không đảo ngược
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def segment_adaptive(img):
    # Thử nhiều tham số adaptive threshold
    block_sizes = SegmentationConfig.ADAPTIVE_BLOCK_SIZES
    c_constants = SegmentationConfig.ADAPTIVE_C_CONSTANTS
    
    best_binary = None
    best_score = 0
    
    for block_size in block_sizes:
        for c_constant in c_constants:
            binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block_size, c_constant)
            
            # Đánh giá chất lượng bằng cách đếm số vùng foreground hợp lý
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 20]
            
            # Score dựa trên số lượng vùng và độ phân tán
            if len(valid_contours) > 0:
                areas = [cv2.contourArea(cnt) for cnt in valid_contours]
                score = len(valid_contours) * np.std(areas)  # Ưu tiên nhiều vùng và đa dạng kích thước
                
                if score > best_score:
                    best_score = score
                    best_binary = binary
    
    return best_binary if best_binary is not None else cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                              cv2.THRESH_BINARY, SegmentationConfig.ADAPTIVE_BLOCK_SIZE, SegmentationConfig.ADAPTIVE_C_CONSTANT)

def postprocess(binary):
    # Tạo một bản sao để xử lý các hạt ở biên
    binary_expanded = binary.copy()
    
    # Mở rộng vùng quan tâm bằng cách thêm border
    border_size = PostprocessingConfig.BORDER_SIZE
    h, w = binary.shape
    expanded = np.zeros((h + 2*border_size, w + 2*border_size), dtype=np.uint8)
    expanded[border_size:border_size+h, border_size:border_size+w] = binary
    
    # Áp dụng morphology trên ảnh mở rộng
    kernel = np.ones(PostprocessingConfig.MORPH_KERNEL_SIZE, np.uint8)
    opened = cv2.morphologyEx(expanded, cv2.MORPH_OPEN, kernel, iterations=PostprocessingConfig.MORPH_ITERATIONS)
    
    # Cắt lại về kích thước ban đầu
    opened = opened[border_size:border_size+h, border_size:border_size+w]
    
    # Áp dụng distance transform và watershed
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, PostprocessingConfig.DISTANCE_MASK_SIZE)
    ret, sure_fg = cv2.threshold(dist, PostprocessingConfig.WATERSHED_THRESHOLD_RATIO*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(opened, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    img_color = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)
    result = np.zeros_like(opened)
    result[markers > 1] = 255
    
    # Xử lý đặc biệt cho các hạt ở biên
    # Tìm các contour gần biên và cố gắng khôi phục chúng
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Kiểm tra xem contour có gần biên không
        x, y, w, h = cv2.boundingRect(contour)
        margin = 5  # Khoảng cách từ biên
        
        is_near_edge = (x < margin or y < margin or 
                       x + w > result.shape[1] - margin or 
                       y + h > result.shape[0] - margin)
        
        if is_near_edge:
            # Nếu gần biên, kiểm tra trong ảnh gốc xem có hạt không
            # Tìm vùng tương ứng trong ảnh gốc
            roi = binary[max(0, y-margin):min(result.shape[0], y+h+margin),
                        max(0, x-margin):min(result.shape[1], x+w+margin)]
            
            if np.sum(roi) > 0:  # Có pixel trắng trong vùng này
                # Khôi phục hạt này trong kết quả
                cv2.fillPoly(result, [contour], 255)
    
    return result

def count_rice_grains(binary, min_area=None):
    if min_area is None:
        min_area = CountingConfig.MIN_AREA
    contours, _ = cv2.findContours(binary, getattr(cv2, f"RETR_{CountingConfig.CONTOUR_RETRIEVAL_MODE}"), getattr(cv2, f"CHAIN_APPROX_{CountingConfig.CONTOUR_APPROX_METHOD}"))
    
    # Tính toán thống kê diện tích để xác định ngưỡng động
    areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]
    
    valid_contours = []
    
    if len(areas) > 0:
        areas = np.array(areas)
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        median_area = np.median(areas)
        
        # Ngưỡng động dựa trên thống kê - cân bằng giữa độ chính xác và số lượng
        # Cho phép các hạt nhỏ hơn nếu chúng có tỷ lệ hợp lý
        dynamic_min_area = min(min_area, median_area * CountingConfig.DYNAMIC_AREA_RATIO)
        
        # Đặc biệt xử lý các hạt ở biên (diện tích nhỏ nhưng có thể là hạt thật)
        edge_threshold = median_area * CountingConfig.EDGE_THRESHOLD_RATIO
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Kiểm tra các tiêu chí khác nhau
            is_valid_grain = False
            
            if area >= dynamic_min_area:
                # Hạt có diện tích đủ lớn
                is_valid_grain = True
            elif area >= edge_threshold:
                # Hạt nhỏ - kiểm tra thêm các đặc điểm hình học
                # Tính tỷ lệ khung hình (aspect ratio)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                
                # Tính độ tròn (circularity)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Tính độ compact (compactness)
                convex_hull_area = cv2.contourArea(cv2.convexHull(contour))
                compactness = area / convex_hull_area if convex_hull_area > 0 else 0
                
                # Các điều kiện cho hạt ở biên - chặt chẽ hơn để tránh noise:
                # 1. Tỷ lệ khung hình không quá dài (không phải noise dài)
                # 2. Độ tròn hợp lý (hạt gạo thường không quá méo)
                # 3. Độ compact hợp lý (không quá rỗng)
                if (aspect_ratio < CountingConfig.EDGE_ASPECT_RATIO_MAX and
                    circularity > CountingConfig.EDGE_CIRCULARITY_MIN and
                    compactness > CountingConfig.EDGE_COMPACTNESS_MIN):
                    is_valid_grain = True
            
            if is_valid_grain:
                valid_contours.append(contour)
    else:
        # Nếu không có contour nào, sử dụng ngưỡng mặc định
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
    
    h, w = img_cv2.shape[:2]
    
    # Nếu size được chỉ định là kích thước thực tế (không phải max_size)
    if len(size) == 2 and size[0] == w and size[1] == h:
        # Không resize, giữ nguyên kích thước
        pass
    else:
        # Giữ tỷ lệ khung hình khi resize
        max_size = max(size)
        
        # Tính tỷ lệ scale để fit vào kích thước mong muốn
        scale = min(max_size/w, max_size/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize với tỷ lệ khung hình được giữ nguyên
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return ImageTk.PhotoImage(img_pil)

class RiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rice Grain Counter")
        self.method = GeneralConfig.DEFAULT_SEGMENTATION_METHOD
        self.min_area = CountingConfig.MIN_AREA

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

    def add_output(self, img, title, size=(300, 300)):
        tk_img = cv2_to_tk(img, size)
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
        # Hiển thị ảnh gốc với kích thước lớn hơn để dễ nhìn thấy các hạt nhỏ
        # Giới hạn kích thước tối đa để không quá lớn
        h, w = img.shape[:2]
        max_size = 500  # Kích thước tối đa
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            tk_img = cv2_to_tk(img, size=(new_w, new_h))
        else:
            # Không resize nếu ảnh đã nhỏ hơn max_size
            tk_img = cv2_to_tk(img, size=(w, h))
        
        self.canvas_input.config(image=tk_img)
        self.canvas_input.image = tk_img

    def process(self, img):
        """
        Hàm xử lý ảnh chính - thực hiện pipeline hoàn chỉnh để đếm hạt gạo
        Pipeline bao gồm 6 bước chính với các phép biến đổi toán học cụ thể
        """
        self.clear_outputs()
        
        # ========================================
        # BƯỚC 1: HIỂN THỊ ẢNH GỐC
        # ========================================
        # Mục đích: Hiển thị ảnh đầu vào để người dùng có thể so sánh với kết quả
        # Không có phép biến đổi toán học, chỉ là hiển thị trực quan
        self.add_output(img, "Ảnh gốc", size=(400, 400))

        # ========================================
        # BƯỚC 2: CHUYỂN ĐỔI MÀU SẮC (COLOR SPACE CONVERSION)
        # ========================================
        # Phép biến đổi: BGR → Grayscale
        # Công thức toán học: Gray = 0.299*R + 0.587*G + 0.114*B (Weighted Average)
        # Mục đích: Giảm chiều dữ liệu từ 3D (B,G,R) xuống 1D (Gray)
        # Lý do: Các thuật toán xử lý ảnh thường hoạt động tốt hơn trên ảnh xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Chuyển lại để hiển thị
        self.add_output(gray_bgr, "Ảnh xám")

        # ========================================
        # BƯỚC 3: TIỀN XỬ LÝ VÀ TĂNG CƯỜNG ẢNH (IMAGE ENHANCEMENT)
        # ========================================
        # Các phép biến đổi toán học được thực hiện trong preprocess_image():
        # 
        # 3.1. Top-hat Morphology:
        #    - Top-hat = Original - Opening(Original)
        #    - Opening = Dilation(Erosion(Original))
        #    - Mục đích: Tách các đối tượng nhỏ, sáng khỏi nền tối
        #    - Kernel size: (15,15) - kích thước cấu trúc để phát hiện hạt gạo
        #
        # 3.2. Gaussian Blur:
        #    - Công thức: G(x,y) = (1/(2πσ²)) * e^(-(x²+y²)/(2σ²))
        #    - Kernel size: (3,3), σ = 0 (tự động tính)
        #    - Mục đích: Làm mịn nhiễu, chuẩn bị cho bước phân đoạn
        #
        # 3.3. CLAHE (Contrast Limited Adaptive Histogram Equalization):
        #    - Chia ảnh thành các tile nhỏ (8x8)
        #    - Áp dụng histogram equalization cho mỗi tile
        #    - Clip limit: 2.0 (giới hạn độ tương phản để tránh nhiễu)
        #    - Mục đích: Tăng cường độ tương phản cục bộ
        #
        # 3.4. Dynamic Parameter Adjustment:
        #    - Phân tích độ sáng: mean, median, std của pixel values
        #    - Điều chỉnh CLAHE clip limit dựa trên brightness level
        #    - Dark images: clip_limit *= 2.0, Very dark: *= 3.0
        enhanced, brightness_info = preprocess_image(img)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # Hiển thị thông tin phân tích độ sáng
        brightness_text = f"Độ sáng: {brightness_info['brightness_level']} (TB: {brightness_info['mean_brightness']:.1f})"
        self.add_output(enhanced_bgr, f"Làm nét - {brightness_text}")

        # ========================================
        # BƯỚC 4: PHÂN ĐOẠN ẢNH (IMAGE SEGMENTATION)
        # ========================================
        # Thử cả hai phương pháp và chọn phương pháp tốt nhất:
        
        # 4.1. Otsu Thresholding:
        #    - Thuật toán: Tìm threshold tối ưu để minimize intra-class variance
        #    - Công thức: σ²w(t) = w₀(t)σ₀²(t) + w₁(t)σ₁²(t)
        #    - Tìm t* sao cho: t* = argmin(σ²w(t))
        #    - Ưu điểm: Tự động, không cần tham số
        binary_otsu = segment_otsu(enhanced)
        
        # 4.2. Adaptive Thresholding:
        #    - Thuật toán: Threshold cục bộ cho từng vùng
        #    - Công thức: T(x,y) = mean(neighborhood) - C
        #    - Block size: 21x21 (kích thước neighborhood)
        #    - C constant: 2 (giá trị trừ đi từ mean)
        #    - Ưu điểm: Xử lý tốt ảnh có độ sáng không đều
        binary_adaptive = segment_adaptive(enhanced)
        
        # 4.3. So sánh và chọn phương pháp tốt nhất:
        #    - Tìm contours từ cả hai phương pháp
        #    - Lọc contours có diện tích >= 20 pixels
        #    - Chọn phương pháp có nhiều valid contours hơn
        contours_otsu, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_adaptive, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_otsu = [cnt for cnt in contours_otsu if cv2.contourArea(cnt) >= 20]
        valid_adaptive = [cnt for cnt in contours_adaptive if cv2.contourArea(cnt) >= 20]
        
        # Chọn phương pháp có nhiều vùng hợp lý hơn
        if len(valid_adaptive) > len(valid_otsu):
            binary = binary_adaptive
            method_used = "Adaptive"
        else:
            binary = binary_otsu
            method_used = "Otsu"
        
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.add_output(binary_bgr, f"Nhị phân ({method_used})")

        # ========================================
        # BƯỚC 5: HẬU XỬ LÝ VÀ TÁCH HẠT (POST-PROCESSING & SEPARATION)
        # ========================================
        # Các phép biến đổi toán học trong postprocess():
        #
        # 5.1. Border Expansion:
        #    - Thêm border 10 pixels xung quanh ảnh
        #    - Mục đích: Xử lý các hạt ở biên ảnh
        #
        # 5.2. Morphological Opening:
        #    - Opening = Dilation(Erosion(binary))
        #    - Kernel: (3,3), iterations: 1
        #    - Mục đích: Loại bỏ nhiễu nhỏ, tách các hạt dính nhau
        #
        # 5.3. Distance Transform:
        #    - Tính khoảng cách từ mỗi pixel đến biên gần nhất
        #    - Công thức: DT(x,y) = min{√((x-i)² + (y-j)²) | (i,j) ∈ background}
        #    - Mục đích: Tìm các điểm "trung tâm" của hạt
        #
        # 5.4. Watershed Algorithm:
        #    - Sử dụng distance transform làm marker
        #    - Threshold ratio: 0.3 (30% của max distance)
        #    - Thuật toán: Flood fill từ các marker để tách các hạt
        #
        # 5.5. Edge Grain Restoration:
        #    - Phân tích geometric properties: aspect ratio, circularity, compactness
        #    - Aspect ratio = max(width, height) / min(width, height)
        #    - Circularity = 4π*Area / Perimeter²
        #    - Compactness = Area / (π * (Perimeter/(2π))²)
        #    - Khôi phục các hạt ở biên có properties hợp lý
        cleaned = postprocess(binary)
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        self.add_output(cleaned_bgr, "Làm sạch & tách hạt")

        # ========================================
        # BƯỚC 6: ĐẾM VÀ HIỂN THỊ KẾT QUẢ (COUNTING & VISUALIZATION)
        # ========================================
        # 6.1. Contour Analysis:
        #    - Tìm tất cả contours từ ảnh đã xử lý
        #    - Tính diện tích: Area = ∫∫ dx dy (tích phân kép)
        #    - Lọc contours dựa trên diện tích tối thiểu
        #
        # 6.2. Dynamic Thresholding:
        #    - Tính median area của tất cả contours
        #    - Dynamic min_area = median_area * 0.25 (25% của median)
        #    - Edge threshold = median_area * 0.12 (12% của median)
        #    - Mục đích: Thích ứng với kích thước hạt khác nhau
        #
        # 6.3. Geometric Validation:
        #    - Kiểm tra aspect ratio ≤ 2.5 (không quá dài)
        #    - Kiểm tra circularity ≥ 0.4 (gần hình tròn)
        #    - Kiểm tra compactness ≥ 0.5 (độ đặc chắc)
        #
        # 6.4. Visualization:
        #    - Vẽ contours lên ảnh gốc
        #    - Hiển thị số thứ tự cho mỗi hạt
        #    - Tính toán thống kê: count, avg_area, max_area
        count, avg_area, max_area, contours = count_rice_grains(cleaned)
        result = draw_output(cleaned, contours)
        self.add_output(result, "Kết quả cuối")
        
        # ========================================
        # BƯỚC 7: PHÂN TÍCH VÀ HIỂN THỊ DEBUG (DEBUG ANALYSIS & VISUALIZATION)
        # ========================================
        # Mục đích: Phân tích chi tiết kết quả xử lý để debug và tối ưu hóa thuật toán
        # KHÔNG phải chỉ hiển thị UI - đây là bước phân tích toán học quan trọng
        #
        # 7.1. Contour Detection và Area Analysis:
        #    - Tìm TẤT CẢ contours từ ảnh đã xử lý (không lọc theo min_area)
        #    - Tính diện tích của từng contour: Area = ∫∫ dx dy
        #    - Mục đích: Hiểu được phân bố kích thước của các đối tượng trong ảnh
        #
        # 7.2. Statistical Analysis:
        #    - Tính median area: giá trị trung vị của tất cả diện tích
        #    - Median = giá trị ở giữa khi sắp xếp tăng dần
        #    - Mục đích: Tìm kích thước "điển hình" của hạt gạo trong ảnh này
        #
        # 7.3. Low-threshold Visualization:
        #    - Hiển thị contours có diện tích >= 10% của median area
        #    - Threshold = median_area * 0.1 (rất thấp để thấy các hạt nhỏ)
        #    - Mục đích: Kiểm tra xem có hạt nhỏ nào bị bỏ sót không
        #    - Giúp debug: Nếu thấy nhiều hạt nhỏ trong debug nhưng ít trong kết quả cuối
        #      → có thể min_area quá cao hoặc thuật toán filtering quá strict
        all_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_areas = [cv2.contourArea(cnt) for cnt in all_contours if cv2.contourArea(cnt) > 0]
        
        if len(all_areas) > 0:
            median_area = np.median(all_areas)
            # Hiển thị tất cả contour có diện tích >= 10% median
            debug_contours = [cnt for cnt in all_contours if cv2.contourArea(cnt) >= median_area * 0.1]
            debug_result = draw_output(cleaned, debug_contours)
            self.add_output(debug_result, f"Debug: Tất cả hạt (≥10% median={median_area:.1f})")
        
        # ========================================
        # BƯỚC 8: PHÂN TÍCH THỐNG KÊ NGƯỠNG (THRESHOLD STATISTICAL ANALYSIS)
        # ========================================
        # Mục đích: Phân tích toán học về hiệu quả của các ngưỡng khác nhau
        # KHÔNG phải chỉ hiển thị UI - đây là phân tích thống kê để tối ưu thuật toán
        #
        # 8.1. Multi-threshold Analysis:
        #    - Thử các ngưỡng khác nhau: 10%, 20%, 50%, 100% của median area
        #    - Đếm số lượng contours ở mỗi ngưỡng
        #    - Mục đích: Hiểu được độ nhạy của thuật toán với các ngưỡng khác nhau
        #
        # 8.2. Algorithm Optimization Insights:
        #    - Nếu count_10 >> count_100: có nhiều hạt nhỏ bị lọc bỏ
        #    - Nếu count_10 ≈ count_100: thuật toán đã chọn ngưỡng hợp lý
        #    - Nếu count_20 >> count_50: có nhiều hạt trung bình
        #    - Giúp điều chỉnh dynamic_min_area và edge_threshold trong config
        #
        # 8.3. Quality Assessment:
        #    - Tỷ lệ count_50/count_10: đo độ "clean" của segmentation
        #    - Tỷ lệ count_100/count_10: đo độ "strict" của filtering
        #    - Mục đích: Đánh giá chất lượng của pipeline xử lý ảnh
        all_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_areas = [cv2.contourArea(cnt) for cnt in all_contours if cv2.contourArea(cnt) > 0]
        
        debug_info = ""
        if len(all_areas) > 0:
            median_area = np.median(all_areas)
            count_10 = len([a for a in all_areas if a >= median_area * 0.1])
            count_20 = len([a for a in all_areas if a >= median_area * 0.2])
            count_50 = len([a for a in all_areas if a >= median_area * 0.5])
            count_100 = len([a for a in all_areas if a >= median_area])
            
            debug_info = f"Debug: ≥10%={count_10}, ≥20%={count_20}, ≥50%={count_50}, ≥100%={count_100}"
        
        # ========================================
        # BƯỚC 9: PHÂN TÍCH HẠT Ở BIÊN (EDGE GRAIN ANALYSIS)
        # ========================================
        # Mục đích: Phân tích toán học về các hạt ở biên ảnh
        # KHÔNG phải chỉ hiển thị UI - đây là phân tích geometric quan trọng
        #
        # 9.1. Geometric Edge Detection:
        #    - Tính bounding rectangle cho mỗi contour: (x, y, width, height)
        #    - Kiểm tra vị trí của rectangle so với biên ảnh
        #    - Công thức: is_near_edge = (x < margin OR y < margin OR 
        #                                  x + w > width - margin OR 
        #                                  y + h > height - margin)
        #    - margin = 10 pixels (từ PostprocessingConfig.EDGE_MARGIN)
        #
        # 9.2. Edge Grain Counting:
        #    - Đếm số hạt có bounding box gần biên ảnh
        #    - Mục đích: Đánh giá hiệu quả của thuật toán xử lý biên
        #    - Nếu edge_grains/count cao: nhiều hạt ở biên → cần cải thiện border expansion
        #    - Nếu edge_grains/count thấp: ít hạt ở biên → thuật toán hoạt động tốt
        #
        # 9.3. Algorithm Validation:
        #    - So sánh với kết quả từ postprocess() (đã có border expansion)
        #    - Kiểm tra xem border expansion có hiệu quả không
        #    - Giúp điều chỉnh BORDER_SIZE và EDGE_MARGIN trong config
        #
        # 9.4. Quality Metrics:
        #    - Tỷ lệ edge_grains/total_count: đo độ "edge-sensitive" của thuật toán
        #    - Nếu tỷ lệ cao: thuật toán nhạy với hạt ở biên
        #    - Nếu tỷ lệ thấp: thuật toán ổn định với hạt ở biên
        edge_grains = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            margin = PostprocessingConfig.EDGE_MARGIN
            is_near_edge = (x < margin or y < margin or 
                           x + w > cleaned.shape[1] - margin or 
                           y + h > cleaned.shape[0] - margin)
            if is_near_edge:
                edge_grains += 1
        
        brightness_summary = f"Độ sáng: {brightness_info['brightness_level']} (TB: {brightness_info['mean_brightness']:.1f}, Tỷ lệ tối: {brightness_info['dark_ratio']:.1%})"
        edge_summary = f"Hạt ở biên: {edge_grains}/{count}" if count > 0 else "Hạt ở biên: 0/0"
        self.info_label.config(text=f"Số hạt: {count} | Diện tích TB: {avg_area:.2f} px | Diện tích lớn nhất: {max_area:.2f} px | {edge_summary} | {brightness_summary} | {debug_info}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RiceApp(root)
    root.mainloop()