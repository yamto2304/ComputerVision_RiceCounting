# -*- coding: utf-8 -*-
"""
File cấu hình tham số cho ứng dụng đếm hạt gạo
Tất cả các tham số hardcode đã được chuyển vào đây để dễ dàng tùy chỉnh
"""

# =============================================================================
# THAM SỐ TIỀN XỬ LÝ (PREPROCESSING PARAMETERS)
# =============================================================================

class PreprocessingConfig:
    """Cấu hình cho các bước tiền xử lý ảnh"""
    
    # Top-hat morphology kernel sizes để thử
    TOPHAT_KERNEL_SIZES = [(15, 15), (10, 10), (20, 20)]
    GAUSSIAN_KERNEL_SIZE = (3, 3)
    GAUSSIAN_SIGMA = 0
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Tham số cho phân tích độ sáng
    BRIGHTNESS_THRESHOLDS = {
        'very_dark': 30,
        'dark': 60,
        'bright': 150,
        'very_bright': 200
    }
    
    # Hệ số điều chỉnh CLAHE cho ảnh tối
    CLAHE_DARK_MULTIPLIER = 2.0
    CLAHE_VERY_DARK_MULTIPLIER = 3.0

# =============================================================================
# THAM SỐ PHÂN ĐOẠN (SEGMENTATION PARAMETERS)
# =============================================================================

class SegmentationConfig:
    """Cấu hình cho các phương pháp phân đoạn ảnh"""
    
    # Tham số adaptive threshold để thử
    ADAPTIVE_BLOCK_SIZES = [21, 11, 15, 25]
    ADAPTIVE_C_CONSTANTS = [2, 1, 3, 5]
    
    # Tham số mặc định
    ADAPTIVE_BLOCK_SIZE = 21
    ADAPTIVE_C_CONSTANT = 2

# =============================================================================
# THAM SỐ HẬU XỬ LÝ (POSTPROCESSING PARAMETERS)
# =============================================================================

class PostprocessingConfig:
    """Cấu hình cho các bước hậu xử lý"""
    
    MORPH_KERNEL_SIZE = (3, 3)
    MORPH_ITERATIONS = 1
    DISTANCE_MASK_SIZE = 5
    WATERSHED_THRESHOLD_RATIO = 0.3
    
    # Tham số cho xử lý biên
    BORDER_SIZE = 10
    EDGE_MARGIN = 10

# =============================================================================
# THAM SỐ ĐẾM HẠT (COUNTING PARAMETERS)
# =============================================================================

class CountingConfig:
    """Cấu hình cho việc đếm và phân tích hạt gạo"""
    
    MIN_AREA = 50
    CONTOUR_RETRIEVAL_MODE = 'EXTERNAL'
    CONTOUR_APPROX_METHOD = 'SIMPLE'
    
    # Tham số cho ngưỡng động
    DYNAMIC_AREA_RATIO = 0.25  # Tỷ lệ median_area để tính dynamic_min_area
    EDGE_THRESHOLD_RATIO = 0.12  # Tỷ lệ median_area cho edge_threshold
    
    # Tham số hình học cho hạt ở biên
    EDGE_ASPECT_RATIO_MAX = 2.5  # Tỷ lệ khung hình tối đa
    EDGE_CIRCULARITY_MIN = 0.4   # Độ tròn tối thiểu
    EDGE_COMPACTNESS_MIN = 0.5   # Độ compact tối thiểu

# =============================================================================
# THAM SỐ TỔNG QUAN (GENERAL PARAMETERS)
# =============================================================================

class GeneralConfig:
    """Cấu hình tổng quan cho ứng dụng"""
    
    DEFAULT_SEGMENTATION_METHOD = 'otsu'