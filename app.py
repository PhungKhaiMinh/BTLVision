#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import json

# Bỏ qua tất cả cảnh báo
warnings.filterwarnings('ignore')

# Thiết lập biến môi trường để tắt file watching
os.environ['STREAMLIT_SERVER_FILE_WATCHDOG'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCH_POLL'] = 'false'

# Import torch trước mọi module khác
import torch

# Sau đó mới import Streamlit
import streamlit as st
import cv2
import numpy as np
import time
import glob
from PIL import Image
import io
import tempfile
import datetime  # Thêm để tạo tên file duy nhất

# Hàm để cắt và lưu đối tượng đã phát hiện
def crop_and_save_objects(image, detections, model_data, ref_dir="ref", min_size=50, input_size=640):
    """
    Cắt và lưu các đối tượng đã phát hiện vào thư mục tham chiếu
    
    Args:
        image: Ảnh gốc (định dạng BGR từ OpenCV)
        detections: Danh sách các detections (class_id, confidence, box)
        model_data: Dữ liệu model chứa tên các lớp
        ref_dir: Thư mục để lưu ảnh đã cắt
        min_size: Kích thước tối thiểu của vùng cắt (pixel)
        input_size: Kích thước ảnh đầu vào cho model (mặc định 640)
    
    Returns:
        saved_objects: Danh sách các đối tượng đã lưu (đường dẫn, tên lớp)
    """
    # Đảm bảo thư mục tồn tại
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    
    # Timestamp để tạo tên file duy nhất
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_objects = []
    
    # Lấy kích thước ảnh gốc
    img_h, img_w = image.shape[:2]
    
    # Tính tỷ lệ giữa ảnh gốc và ảnh đã resize
    scale_x = img_w / input_size
    scale_y = img_h / input_size
    
    # Duyệt qua các detections
    for i, (cls_id, conf, box) in enumerate(detections):
        # Lấy tọa độ box từ model (x, y, w, h) và điều chỉnh theo tỷ lệ
        x, y, w, h = box
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        
        # Kiểm tra kích thước tối thiểu
        if w < min_size or h < min_size:
            continue
        
        # Đảm bảo tọa độ nằm trong ảnh
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Cắt vùng ảnh
        cropped = image[y:y+h, x:x+w]
        
        # Lấy tên lớp
        class_names = model_data.get("classes", {})
        if isinstance(class_names, dict):
            class_name = class_names.get(cls_id, f"Class_{cls_id}")
        elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"Class_{cls_id}"
        
        # Tạo tên file
        filename = f"{class_name}_{timestamp}_{i}.jpg"
        filepath = os.path.join(ref_dir, filename)
        
        # Lưu ảnh
        cv2.imwrite(filepath, cropped)
        
        saved_objects.append((filepath, class_name))
    
    return saved_objects

# Tạo các fucntion wrapper để tránh import trực tiếp từ run_yolo.py
def load_model_wrapper(model_path, use_optimization=False, opt_format="engine", half=False, verbose=False):
    """Wrapper để tải model YOLO từ run_yolo.py"""
    from run_yolo import load_model
    return load_model(model_path, use_optimization, opt_format, half, verbose)

def process_image_wrapper(image, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, 
                         nms_threshold=0.5, center_distance_threshold=0.2, return_image_with_boxes=True, 
                         debug=False):
    """Wrapper để xử lý ảnh với YOLO từ run_yolo.py"""
    try:
        if image is None:
            return None, []
            
        # Lấy kích thước gốc
        orig_h, orig_w = image.shape[:2]
        
        # Resize ảnh cho mô hình với tỷ lệ giữ nguyên
        # Tránh resize 2 lần để tăng hiệu suất
        if image.shape[0] != input_size or image.shape[1] != input_size:
            resized_img = cv2.resize(image, (input_size, input_size))
        else:
            resized_img = image
            
        # Tỷ lệ scale để chuyển từ kích thước mô hình về kích thước gốc
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size
        
        from run_yolo import process_image
        result = process_image(resized_img, model_data, conf_threshold, input_size, apply_nms, nms_threshold,
                       center_distance_threshold, return_image_with_boxes, debug)
        
        # Đảm bảo đầu ra là tuple (image, detections)
        if isinstance(result, tuple) and len(result) == 2:
            img, detections = result
            if detections is None:
                detections = []
                
            # Điều chỉnh tọa độ bounding box về kích thước ảnh gốc
            adjusted_detections = []
            for det in detections:
                if len(det) == 3:  # (cls_id, conf, box)
                    cls_id, conf, box = det
                    x, y, w, h = box
                    # Scale từ input_size về kích thước gốc
                    x_orig = x * scale_x
                    y_orig = y * scale_y
                    w_orig = w * scale_x
                    h_orig = h * scale_y
                    # Đảm bảo tọa độ nằm trong giới hạn ảnh
                    x_orig = max(0, min(x_orig, orig_w-1))
                    y_orig = max(0, min(y_orig, orig_h-1))
                    w_orig = min(w_orig, orig_w-x_orig)
                    h_orig = min(h_orig, orig_h-y_orig)
                    adjusted_detections.append((cls_id, conf, [x_orig, y_orig, w_orig, h_orig]))
                else:
                    adjusted_detections.append(det)
            
            # Nếu yêu cầu hiển thị bounding box, nhưng kết quả không có ảnh
            if return_image_with_boxes and img is None:
                # Tự tạo kết quả hiển thị nếu cần
                img = image.copy()
                
                # Vẽ bounding box cho từng detection (đã điều chỉnh tọa độ)
                for cls_id, conf, box in adjusted_detections:
                    # Tọa độ đã được điều chỉnh về kích thước gốc
                    x, y, w, h = [int(v) for v in box]
                    
                    # Lấy tên class
                    class_names = model_data.get('classes', {})
                    if isinstance(class_names, dict):
                        class_name = class_names.get(cls_id, f"Class {cls_id}")
                    elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                        class_name = class_names[cls_id]
                    else:
                        class_name = f"Class {cls_id}"
                    
                    # Vẽ với style giống run_yolo.py
                    color = (0, 0, 255)  # Đỏ trong BGR
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                    
                    # Vẽ nhãn với định dạng giống run_yolo.py
                    label = f"{class_name}: {conf:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
                    cv2.putText(img, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif return_image_with_boxes and img is not None and (img.shape[0] != orig_h or img.shape[1] != orig_w):
                # Nếu kích thước ảnh kết quả khác với ảnh gốc, resize lại
                img = cv2.resize(img, (orig_w, orig_h))
                
            return img, adjusted_detections
        else:
            # Trường hợp kết quả không đúng định dạng
            print("Kết quả từ process_image không đúng định dạng")
            return None, []
    except Exception as e:
        print(f"Lỗi trong process_image_wrapper: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, []

def create_object_tracker(buffer_size=10, iou_threshold=0.5, stability_threshold=0.5, confidence_threshold=0.3):
    """Wrapper để tạo object tracker từ run_yolo.py"""
    from run_yolo import ObjectTracker
    return ObjectTracker(buffer_size, iou_threshold, stability_threshold, confidence_threshold)

def custom_nms(detections, iou_threshold=0.5, same_class_only=True, center_distance_threshold=0.2):
    """Wrapper cho hàm NMS tùy chỉnh từ run_yolo.py"""
    from run_yolo import apply_custom_nms
    return apply_custom_nms(detections, iou_threshold, same_class_only, center_distance_threshold)

# Thêm các hàm cho tính năng theo dõi tồn kho
def load_inventory_data(json_file="Data.json"):
    """
    Tải dữ liệu tồn kho từ file JSON
    
    Args:
        json_file: Đường dẫn đến file JSON chứa dữ liệu sản phẩm
        
    Returns:
        Dict chứa dữ liệu sản phẩm đã được tổ chức lại theo class
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Chuyển đổi danh sách sản phẩm thành dict để dễ tra cứu theo class
        inventory_dict = {}
        if 'products' in data:
            for product in data['products']:
                if 'class' in product and 'on_shelf' in product and 'price' in product:
                    inventory_dict[product['class']] = {
                        'expected': product['on_shelf'],
                        'price': product['price']
                    }
        return inventory_dict
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu tồn kho: {e}")
        return {}

def count_products_from_detections(detections, model_data):
    """
    Đếm số lượng sản phẩm theo loại từ kết quả phát hiện
    
    Args:
        detections: Danh sách các detection từ YOLO
        model_data: Dữ liệu model chứa tên các lớp
        
    Returns:
        Dict chứa số lượng sản phẩm đếm được theo loại
    """
    class_names = model_data.get('classes', {})
    product_counts = {}
    
    for det in detections:
        if det is None:
            continue
            
        if len(det) == 4:
            cls_id, obj_id, conf, _ = det
        else:
            cls_id, conf, _ = det
            
        # Lấy tên lớp
        if isinstance(class_names, dict):
            class_name = class_names.get(cls_id, f"Class {cls_id}")
        elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"Class {cls_id}"
            
        # Cập nhật số lượng
        if class_name in product_counts:
            product_counts[class_name] += 1
        else:
            product_counts[class_name] = 1
            
    return product_counts

def compare_inventory(actual_counts, expected_inventory):
    """
    So sánh số lượng thực tế với số lượng dự kiến và xác định trạng thái tồn kho
    
    Args:
        actual_counts: Dict chứa số lượng sản phẩm đếm được
        expected_inventory: Dict chứa thông tin tồn kho dự kiến
        
    Returns:
        Dict chứa kết quả so sánh và trạng thái tồn kho
    """
    inventory_status = {}
    
    # Xử lý các sản phẩm có trong dữ liệu dự kiến
    for product_class, info in expected_inventory.items():
        expected = info['expected']
        actual = actual_counts.get(product_class, 0)
        
        # Tính chênh lệch
        difference = actual - expected
        
        # Xác định trạng thái
        if actual == 0:
            status = "Hết hàng"
            alert = True
        elif actual < expected:
            status = "Sắp hết"
            alert = True
        elif actual == expected:
            status = "Đủ hàng"
            alert = False
        else:
            status = "Dư hàng"
            alert = False
            
        inventory_status[product_class] = {
            "expected": expected,
            "actual": actual,
            "difference": difference,
            "status": status,
            "alert": alert,
            "price": info['price']
        }
        
    # Thêm các sản phẩm phát hiện được nhưng không có trong dữ liệu dự kiến
    for product_class, count in actual_counts.items():
        if product_class not in inventory_status:
            inventory_status[product_class] = {
                "expected": 0,
                "actual": count,
                "difference": count,
                "status": "Không trong kế hoạch",
                "alert": False,
                "price": 0
            }
            
    return inventory_status

# Thiết lập trang Streamlit
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")

# Ẩn một số thông báo lỗi
try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
except Exception:
    pass

def get_video_frame(video_path, frame_number):
    """Lấy frame từ video tại vị trí cụ thể"""
    cap = cv2.VideoCapture(video_path)
    
    # Thiết lập vị trí frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Đọc frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame

def process_video(video_path, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, nms_threshold=0.5, center_distance_threshold=0.2, 
                  buffer_size=10, iou_threshold=0.5, stability_threshold=0.6):
    """Xử lý video và tạo phiên bản mới với detections"""
    # Khởi tạo object tracker cho video
    tracker = create_object_tracker(
        buffer_size=buffer_size,
        iou_threshold=iou_threshold,
        stability_threshold=stability_threshold,
        confidence_threshold=conf_threshold
    )
    
    # Tạo tệp tạm thời để lưu video đầu ra
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    
    # Mở video nguồn
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Không thể mở video: {video_path}")
        return None
    
    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Thiết lập writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Progress bar
    progress_bar = st.progress(0)
    frame_text = st.empty()
    
    # Xử lý từng frame
    frame_idx = 0
    processed_frames = 0
    
    # Thông báo thông số
    st.info(f"Xử lý video với: Buffer={buffer_size}, IoU={iou_threshold}, Stability={stability_threshold}, NMS IoU={nms_threshold}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ xử lý mỗi 2-3 frame để tăng tốc độ (có thể điều chỉnh)
        if frame_idx % 2 == 0:
            # Sao chép frame gốc để hiển thị
            original_frame = frame.copy()
            
            # Thay đổi kích thước frame
            resized_frame = cv2.resize(frame, (input_size, input_size))
            
            # Lấy kết quả detections từ model
            _, detections = process_image_wrapper(
                resized_frame, 
                model_data, 
                conf_threshold, 
                input_size, 
                apply_nms, 
                nms_threshold, 
                center_distance_threshold,
                return_image_with_boxes=False
            )
            
            # Đảm bảo detections không phải None
            if detections is None:
                detections = []
            
            # Cập nhật tracker
            tracker.update(detections)
            stable_objects = tracker.get_stable_objects()
            
            # Chọn đối tượng để hiển thị (đối tượng ổn định)
            objects_to_display = stable_objects
            
            # Đảm bảo objects_to_display không phải None
            if objects_to_display is None:
                objects_to_display = []
            
            # Tỷ lệ để scale bounding box từ kích thước resize về kích thước gốc
            scale_x = width / input_size
            scale_y = height / input_size
            
            # Vẽ các đối tượng
            class_names = model_data.get("classes", {})
            
            # Vẽ kết quả
            result_img = original_frame.copy()
            
            for det in objects_to_display:
                # Kiểm tra det không phải None
                if det is None:
                    continue
                    
                # Unpack tuple một cách an toàn
                try:
                    if len(det) == 4:
                        cls_id, obj_id, conf, box = det
                    else:
                        cls_id, conf, box = det
                        obj_id = "-"
                        
                    # Đảm bảo box là iterable
                    if box is None:
                        continue
                    
                    # Scale bounding box về kích thước gốc
                    x, y, w, h = box
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)
                
                    # Kiểm tra và đảm bảo tọa độ nằm trong frame
                    x = max(0, min(x, width-1))
                    y = max(0, min(y, height-1))
                    w = min(w, width-x)
                    h = min(h, height-y)
                
                    # Lấy tên lớp
                    if isinstance(class_names, dict):
                        class_name = class_names.get(cls_id, f"Class {cls_id}")
                    elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                        class_name = class_names[cls_id]
                    else:
                        class_name = f"Class {cls_id}"
                
                    # Sử dụng màu đỏ như trong run_yolo.py
                    color = (0, 0, 255)  # BGR - màu đỏ
                    
                    # Vẽ bounding box với độ dày 3 như trong run_yolo.py
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
                
                    # Hiển thị label và độ tin cậy - đúng định dạng như run_yolo.py
                    label = f"{class_name} | ID:{obj_id} | {conf:.2f}"
                
                    # Sử dụng font và kích thước giống với run_yolo.py
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(result_img, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
                    cv2.putText(result_img, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception as e:
                    # Log lỗi nếu cần nhưng không gây crash
                    print(f"Lỗi khi xử lý detection: {e}")
                    continue
            
            # Hiển thị FPS và thông tin lọc theo style của run_yolo.py
            cv2.putText(result_img, f"Objects: {len(objects_to_display)}/{len(detections)}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Hiển thị thông tin các đối tượng phát hiện được
            if detections:
                with st.expander("Chi tiết các đối tượng phát hiện", expanded=debug_mode):
                    for i, det in enumerate(detections):
                        try:
                            if det is None:
                                continue
                            
                            if len(det) == 4:
                                cls_id, obj_id, conf, _ = det
                            else:
                                cls_id, conf, _ = det
                                obj_id = "-"
                                
                            class_names = model_data.get("classes", {})
                            if isinstance(class_names, dict):
                                class_name = class_names.get(cls_id, f"Class {cls_id}")
                            elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                                class_name = class_names[cls_id]
                            else:
                                class_name = f"Class {cls_id}"
                                
                            st.write(f"{i+1}. {class_name} | ID:{obj_id} | {conf:.2f}")
                        except Exception as e:
                            st.write(f"Lỗi khi hiển thị chi tiết detection: {str(e)}")
                            continue
            
            writer.write(result_img)
            
            # Cập nhật tiến trình
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)
            frame_text.text(f"Xử lý frame: {frame_idx+1}/{total_frames}")
            
            processed_frames += 1
        else:
            writer.write(frame)
            
        frame_idx += 1
    
    # Giải phóng tài nguyên
    cap.release()
    writer.release()
    
    st.success(f"Đã xử lý {processed_frames} frames. Video được lưu tại: {output_path}")
    return output_path

# Giữ lại hàm wrapper cho tương thích ngược
def process_camera_frame(frame, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, nms_threshold=0.5, center_distance_threshold=0.2, tracker=None):
    """Xử lý frame từ camera với object tracking - hàm tương thích ngược"""
    return process_camera_frame_wrapper(frame, model_data, conf_threshold, input_size, apply_nms, nms_threshold, center_distance_threshold, tracker)

def process_camera_frame_wrapper(frame, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, nms_threshold=0.5, center_distance_threshold=0.2, tracker=None):
    """Wrapper cho hàm xử lý camera frame"""
    if frame is None:
        return None, []
    
    # Lấy kích thước gốc trước khi resize
    height, width = frame.shape[:2]
    
    # Lưu tỷ lệ kích thước gốc/input_size - sử dụng các phép tính float để chính xác
    scale_x = width / input_size
    scale_y = height / input_size
    
    # Resize frame với tỷ lệ giữ nguyên cho mô hình
    resized_frame = cv2.resize(frame, (input_size, input_size))
        
    # Lấy detections từ model
    _, detections = process_image_wrapper(
        resized_frame, 
        model_data, 
        conf_threshold, 
        input_size, 
        apply_nms, 
        nms_threshold, 
        center_distance_threshold,
        return_image_with_boxes=False
    )
    
    # Đảm bảo detections không phải None
    if detections is None:
        detections = []
    
    # Nếu có tracker, cập nhật và lấy các đối tượng ổn định
    if tracker:
        tracker.update(detections)
        objects_to_display = tracker.get_stable_objects()
    else:
        objects_to_display = detections
    
    # Đảm bảo objects_to_display không phải None
    if objects_to_display is None:
        objects_to_display = []
    
    # Vẽ kết quả
    result_img = frame.copy()
    
    # Vẽ các đối tượng
    class_names = model_data.get("classes", {})
    
    for det in objects_to_display:
        # Kiểm tra det không phải None
        if det is None:
            continue
            
        # Unpack tuple một cách an toàn
        try:
            if len(det) == 4:
                cls_id, obj_id, conf, box = det
            else:
                cls_id, conf, box = det
                obj_id = "-"
                
            # Đảm bảo box là iterable
            if box is None:
                continue
                
            # Scale bounding box về kích thước gốc
            x, y, w, h = box
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
        
            # Kiểm tra và đảm bảo tọa độ nằm trong frame
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = min(w, width-x)
            h = min(h, height-y)
        
            # Lấy tên lớp
            if isinstance(class_names, dict):
                class_name = class_names.get(cls_id, f"Class {cls_id}")
            elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = f"Class {cls_id}"
        
            # Sử dụng màu đỏ như trong run_yolo.py
            color = (0, 0, 255)  # BGR - màu đỏ
            
            # Vẽ bounding box với độ dày 3 như trong run_yolo.py
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        
            # Hiển thị label và độ tin cậy - đúng định dạng như run_yolo.py
            label = f"{class_name} | ID:{obj_id} | {conf:.2f}"
        
            # Sử dụng font và kích thước giống với run_yolo.py
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_img, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
            cv2.putText(result_img, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            # Log lỗi nếu cần nhưng không gây crash
            print(f"Lỗi khi xử lý detection: {e}")
            continue
    
    # Hiển thị thông tin đối tượng giống với run_yolo.py
    info_text = f"Objects: {len(objects_to_display)}/{len(detections)}"
    
    # Thay đổi vị trí hiển thị như trong run_yolo.py (vị trí 10, 30)
    cv2.putText(result_img, info_text, (10, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_img, detections

def load_model_with_retry(model_path, use_optimization, opt_format, half_precision, debug_mode, max_retries=3):
    """Tải model với retry mechanism để tránh lỗi khi khởi động"""
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Đang tải model... (lần thử {attempt+1}/{max_retries})"):
                # Tối ưu: cache model để tránh tải lại
                if 'cached_model' in st.session_state and st.session_state.cached_model_path == model_path:
                    if debug_mode:
                        st.info(f"Sử dụng model đã cache {model_path}")
                    return st.session_state.cached_model
                
                # Kiểm tra GPU và torch - tối ưu quá trình kiểm tra
                is_cuda_available = torch.cuda.is_available()
                if is_cuda_available:
                    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
                    # Thiết lập memory caching
                    torch.backends.cudnn.benchmark = True
                    # Thiết lập để giảm bộ nhớ không cần thiết
                    if half_precision:
                        torch.set_default_tensor_type(torch.cuda.HalfTensor)
                else:
                    gpu_info = "CPU only"
                    
                if debug_mode:
                    st.info(f"Môi trường: {gpu_info}, PyTorch {torch.__version__}")
                
                # Kiểm tra sự tương thích của TensorRT
                if opt_format == "engine" and is_cuda_available:
                    try:
                        import tensorrt
                        tensorrt_version = tensorrt.__version__
                        if debug_mode:
                            st.info(f"TensorRT: {tensorrt_version}")
                    except ImportError:
                        st.warning("TensorRT không được cài đặt. Chuyển sang ONNX...")
                        opt_format = "onnx"
                    except Exception as e:
                        st.warning(f"Lỗi khi kiểm tra TensorRT: {str(e)}. Chuyển sang ONNX...")
                        opt_format = "onnx"
                
                # Kiểm tra GPU cụ thể
                if opt_format == "engine" and is_cuda_available:
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    if "1050" in gpu_name or "gtx" in gpu_name:
                        st.warning(f"GPU {gpu_name} có thể không tương thích với TensorRT mới. Chuyển sang ONNX...")
                        opt_format = "onnx"
                
                # Tải model
                model = load_model_wrapper(model_path, use_optimization, opt_format, half_precision, debug_mode)
                
                # Cache model
                st.session_state.cached_model = model
                st.session_state.cached_model_path = model_path
                
                return model
        except Exception as e:
            error_msg = str(e)
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
            
            # Xử lý một số lỗi cụ thể
            if "torch.classes" in error_msg:
                st.warning("Đang gặp lỗi với torch.classes. Thử với cấu hình khác...")
                if opt_format == "engine":
                    opt_format = "onnx"
                elif opt_format == "onnx":
                    use_optimization = False
            
            if attempt < max_retries - 1:
                st.warning(f"Lỗi khi tải model: {error_msg}. Thử lại...")
                time.sleep(1)  # Chờ một chút trước khi thử lại
            else:
                if use_optimization:
                    st.warning("Không thể tải model tối ưu. Thử với model PyTorch gốc...")
                    # Lần cuối, thử với model PyTorch gốc
                    try:
                        model = load_model_wrapper(model_path, False, None, False, debug_mode)
                        # Cache model
                        st.session_state.cached_model = model
                        st.session_state.cached_model_path = model_path
                        return model
                    except Exception as e2:
                        st.error(f"Không thể tải model: {str(e2)}")
                        return None
                else:
                    st.error(f"Không thể tải model sau {max_retries} lần thử: {error_msg}")
                    return None
                    
# Tối ưu xử lý DroidCam
def optimize_droidcam_connection(droidcam_ip, droidcam_port):
    """Tối ưu kết nối với DroidCam"""
    camera_url = f"http://{droidcam_ip}:{droidcam_port}/video"
    
    # Thử sử dụng UDP thay vì HTTP khi có thể
    if ":" in droidcam_ip and int(droidcam_port) > 4000:
        # Chỉ thử UDP khi không phải port mặc định (4747)
        try:
            # Thử kết nối UDP cho độ trễ thấp hơn
            udp_url = f"udp://{droidcam_ip}:{int(droidcam_port)-1}"
            udp_cap = cv2.VideoCapture(udp_url)
            if udp_cap.isOpened():
                # Thiết lập cho kết nối UDP
                udp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                udp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                # Kiểm tra kết nối
                ret, frame = udp_cap.read()
                if ret and frame is not None:
                    print("Đã kết nối qua UDP")
                    return udp_cap, "Đã kết nối DroidCam qua UDP (độ trễ thấp)"
                else:
                    udp_cap.release()
        except:
            pass
    
    try:
        # Tạo kết nối với buffer thấp để giảm độ trễ
        cap = cv2.VideoCapture(camera_url)
        
        # Đặt các tham số tối ưu
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Đặt buffer size=1 để giảm độ trễ
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Sử dụng MJPEG để giảm độ trễ xử lý
        
        # Kiểm tra kết nối
        if not cap.isOpened():
            # Thử một URL thay thế
            alt_url = f"http://{droidcam_ip}:{droidcam_port}/mjpegfeed?640x480"
            cap = cv2.VideoCapture(alt_url)
            if not cap.isOpened():
                return None, f"Không thể kết nối với DroidCam tại {camera_url}"
        
        # Đọc một frame để kiểm tra kết nối
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None, "Kết nối với DroidCam thành công nhưng không đọc được dữ liệu"
        
        # Thiết lập độ phân giải thấp hơn để tăng tốc xử lý
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Thiết lập hiệu suất tốt hơn
        cap.set(cv2.CAP_PROP_FPS, 30)  # Thử đặt FPS thành 30
        
        # Kiểm tra kích thước frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        return cap, f"Đã kết nối DroidCam thành công ({width}x{height} @ {fps:.1f}fps)"
    except Exception as e:
        print(f"Lỗi khi kết nối DroidCam: {e}")
        import traceback
        print(traceback.format_exc())
        return None, f"Lỗi khi kết nối DroidCam: {str(e)}"

def main():
    # Thử chặn các lỗi chung khi khởi chạy Streamlit
    try:
        st.title("YOLOv8 Object Detection - Streamlit UI")
        
        # Sidebar cho cấu hình
        st.sidebar.header("Cấu hình")
        
        # Chọn model
        model_path = st.sidebar.text_input("Đường dẫn đến model YOLOv8", "last(1)-can-n.pt")
        
        # Thêm chế độ debug
        debug_mode = st.sidebar.checkbox("Chế độ debug", False)
        st.session_state.debug_mode = debug_mode
        
        # Cấu hình nâng cao
        with st.sidebar.expander("Cấu hình nâng cao", expanded=True):
            conf_threshold = st.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.25, 0.01)
            input_size = st.select_slider("Kích thước đầu vào", options=[320, 416, 512, 640, 768, 1024], value=640)
            use_optimization = st.checkbox("Sử dụng tối ưu hóa", True)
            # Đổi default sang ONNX vì GTX 1050 Ti gặp vấn đề với TensorRT
            opt_format = st.selectbox("Định dạng tối ưu", ["onnx", "engine"], index=0)
            half_precision = st.checkbox("Sử dụng FP16 (half precision)", True)
            
            # NMS
            apply_nms = st.checkbox("Áp dụng NMS tùy chỉnh", True)
            nms_threshold = st.slider("Ngưỡng IoU cho NMS", 0.1, 0.9, 0.5, 0.1)
            center_distance_threshold = st.slider("Ngưỡng khoảng cách tâm cho NMS", 0.1, 0.5, 0.2, 0.05)
            
            # Tùy chọn lưu đối tượng
            st.divider()
            st.subheader("Tùy chọn lưu đối tượng")
            save_objects = st.checkbox("Tự động lưu vật thể phát hiện vào thư mục ref", True)
            ref_dir = st.text_input("Thư mục lưu vật thể", "ref")
            min_object_size = st.slider("Kích thước tối thiểu của vật thể (pixel)", 20, 200, 50)
            
            # Thêm các tham số từ run_yolo.py
            st.divider()
            st.subheader("Tham số theo dõi đối tượng")
            buffer_size = st.slider("Buffer size (số frame để theo dõi)", 5, 20, 10, 1)
            iou_threshold = st.slider("Ngưỡng IoU cho việc khớp đối tượng", 0.3, 0.7, 0.5, 0.05)
            stability_threshold = st.slider("Ngưỡng ổn định để hiển thị đối tượng", 0.3, 0.9, 0.6, 0.05)
            
            # Thêm cấu hình cho tính năng theo dõi tồn kho
            st.divider()
            st.subheader("Cấu hình theo dõi tồn kho")
            inventory_file = st.text_input("File dữ liệu tồn kho", "Data.json")
            low_stock_threshold = st.slider("Ngưỡng cảnh báo tồn kho thấp (%)", 0, 100, 30, 5)
        
        # Tab chọn loại đầu vào
        tab1, tab2, tab3, tab4 = st.tabs(["Ảnh", "Video", "Camera", "Theo dõi tồn kho"])
        
        # Tải model khi cần
        model_data = None
        if os.path.exists(model_path):
            model_placeholder = st.empty()
            with model_placeholder.container():
                model_data = load_model_with_retry(model_path, use_optimization, opt_format, half_precision, debug_mode)
            model_placeholder.empty()
        else:
            if model_path:
                st.error(f"Không tìm thấy model tại: {model_path}")
                
        # Tải dữ liệu tồn kho từ file JSON
        inventory_data = {}
        if os.path.exists(inventory_file):
            inventory_data = load_inventory_data(inventory_file)
        
        # Tab 1: Xử lý ảnh
        with tab1:
            st.header("Nhận diện đối tượng trong ảnh")
            
            # Tải ảnh
            uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")
            image_path = st.text_input("Hoặc nhập đường dẫn đến ảnh", key="image_path_input")
            
            if uploaded_file is not None:
                # Xử lý ảnh được tải lên
                # Đọc ảnh theo cách thống nhất
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                st.image(image_rgb, caption="Ảnh đã tải lên", use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Nhận diện đối tượng", key="detect_uploaded_image"):
                        if model_data:
                            with st.spinner("Đang xử lý..."):
                                # Xử lý ảnh sử dụng hàm từ run_yolo.py
                                result_img, detections = process_image_wrapper(
                                    image, 
                                    model_data, 
                                    conf_threshold, 
                                    input_size, 
                                    apply_nms, 
                                    nms_threshold,
                                    center_distance_threshold,
                                    debug=debug_mode
                                )
                                
                                # Cắt và lưu các đối tượng đã phát hiện vào thư mục ref nếu được yêu cầu
                                saved_objects = []
                                if save_objects and detections:
                                    saved_objects = crop_and_save_objects(
                                        image, 
                                        detections, 
                                        model_data, 
                                        ref_dir=ref_dir, 
                                        min_size=min_object_size,
                                        input_size=input_size
                                    )
                                
                                # Chuyển kết quả về RGB để hiển thị đúng
                                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                st.image(result_rgb, caption=f"Kết quả nhận diện: {len(detections)} đối tượng", use_container_width=True)
                                
                                # Hiển thị thông tin các đối tượng phát hiện được
                                if detections:
                                    with st.expander("Chi tiết các đối tượng phát hiện", expanded=debug_mode):
                                        for i, det in enumerate(detections):
                                            try:
                                                if det is None:
                                                    continue
                                                
                                                if len(det) == 4:
                                                    cls_id, obj_id, conf, _ = det
                                                else:
                                                    cls_id, conf, _ = det
                                                    obj_id = "-"
                                                
                                                class_names = model_data.get("classes", {})
                                                if isinstance(class_names, dict):
                                                    class_name = class_names.get(cls_id, f"Class {cls_id}")
                                                elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                                                    class_name = class_names[cls_id]
                                                else:
                                                    class_name = f"Class {cls_id}"
                                                
                                                st.write(f"{i+1}. {class_name} | ID:{obj_id} | {conf:.2f}")
                                            except Exception as e:
                                                st.write(f"Lỗi khi hiển thị chi tiết detection: {str(e)}")
                                                continue
                                
                                # Hiển thị thông tin về các đối tượng đã lưu
                                if saved_objects:
                                    with st.expander("Đối tượng đã lưu vào thư mục ref", expanded=True):
                                        st.success(f"Đã lưu {len(saved_objects)} đối tượng vào thư mục '{ref_dir}'")
                                        for filepath, class_name in saved_objects:
                                            st.write(f"- {class_name}: {os.path.basename(filepath)}")
                        else:
                            st.error("Vui lòng kiểm tra lại đường dẫn model.")
            
            elif image_path and os.path.exists(image_path):
                # Xử lý ảnh từ đường dẫn
                try:
                    # Đọc file
                    image = cv2.imread(image_path)
                    if image is None:
                        st.error(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
                    else:
                        # Chuyển đổi để hiển thị trong Streamlit
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, caption="Ảnh từ đường dẫn", use_container_width=True)
                        
                        if st.button("Nhận diện đối tượng", key="detect_path_image"):
                            if model_data:
                                with st.spinner("Đang xử lý..."):
                                    # Xử lý ảnh sử dụng hàm từ run_yolo.py
                                    result_img, detections = process_image_wrapper(
                                        image, 
                                        model_data, 
                                        conf_threshold, 
                                        input_size, 
                                        apply_nms, 
                                        nms_threshold,
                                        center_distance_threshold,
                                        debug=debug_mode
                                    )
                                    
                                    # Cắt và lưu các đối tượng đã phát hiện vào thư mục ref nếu được yêu cầu
                                    saved_objects = []
                                    if save_objects and detections:
                                        saved_objects = crop_and_save_objects(
                                            image, 
                                            detections, 
                                            model_data, 
                                            ref_dir=ref_dir, 
                                            min_size=min_object_size,
                                            input_size=input_size
                                        )
                                    
                                    # Chuyển kết quả về RGB để hiển thị đúng
                                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                    st.image(result_rgb, caption=f"Kết quả nhận diện: {len(detections)} đối tượng", use_container_width=True)
                                    
                                    # Hiển thị thông tin các đối tượng phát hiện được
                                    if detections:
                                        with st.expander("Chi tiết các đối tượng phát hiện", expanded=debug_mode):
                                            for i, det in enumerate(detections):
                                                try:
                                                    if det is None:
                                                        continue
                                                    
                                                    if len(det) == 4:
                                                        cls_id, obj_id, conf, _ = det
                                                    else:
                                                        cls_id, conf, _ = det
                                                        obj_id = "-"
                                                    
                                                    class_names = model_data.get("classes", {})
                                                    if isinstance(class_names, dict):
                                                        class_name = class_names.get(cls_id, f"Class {cls_id}")
                                                    elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                                                        class_name = class_names[cls_id]
                                                    else:
                                                        class_name = f"Class {cls_id}"
                                                    
                                                    st.write(f"{i+1}. {class_name} | ID:{obj_id} | {conf:.2f}")
                                                except Exception as e:
                                                    st.write(f"Lỗi khi hiển thị chi tiết detection: {str(e)}")
                                                    continue
                                    
                                    # Hiển thị thông tin về các đối tượng đã lưu
                                    if saved_objects:
                                        with st.expander("Đối tượng đã lưu vào thư mục ref", expanded=True):
                                            st.success(f"Đã lưu {len(saved_objects)} đối tượng vào thư mục '{ref_dir}'")
                                            for filepath, class_name in saved_objects:
                                                st.write(f"- {class_name}: {os.path.basename(filepath)}")
                            else:
                                st.error("Vui lòng kiểm tra lại đường dẫn model.")
                except Exception as e:
                    st.error(f"Lỗi khi đọc ảnh: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Tab 2: Xử lý video
        with tab2:
            st.header("Nhận diện đối tượng trong video")
            
            video_path = st.text_input("Nhập đường dẫn đến file video", key="video_path_input")
            uploaded_video = st.file_uploader("Hoặc tải lên video", type=["mp4", "avi", "mov"], key="video_uploader")
            
            if uploaded_video is not None:
                # Lưu video tải lên vào tệp tạm thời
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_video.read())
                video_path = temp_file.name
                temp_file.close()
                
                st.video(video_path)
            
            if video_path and os.path.exists(video_path):
                if st.button("Xử lý video", key="process_video_button"):
                    if model_data:
                        with st.spinner("Đang xử lý video..."):
                            output_video = process_video(
                                video_path, 
                                model_data, 
                                conf_threshold, 
                                input_size, 
                                apply_nms, 
                                nms_threshold,
                                center_distance_threshold,
                                buffer_size,
                                iou_threshold,
                                stability_threshold
                            )
                        
                        if output_video:
                            st.success("Xử lý video hoàn tất!")
                            st.video(output_video)
                            
                            # Tạo nút tải xuống
                            with open(output_video, 'rb') as f:
                                video_bytes = f.read()
                            
                            st.download_button(
                                label="Tải xuống video kết quả",
                                data=video_bytes,
                                file_name="yolo_result.mp4",
                                mime="video/mp4",
                                key="download_video_button"
                            )
                    else:
                        st.error("Vui lòng kiểm tra lại đường dẫn model.")
        
        # Tab 3: Xử lý camera
        with tab3:
            st.header("Nhận diện đối tượng qua camera")
            
            camera_option = st.selectbox(
                "Chọn nguồn camera",
                ["Webcam", "DroidCam"],
                key="camera_source_select"
            )
            
            if camera_option == "DroidCam":
                droidcam_ip = st.text_input("Địa chỉ IP DroidCam", "10.229.161.17", key="droidcam_ip_input")
                droidcam_port = st.text_input("Cổng DroidCam", "4747", key="droidcam_port_input")
                camera_url = f"http://{droidcam_ip}:{droidcam_port}/video"
            else:
                camera_url = 0  # Webcam mặc định
            
            if st.button("Bắt đầu camera", key="start_camera_button"):
                if model_data:
                    # Placeholder cho khung hình camera
                    frame_placeholder = st.empty()
                    stop_button_placeholder = st.empty()
                    stop_button = stop_button_placeholder.button("Dừng camera", key="unique_stop_camera_button")
                    
                    try:
                        connection_successful = False
                        # Tối ưu tùy chọn VideoCapture cho DroidCam
                        if camera_option == "DroidCam":
                            cap, connection_status = optimize_droidcam_connection(droidcam_ip, droidcam_port)
                            if cap is None:
                                st.error(connection_status)
                                st.info("Đảm bảo rằng ứng dụng DroidCam đã được khởi động trên điện thoại và điện thoại kết nối cùng mạng với máy tính.")
                            else:
                                st.success(connection_status)
                                connection_successful = True
                        else:
                            # Sử dụng webcam thông thường
                            cap = cv2.VideoCapture(camera_url)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            if not cap.isOpened():
                                st.error(f"Không thể kết nối webcam tại {camera_url}")
                            else:
                                # Kiểm tra đọc frame đầu tiên để xác nhận kết nối ổn định
                                ret, test_frame = cap.read()
                                if not ret or test_frame is None:
                                    st.error(f"Kết nối với webcam thành công nhưng không đọc được dữ liệu. Vui lòng kiểm tra lại camera.")
                                    cap.release()
                                else:
                                    st.success("Đã kết nối webcam thành công!")
                                    connection_successful = True
                            
                        # Chỉ tiếp tục nếu kết nối thành công
                        if connection_successful:
                            # Tối ưu buffer size cho tracker - nhỏ hơn để giảm độ trễ
                            camera_tracker = create_object_tracker(
                                buffer_size=5,  # Giảm từ 10 xuống 5 để giảm độ trễ
                                iou_threshold=iou_threshold,
                                stability_threshold=stability_threshold,
                                confidence_threshold=conf_threshold
                            )
                            
                            st.info(f"Xử lý camera với: Buffer=5, IoU={iou_threshold}, Stability={stability_threshold}")
                            
                            # Sử dụng session_state để theo dõi trạng thái dừng
                            if "camera_running" not in st.session_state:
                                st.session_state.camera_running = True
                                
                            # Biến đếm frames để skip một số frame nếu cần
                            frame_count = 0
                            
                            # Biến thời gian để theo dõi và tối ưu FPS
                            last_time = time.time()
                                
                            while st.session_state.camera_running and not stop_button:
                                # Đọc frame từ camera
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    st.error("Không thể đọc frame từ camera")
                                    st.session_state.camera_running = False
                                    break
                                
                                # Xử lý mỗi frame thứ hai để giảm tải hệ thống (tăng tốc độ)
                                frame_count += 1
                                if frame_count % 2 != 0 and frame_count > 1:  # Bỏ qua một số frame
                                    continue
                                
                                try:
                                    # Xử lý frame với tracker
                                    result_img, detections = process_camera_frame_wrapper(
                                        frame, 
                                        model_data, 
                                        conf_threshold, 
                                        input_size, 
                                        apply_nms, 
                                        nms_threshold,
                                        center_distance_threshold,
                                        camera_tracker
                                    )
                                    
                                    # Tính và hiển thị FPS
                                    current_time = time.time()
                                    fps = 1.0 / (current_time - last_time)
                                    last_time = current_time
                                    
                                    # Thêm FPS lên hình ảnh
                                    if result_img is not None:
                                        cv2.putText(result_img, f"FPS: {fps:.1f}", (10, 70), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        
                                        # Chuyển đổi từ BGR sang RGB để hiển thị
                                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                        
                                        # Hiển thị frame
                                        frame_placeholder.image(result_img_rgb, channels="RGB", use_container_width=True)
                                    else:
                                        # Hiển thị frame gốc nếu process_camera_frame_wrapper trả về None
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                        st.warning("Không thể xử lý frame - hiển thị frame gốc")
                                except Exception as e:
                                    st.error(f"Lỗi khi xử lý frame: {str(e)}")
                                    if debug_mode:
                                        import traceback
                                        st.code(traceback.format_exc())
                                    # Hiển thị frame gốc khi có lỗi
                                    try:
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                    except:
                                        pass
                                    # Tiếp tục vòng lặp thay vì thoát
                                    pass
                                
                                # Giảm thời gian chờ để tăng hiệu suất (từ 0.1 xuống 0.01)
                                time.sleep(0.01)
                                
                                # Thay vì liên tục kiểm tra nút dừng cũ, tạo một widget mới để kiểm tra
                                stop_key = str(time.time())
                                if stop_button_placeholder.button("Dừng", key=stop_key):
                                    st.session_state.camera_running = False
                                    break
                                
                            # Đảm bảo giải phóng camera khi xong
                            cap.release()
                            st.session_state.camera_running = False
                            st.write("Đã dừng camera")
                        else:
                            st.warning("Không thể khởi chạy camera. Vui lòng kiểm tra lại kết nối.")
                    
                    except Exception as e:
                        st.session_state.camera_running = False
                        st.error(f"Lỗi khi xử lý camera: {e}")
                        if debug_mode:
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.error("Vui lòng kiểm tra lại đường dẫn model.")

        # Tab 4: Theo dõi tồn kho
        with tab4:
            st.header("Theo dõi tồn kho sản phẩm")
            
            # Chọn nguồn đầu vào
            st.subheader("Cấu hình nguồn")
            inventory_input = st.radio(
                "Chọn nguồn để theo dõi tồn kho",
                ["DroidCam", "Webcam"],
                key="inventory_source_select"
            )
            
            # Cấu hình nguồn
            if inventory_input == "DroidCam":
                droidcam_ip = st.text_input("Địa chỉ IP DroidCam", "10.229.161.17", key="inventory_droidcam_ip")
                droidcam_port = st.text_input("Cổng DroidCam", "4747", key="inventory_droidcam_port")
                camera_url = f"http://{droidcam_ip}:{droidcam_port}/video"
            else:
                camera_url = 0  # Webcam mặc định
                
            # Tải dữ liệu tồn kho
            st.subheader("Dữ liệu tồn kho")
            
            # Hiển thị dữ liệu kế hoạch
            if inventory_data:
                st.success(f"Đã tải dữ liệu tồn kho từ {inventory_file}: {len(inventory_data)} sản phẩm")
                
                # Hiển thị bảng dữ liệu kế hoạch
                plan_data = []
                for product_class, info in inventory_data.items():
                    plan_data.append({
                        "Sản phẩm": product_class,
                        "Số lượng dự kiến": info["expected"],
                        "Giá": f"{info['price']:,} VND"
                    })
                    
                st.dataframe(plan_data)
            else:
                st.warning(f"Không tìm thấy dữ liệu tồn kho. Vui lòng kiểm tra file {inventory_file}")
                
            # Nút bắt đầu theo dõi tồn kho
            if st.button("Bắt đầu theo dõi tồn kho", key="start_inventory_tracking"):
                if not model_data:
                    st.error("Vui lòng kiểm tra lại đường dẫn model.")
                elif not inventory_data:
                    st.error(f"Không tìm thấy dữ liệu tồn kho. Vui lòng kiểm tra file {inventory_file}")
                else:
                    # Placeholder cho khung hình camera và bảng theo dõi
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        frame_placeholder = st.empty()
                        detection_info = st.empty()
                    
                    with col2:
                        inventory_table = st.empty()
                        inventory_summary = st.empty()
                        
                    stop_button_placeholder = st.empty()
                    stop_button = stop_button_placeholder.button("Dừng theo dõi", key="stop_inventory_tracking")
                    
                    try:
                        connection_successful = False
                        
                        # Kết nối với camera
                        if inventory_input == "DroidCam":
                            cap, connection_status = optimize_droidcam_connection(droidcam_ip, droidcam_port)
                            if cap is None:
                                st.error(connection_status)
                            else:
                                st.success(connection_status)
                                connection_successful = True
                        else:
                            cap = cv2.VideoCapture(camera_url)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            if not cap.isOpened():
                                st.error(f"Không thể kết nối webcam tại {camera_url}")
                            else:
                                # Kiểm tra đọc frame đầu tiên
                                ret, test_frame = cap.read()
                                if not ret or test_frame is None:
                                    st.error("Kết nối với webcam thành công nhưng không đọc được dữ liệu.")
                                    cap.release()
                                else:
                                    st.success("Đã kết nối webcam thành công!")
                                    connection_successful = True
                        
                        # Theo dõi tồn kho nếu kết nối thành công
                        if connection_successful:
                            # Khởi tạo tracker
                            inventory_tracker = create_object_tracker(
                                buffer_size=buffer_size,
                                iou_threshold=iou_threshold,
                                stability_threshold=stability_threshold,
                                confidence_threshold=conf_threshold
                            )
                            
                            # Biến để theo dõi trạng thái
                            st.session_state.inventory_running = True
                            frame_count = 0
                            last_time = time.time()
                            
                            # Biến theo dõi cập nhật
                            last_update_time = time.time()
                            update_interval = 1.0  # Cập nhật bảng mỗi 1 giây
                            
                            while st.session_state.inventory_running and not stop_button:
                                # Đọc frame từ camera
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    st.error("Không thể đọc frame từ camera")
                                    st.session_state.inventory_running = False
                                    break
                                
                                # Xử lý mỗi 2 frame để tăng tốc độ
                                frame_count += 1
                                if frame_count % 2 != 0 and frame_count > 1:
                                    continue
                                
                                try:
                                    # Xử lý frame với tracker
                                    result_img, detections = process_camera_frame_wrapper(
                                        frame, 
                                        model_data, 
                                        conf_threshold, 
                                        input_size, 
                                        apply_nms, 
                                        nms_threshold,
                                        center_distance_threshold,
                                        inventory_tracker
                                    )
                                    
                                    # Tính FPS
                                    current_time = time.time()
                                    fps = 1.0 / (current_time - last_time)
                                    last_time = current_time
                                    
                                    # Thêm FPS lên hình ảnh
                                    if result_img is not None:
                                        cv2.putText(result_img, f"FPS: {fps:.1f}", (10, 70), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        
                                        # Chuyển đổi từ BGR sang RGB để hiển thị
                                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(result_img_rgb, channels="RGB", use_container_width=True)
                                        
                                        # Cập nhật thông tin tồn kho mỗi update_interval giây
                                        if current_time - last_update_time > update_interval:
                                            # Lấy kết quả theo dõi ổn định
                                            stable_objects = inventory_tracker.get_stable_objects()
                                            
                                            # Đếm sản phẩm theo loại
                                            product_counts = count_products_from_detections(stable_objects, model_data)
                                            
                                            # So sánh với dữ liệu dự kiến
                                            inventory_status = compare_inventory(product_counts, inventory_data)
                                            
                                            # Hiển thị kết quả theo dõi
                                            inventory_table_data = []
                                            alerts = []
                                            total_expected = 0
                                            total_actual = 0
                                            total_difference = 0
                                            
                                            for product_class, status in inventory_status.items():
                                                # Định dạng giá tiền
                                                price_formatted = f"{status['price']:,} VND"
                                                
                                                # Thêm vào bảng hiển thị
                                                inventory_table_data.append({
                                                    "Sản phẩm": product_class,
                                                    "Kế hoạch": status["expected"],
                                                    "Thực tế": status["actual"],
                                                    "Chênh lệch": status["difference"],
                                                    "Trạng thái": status["status"],
                                                    "Giá": price_formatted
                                                })
                                                
                                                # Thống kê
                                                total_expected += status["expected"]
                                                total_actual += status["actual"]
                                                total_difference += status["difference"]
                                                
                                                # Thêm vào danh sách cảnh báo nếu cần
                                                if status["alert"]:
                                                    alerts.append(f"{product_class}: {status['status']} (còn {status['actual']}/{status['expected']})")
                                            
                                            # Hiển thị bảng theo dõi tồn kho
                                            inventory_table.dataframe(inventory_table_data)
                                            
                                            # Hiển thị tóm tắt
                                            summary_md = f"""
                                            ### Tóm tắt tồn kho
                                            - **Tổng sản phẩm dự kiến:** {total_expected}
                                            - **Tổng sản phẩm thực tế:** {total_actual}
                                            - **Chênh lệch:** {total_difference}
                                            
                                            ### Cảnh báo
                                            """
                                            
                                            if alerts:
                                                for alert in alerts:
                                                    summary_md += f"- 🔴 **{alert}**\n"
                                            else:
                                                summary_md += "- ✅ Không có cảnh báo.\n"
                                            
                                            inventory_summary.markdown(summary_md)
                                            
                                            # Hiển thị thông tin chi tiết các đối tượng phát hiện
                                            if debug_mode:
                                                detection_info.write(f"Phát hiện {len(stable_objects)} đối tượng ổn định")
                                            
                                            # Cập nhật thời gian
                                            last_update_time = current_time
                                    else:
                                        # Hiển thị frame gốc nếu process_camera_frame_wrapper trả về None
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                        st.warning("Không thể xử lý frame - hiển thị frame gốc")
                                except Exception as e:
                                    st.error(f"Lỗi khi xử lý frame: {str(e)}")
                                    if debug_mode:
                                        import traceback
                                        st.code(traceback.format_exc())
                                    # Hiển thị frame gốc khi có lỗi
                                    try:
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                    except:
                                        pass
                                
                                # Giảm thời gian chờ
                                time.sleep(0.01)
                                
                                # Kiểm tra nút dừng
                                stop_key = str(time.time())
                                if stop_button_placeholder.button("Dừng theo dõi", key=stop_key):
                                    st.session_state.inventory_running = False
                                    break
                            
                            # Giải phóng camera
                            cap.release()
                            st.session_state.inventory_running = False
                            st.write("Đã dừng theo dõi tồn kho")
                            
                    except Exception as e:
                        st.error(f"Lỗi khi theo dõi tồn kho: {e}")
                        if debug_mode:
                            import traceback
                            st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Lỗi chung khi chạy ứng dụng: {str(e)}")
        if st.session_state.get('debug_mode', False):
            import traceback
            st.code(traceback.format_exc())
        st.info("Ứng dụng gặp sự cố. Vui lòng làm mới trang (F5) và thử lại.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Lỗi chung: {str(e)}")
        st.info("Vui lòng làm mới trang và thử lại.")
        import traceback
        print(f"Lỗi ứng dụng: {traceback.format_exc()}") 