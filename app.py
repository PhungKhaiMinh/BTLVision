


import os
import sys
import warnings
import json


warnings.filterwarnings('ignore')


os.environ['STREAMLIT_SERVER_FILE_WATCHDOG'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCH_POLL'] = 'false'


import torch


import streamlit as st
import cv2
import numpy as np
import time
import glob
from PIL import Image
import io
import tempfile
import datetime  


def crop_and_save_objects(image, detections, model_data, ref_dir="ref", min_size=50, input_size=640):
    
    
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_objects = []
    
    
    img_h, img_w = image.shape[:2]
    
    
    scale_x = img_w / input_size
    scale_y = img_h / input_size
    
    
    for i, (cls_id, conf, box) in enumerate(detections):
        
        x, y, w, h = box
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        
        
        if w < min_size or h < min_size:
            continue
        
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        
        cropped = image[y:y+h, x:x+w]
        
        
        class_names = model_data.get("classes", {})
        if isinstance(class_names, dict):
            class_name = class_names.get(cls_id, f"Class_{cls_id}")
        elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"Class_{cls_id}"
        
        
        filename = f"{class_name}_{timestamp}_{i}.jpg"
        filepath = os.path.join(ref_dir, filename)
        
        
        cv2.imwrite(filepath, cropped)
        
        saved_objects.append((filepath, class_name))
    
    return saved_objects


def load_model_wrapper(model_path, use_optimization=False, opt_format="engine", half=False, verbose=False):
    
    from run_yolo import load_model
    return load_model(model_path, use_optimization, opt_format, half, verbose)

def process_image_wrapper(image, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, 
                         nms_threshold=0.5, center_distance_threshold=0.2, return_image_with_boxes=True, 
                         debug=False):
    
    try:
        if image is None:
            return None, []
            
        
        orig_h, orig_w = image.shape[:2]
        
        
        
        if image.shape[0] != input_size or image.shape[1] != input_size:
            resized_img = cv2.resize(image, (input_size, input_size))
        else:
            resized_img = image
            
        
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size
        
        from run_yolo import process_image
        result = process_image(resized_img, model_data, conf_threshold, input_size, apply_nms, nms_threshold,
                       center_distance_threshold, return_image_with_boxes, debug)
        
        
        if isinstance(result, tuple) and len(result) == 2:
            img, detections = result
            if detections is None:
                detections = []
                
            
            adjusted_detections = []
            for det in detections:
                if len(det) == 3:  
                    cls_id, conf, box = det
                    x, y, w, h = box
                    
                    x_orig = x * scale_x
                    y_orig = y * scale_y
                    w_orig = w * scale_x
                    h_orig = h * scale_y
                    
                    x_orig = max(0, min(x_orig, orig_w-1))
                    y_orig = max(0, min(y_orig, orig_h-1))
                    w_orig = min(w_orig, orig_w-x_orig)
                    h_orig = min(h_orig, orig_h-y_orig)
                    adjusted_detections.append((cls_id, conf, [x_orig, y_orig, w_orig, h_orig]))
                else:
                    adjusted_detections.append(det)
            
            
            if return_image_with_boxes and img is None:
                
                img = image.copy()
                
                
                for cls_id, conf, box in adjusted_detections:
                    
                    x, y, w, h = [int(v) for v in box]
                    
                    
                    class_names = model_data.get('classes', {})
                    if isinstance(class_names, dict):
                        class_name = class_names.get(cls_id, f"Class {cls_id}")
                    elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                        class_name = class_names[cls_id]
                    else:
                        class_name = f"Class {cls_id}"
                    
                    
                    color = (0, 0, 255)  
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                    
                    
                    label = f"{class_name}: {conf:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
                    cv2.putText(img, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif return_image_with_boxes and img is not None and (img.shape[0] != orig_h or img.shape[1] != orig_w):
                
                img = cv2.resize(img, (orig_w, orig_h))
                
            return img, adjusted_detections
        else:
            
            print("Kết quả từ process_image không đúng định dạng")
            return None, []
    except Exception as e:
        print(f"Lỗi trong process_image_wrapper: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, []

def create_object_tracker(buffer_size=10, iou_threshold=0.5, stability_threshold=0.5, confidence_threshold=0.3):
    
    from run_yolo import ObjectTracker
    return ObjectTracker(buffer_size, iou_threshold, stability_threshold, confidence_threshold)

def custom_nms(detections, iou_threshold=0.5, same_class_only=True, center_distance_threshold=0.2):
    
    from run_yolo import apply_custom_nms
    return apply_custom_nms(detections, iou_threshold, same_class_only, center_distance_threshold)


def load_inventory_data(json_file="Data.json"):
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        
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
    
    class_names = model_data.get('classes', {})
    product_counts = {}
    
    for det in detections:
        if det is None:
            continue
            
        if len(det) == 4:
            cls_id, obj_id, conf, _ = det
        else:
            cls_id, conf, _ = det
            
        
        if isinstance(class_names, dict):
            class_name = class_names.get(cls_id, f"Class {cls_id}")
        elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"Class {cls_id}"
            
        
        if class_name in product_counts:
            product_counts[class_name] += 1
        else:
            product_counts[class_name] = 1
            
    return product_counts

def compare_inventory(actual_counts, expected_inventory):
    
    inventory_status = {}
    
    
    for product_class, info in expected_inventory.items():
        expected = info['expected']
        actual = actual_counts.get(product_class, 0)
        
        
        difference = actual - expected
        
        
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


st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")


try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
except Exception:
    pass

def get_video_frame(video_path, frame_number):
    
    cap = cv2.VideoCapture(video_path)
    
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    return frame

def process_video(video_path, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, nms_threshold=0.5, center_distance_threshold=0.2, 
                  buffer_size=10, iou_threshold=0.5, stability_threshold=0.6, debug_mode=False):
    
    
    tracker = create_object_tracker(
        buffer_size=buffer_size,
        iou_threshold=iou_threshold,
        stability_threshold=stability_threshold,
        confidence_threshold=conf_threshold
    )
    
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Không thể mở video: {video_path}")
        return None
    
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    
    progress_bar = st.progress(0)
    frame_text = st.empty()
    
    
    frame_idx = 0
    processed_frames = 0
    
    
    st.info(f"Xử lý video với: Buffer={buffer_size}, IoU={iou_threshold}, Stability={stability_threshold}, NMS IoU={nms_threshold}")
    
    
    last_time = time.time()
    fps_display = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        if frame_idx % 2 == 0:
            current_time = time.time()
            fps_display = 1.0 / (current_time - last_time) if current_time != last_time else 0
            last_time = current_time
                
            
            result_img, detections = process_camera_frame_wrapper(
                frame, 
                model_data, 
                conf_threshold, 
                input_size, 
                apply_nms, 
                nms_threshold,
                center_distance_threshold,
                tracker
            )
            
            
            if result_img is not None:
                cv2.putText(result_img, f"FPS: {fps_display:.1f}", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            
            writer.write(result_img)
            
            
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)
            frame_text.text(f"Xử lý frame: {frame_idx+1}/{total_frames} | FPS: {fps_display:.1f}")
            
            processed_frames += 1
        else:
            
            writer.write(frame)
            
        frame_idx += 1
    
    
    cap.release()
    writer.release()
    
    st.success(f"Đã xử lý {processed_frames} frames. Video được lưu tại: {output_path}")
    return output_path


def process_camera_frame(frame, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, nms_threshold=0.5, center_distance_threshold=0.2, tracker=None):
    
    return process_camera_frame_wrapper(frame, model_data, conf_threshold, input_size, apply_nms, nms_threshold, center_distance_threshold, tracker)

def process_camera_frame_wrapper(frame, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, nms_threshold=0.5, center_distance_threshold=0.2, tracker=None):
    
    if frame is None:
        return None, []
    
    
    height, width = frame.shape[:2]
    
    
    scale_x = width / input_size
    scale_y = height / input_size
    
    
    resized_frame = cv2.resize(frame, (input_size, input_size))
        
    
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
    
    
    if detections is None:
        detections = []
    
    
    if tracker:
        tracker.update(detections)
        objects_to_display = tracker.get_stable_objects()
    else:
        objects_to_display = detections
    
    
    if objects_to_display is None:
        objects_to_display = []
    
    
    result_img = frame.copy()
    
    
    class_names = model_data.get("classes", {})
    
    for det in objects_to_display:
        
        if det is None:
            continue
            
        
        try:
            if len(det) == 4:
                cls_id, obj_id, conf, box = det
            else:
                cls_id, conf, box = det
                obj_id = "-"
                
            
            if box is None:
                continue
                
            
            x, y, w, h = box
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
        
            
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = min(w, width-x)
            h = min(h, height-y)
        
            
            if isinstance(class_names, dict):
                class_name = class_names.get(cls_id, f"Class {cls_id}")
            elif isinstance(class_names, list) and 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = f"Class {cls_id}"
        
            
            color = (0, 0, 255)  
            
            
            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 3)
        
            
            label = f"{class_name} | ID:{obj_id} | {conf:.2f}"
        
            
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_img, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
            cv2.putText(result_img, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            
            print(f"Lỗi khi xử lý detection: {e}")
            continue
    
    
    info_text = f"Objects: {len(objects_to_display)}/{len(detections)}"
    
    
    cv2.putText(result_img, info_text, (10, 30), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_img, detections

def load_model_with_retry(model_path, use_optimization, opt_format, half_precision, debug_mode, max_retries=3):
    
    for attempt in range(max_retries):
        try:
            with st.spinner(f"Đang tải model... (lần thử {attempt+1}/{max_retries})"):
                
                if 'cached_model' in st.session_state and st.session_state.cached_model_path == model_path:
                    if debug_mode:
                        st.info(f"Sử dụng model đã cache {model_path}")
                    return st.session_state.cached_model
                
                
                is_cuda_available = torch.cuda.is_available()
                if is_cuda_available:
                    gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
                    
                    torch.backends.cudnn.benchmark = True
                    
                    if half_precision:
                        torch.set_default_tensor_type(torch.cuda.HalfTensor)
                else:
                    gpu_info = "CPU only"
                    
                if debug_mode:
                    st.info(f"Môi trường: {gpu_info}, PyTorch {torch.__version__}")
                
                
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
                
                
                if opt_format == "engine" and is_cuda_available:
                    gpu_name = torch.cuda.get_device_name(0).lower()
                    if "1050" in gpu_name or "gtx" in gpu_name:
                        st.warning(f"GPU {gpu_name} có thể không tương thích với TensorRT mới. Chuyển sang ONNX...")
                        opt_format = "onnx"
                
                
                model = load_model_wrapper(model_path, use_optimization, opt_format, half_precision, debug_mode)
                
                
                st.session_state.cached_model = model
                st.session_state.cached_model_path = model_path
                
                return model
        except Exception as e:
            error_msg = str(e)
            if debug_mode:
                import traceback
                st.code(traceback.format_exc())
            
            
            if "torch.classes" in error_msg:
                st.warning("Đang gặp lỗi với torch.classes. Thử với cấu hình khác...")
                if opt_format == "engine":
                    opt_format = "onnx"
                elif opt_format == "onnx":
                    use_optimization = False
            
            if attempt < max_retries - 1:
                st.warning(f"Lỗi khi tải model: {error_msg}. Thử lại...")
                time.sleep(1)  
            else:
                if use_optimization:
                    st.warning("Không thể tải model tối ưu. Thử với model PyTorch gốc...")
                    
                    try:
                        model = load_model_wrapper(model_path, False, None, False, debug_mode)
                        
                        st.session_state.cached_model = model
                        st.session_state.cached_model_path = model_path
                        return model
                    except Exception as e2:
                        st.error(f"Không thể tải model: {str(e2)}")
                        return None
                else:
                    st.error(f"Không thể tải model sau {max_retries} lần thử: {error_msg}")
                    return None
                    

def optimize_droidcam_connection(droidcam_ip, droidcam_port):
    
    camera_url = f"http://{droidcam_ip}:{droidcam_port}/video"
    
    
    if ":" in droidcam_ip and int(droidcam_port) > 4000:
        
        try:
            
            udp_url = f"udp://{droidcam_ip}:{int(droidcam_port)-1}"
            udp_cap = cv2.VideoCapture(udp_url)
            if udp_cap.isOpened():
                
                udp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                udp_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                ret, frame = udp_cap.read()
                if ret and frame is not None:
                    print("Đã kết nối qua UDP")
                    return udp_cap, "Đã kết nối DroidCam qua UDP (độ trễ thấp)"
                else:
                    udp_cap.release()
        except:
            pass
    
    try:
        
        cap = cv2.VideoCapture(camera_url)
        
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  
        
        
        if not cap.isOpened():
            
            alt_url = f"http://{droidcam_ip}:{droidcam_port}/mjpegfeed?640x480"
            cap = cv2.VideoCapture(alt_url)
            if not cap.isOpened():
                return None, f"Không thể kết nối với DroidCam tại {camera_url}"
        
        
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None, "Kết nối với DroidCam thành công nhưng không đọc được dữ liệu"
        
        
        
        
        
        
        cap.set(cv2.CAP_PROP_FPS, 30)  
        
        
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
    
    try:
        st.title("YOLOv8 Object Detection - Streamlit UI")
        
        
        st.sidebar.header("Cấu hình")
        
        
        model_path = st.sidebar.text_input("Đường dẫn đến model YOLOv8", "best(3)-can-n.pt")
        
        
        debug_mode = st.sidebar.checkbox("Chế độ debug", False)
        st.session_state.debug_mode = debug_mode
        
        
        with st.sidebar.expander("Cấu hình nâng cao", expanded=True):
            conf_threshold = st.slider("Ngưỡng tin cậy", 0.0, 1.0, 0.25, 0.01)
            input_size = st.select_slider("Kích thước đầu vào", options=[320, 416, 512, 640, 768, 1024], value=640)
            use_optimization = st.checkbox("Sử dụng tối ưu hóa", True)
            
            opt_format = st.selectbox("Định dạng tối ưu", ["onnx", "engine"], index=0)
            half_precision = st.checkbox("Sử dụng FP16 (half precision)", True)
            
            
            apply_nms = st.checkbox("Áp dụng NMS tùy chỉnh", True)
            nms_threshold = st.slider("Ngưỡng IoU cho NMS", 0.1, 0.9, 0.5, 0.1)
            center_distance_threshold = st.slider("Ngưỡng khoảng cách tâm cho NMS", 0.1, 0.5, 0.2, 0.05)
            
            
            st.divider()
            st.subheader("Tùy chọn lưu đối tượng")
            save_objects = st.checkbox("Tự động lưu vật thể phát hiện vào thư mục ref", True)
            ref_dir = st.text_input("Thư mục lưu vật thể", "ref")
            min_object_size = st.slider("Kích thước tối thiểu của vật thể (pixel)", 20, 200, 50)
            
            
            st.divider()
            st.subheader("Tham số theo dõi đối tượng")
            buffer_size = st.slider("Buffer size (số frame để theo dõi)", 5, 20, 10, 1)
            iou_threshold = st.slider("Ngưỡng IoU cho việc khớp đối tượng", 0.3, 0.7, 0.5, 0.05)
            stability_threshold = st.slider("Ngưỡng ổn định để hiển thị đối tượng", 0.3, 0.9, 0.6, 0.05)
            
            
            st.divider()
            st.subheader("Cấu hình theo dõi tồn kho")
            inventory_file = st.text_input("File dữ liệu tồn kho", "Data.json")
            low_stock_threshold = st.slider("Ngưỡng cảnh báo tồn kho thấp (%)", 0, 100, 30, 5)
        
        
        tab1, tab2, tab3, tab4 = st.tabs(["Ảnh", "Video", "Camera", "Theo dõi tồn kho"])
        
        
        model_data = None
        if os.path.exists(model_path):
            model_placeholder = st.empty()
            with model_placeholder.container():
                model_data = load_model_with_retry(model_path, use_optimization, opt_format, half_precision, debug_mode)
            model_placeholder.empty()
        else:
            if model_path:
                st.error(f"Không tìm thấy model tại: {model_path}")
                
        
        inventory_data = {}
        if os.path.exists(inventory_file):
            inventory_data = load_inventory_data(inventory_file)
        
        
        with tab1:
            st.header("Nhận diện đối tượng trong ảnh")
            
            
            uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader")
            image_path = st.text_input("Hoặc nhập đường dẫn đến ảnh", key="image_path_input")
            
            if uploaded_file is not None:
                
                
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                st.image(image_rgb, caption="Ảnh đã tải lên", use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Nhận diện đối tượng", key="detect_uploaded_image"):
                        if model_data:
                            with st.spinner("Đang xử lý..."):
                                
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
                                
                                
                                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                st.image(result_rgb, caption=f"Kết quả nhận diện: {len(detections)} đối tượng", use_container_width=True)
                                
                                
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
                                
                                
                                if saved_objects:
                                    with st.expander("Đối tượng đã lưu vào thư mục ref", expanded=True):
                                        st.success(f"Đã lưu {len(saved_objects)} đối tượng vào thư mục '{ref_dir}'")
                                        for filepath, class_name in saved_objects:
                                            st.write(f"- {class_name}: {os.path.basename(filepath)}")
                        else:
                            st.error("Vui lòng kiểm tra lại đường dẫn model.")
            
            elif image_path and os.path.exists(image_path):
                
                try:
                    
                    image = cv2.imread(image_path)
                    if image is None:
                        st.error(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
                    else:
                        
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, caption="Ảnh từ đường dẫn", use_container_width=True)
                        
                        if st.button("Nhận diện đối tượng", key="detect_path_image"):
                            if model_data:
                                with st.spinner("Đang xử lý..."):
                                    
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
                                    
                                    
                                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                    st.image(result_rgb, caption=f"Kết quả nhận diện: {len(detections)} đối tượng", use_container_width=True)
                                    
                                    
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

        
        with tab2:
            st.header("Nhận diện đối tượng trong video")
            
            video_path = st.text_input("Nhập đường dẫn đến file video", key="video_path_input")
            uploaded_video = st.file_uploader("Hoặc tải lên video", type=["mp4", "avi", "mov"], key="video_uploader")
            
            if uploaded_video is not None:
                
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
                                stability_threshold,
                                debug_mode
                            )
                        
                        if output_video:
                            st.success("Xử lý video hoàn tất!")
                            st.video(output_video)
                            
                            
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
        
        
        with tab3:
            st.header("Nhận diện đối tượng qua camera")
            
            camera_option = st.selectbox(
                "Chọn nguồn camera",
                ["Webcam", "DroidCam"],
                key="camera_source_select"
            )
            
            if camera_option == "DroidCam":
                droidcam_ip = st.text_input("Địa chỉ IP DroidCam", "192.168.1.8", key="droidcam_ip_input")
                droidcam_port = st.text_input("Cổng DroidCam", "4747", key="droidcam_port_input")
                camera_url = f"http://{droidcam_ip}:{droidcam_port}/video"
            else:
                camera_url = 0  
            
            if st.button("Bắt đầu camera", key="start_camera_button"):
                if model_data:
                    
                    frame_placeholder = st.empty()
                    stop_button_placeholder = st.empty()
                    stop_button = stop_button_placeholder.button("Dừng camera", key="unique_stop_camera_button")
                    
                    try:
                        connection_successful = False
                        
                        if camera_option == "DroidCam":
                            cap, connection_status = optimize_droidcam_connection(droidcam_ip, droidcam_port)
                            if cap is None:
                                st.error(connection_status)
                                st.info("Đảm bảo rằng ứng dụng DroidCam đã được khởi động trên điện thoại và điện thoại kết nối cùng mạng với máy tính.")
                            else:
                                st.success(connection_status)
                                connection_successful = True
                        else:
                            
                            cap = cv2.VideoCapture(camera_url)
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            if not cap.isOpened():
                                st.error(f"Không thể kết nối webcam tại {camera_url}")
                            else:
                                
                                ret, test_frame = cap.read()
                                if not ret or test_frame is None:
                                    st.error(f"Kết nối với webcam thành công nhưng không đọc được dữ liệu. Vui lòng kiểm tra lại camera.")
                                    cap.release()
                                else:
                                    st.success("Đã kết nối webcam thành công!")
                                    connection_successful = True
                            
                        
                        if connection_successful:
                            
                            camera_tracker = create_object_tracker(
                                buffer_size=5,  
                                iou_threshold=iou_threshold,
                                stability_threshold=stability_threshold,
                                confidence_threshold=conf_threshold
                            )
                            
                            st.info(f"Xử lý camera với: Buffer=5, IoU={iou_threshold}, Stability={stability_threshold}")
                            
                            
                            if "camera_running" not in st.session_state:
                                st.session_state.camera_running = True
                                
                            
                            frame_count = 0
                            
                            
                            last_time = time.time()
                                
                            while st.session_state.camera_running and not stop_button:
                                
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    st.error("Không thể đọc frame từ camera")
                                    st.session_state.camera_running = False
                                    break
                                
                                
                                frame_count += 1
                                if frame_count % 2 != 0 and frame_count > 1:  
                                    continue
                                
                                try:
                                    
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
                                    
                                    
                                    current_time = time.time()
                                    fps = 1.0 / (current_time - last_time)
                                    last_time = current_time
                                    
                                    
                                    if result_img is not None:
                                        cv2.putText(result_img, f"FPS: {fps:.1f}", (10, 70), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        
                                        
                                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                        
                                        
                                        frame_placeholder.image(result_img_rgb, channels="RGB", use_container_width=True)
                                    else:
                                        
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                        st.warning("Không thể xử lý frame - hiển thị frame gốc")
                                except Exception as e:
                                    st.error(f"Lỗi khi xử lý frame: {str(e)}")
                                    if debug_mode:
                                        import traceback
                                        st.code(traceback.format_exc())
                                    
                                    try:
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                    except:
                                        pass
                                    
                                    pass
                                
                                
                                time.sleep(0.01)
                                
                                
                                stop_key = str(time.time())
                                if stop_button_placeholder.button("Dừng", key=stop_key):
                                    st.session_state.camera_running = False
                                    break
                                
                            
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

        
        with tab4:
            st.header("Theo dõi tồn kho sản phẩm")
            
            
            st.subheader("Cấu hình nguồn")
            inventory_input = st.radio(
                "Chọn nguồn để theo dõi tồn kho",
                ["DroidCam", "Webcam"],
                key="inventory_source_select"
            )
            
            
            if inventory_input == "DroidCam":
                droidcam_ip = st.text_input("Địa chỉ IP DroidCam", "192.168.1.8", key="inventory_droidcam_ip")
                droidcam_port = st.text_input("Cổng DroidCam", "4747", key="inventory_droidcam_port")
                camera_url = f"http://{droidcam_ip}:{droidcam_port}/video"
            else:
                camera_url = 0  
                
            
            st.subheader("Dữ liệu tồn kho")
            
            
            if inventory_data:
                st.success(f"Đã tải dữ liệu tồn kho từ {inventory_file}: {len(inventory_data)} sản phẩm")
                
                
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
                
            
            if st.button("Bắt đầu theo dõi tồn kho", key="start_inventory_tracking"):
                if not model_data:
                    st.error("Vui lòng kiểm tra lại đường dẫn model.")
                elif not inventory_data:
                    st.error(f"Không tìm thấy dữ liệu tồn kho. Vui lòng kiểm tra file {inventory_file}")
                else:
                    
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
                                
                                ret, test_frame = cap.read()
                                if not ret or test_frame is None:
                                    st.error("Kết nối với webcam thành công nhưng không đọc được dữ liệu.")
                                    cap.release()
                                else:
                                    st.success("Đã kết nối webcam thành công!")
                                    connection_successful = True
                        
                        
                        if connection_successful:
                            
                            inventory_tracker = create_object_tracker(
                                buffer_size=buffer_size,
                                iou_threshold=iou_threshold,
                                stability_threshold=stability_threshold,
                                confidence_threshold=conf_threshold
                            )
                            
                            
                            st.session_state.inventory_running = True
                            frame_count = 0
                            last_time = time.time()
                            
                            
                            last_update_time = time.time()
                            update_interval = 1.0  
                            
                            while st.session_state.inventory_running and not stop_button:
                                
                                ret, frame = cap.read()
                                if not ret or frame is None:
                                    st.error("Không thể đọc frame từ camera")
                                    st.session_state.inventory_running = False
                                    break
                                
                                
                                frame_count += 1
                                if frame_count % 2 != 0 and frame_count > 1:
                                    continue
                                
                                try:
                                    
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
                                    
                                    
                                    current_time = time.time()
                                    fps = 1.0 / (current_time - last_time)
                                    last_time = current_time
                                    
                                    
                                    if result_img is not None:
                                        cv2.putText(result_img, f"FPS: {fps:.1f}", (10, 70), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        
                                        
                                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(result_img_rgb, channels="RGB", use_container_width=True)
                                        
                                        
                                        if current_time - last_update_time > update_interval:
                                            
                                            stable_objects = inventory_tracker.get_stable_objects()
                                            
                                            
                                            product_counts = count_products_from_detections(stable_objects, model_data)
                                            
                                            
                                            inventory_status = compare_inventory(product_counts, inventory_data)
                                            
                                            
                                            inventory_table_data = []
                                            alerts = []
                                            total_expected = 0
                                            total_actual = 0
                                            total_difference = 0
                                            
                                            for product_class, status in inventory_status.items():
                                                
                                                price_formatted = f"{status['price']:,} VND"
                                                
                                                
                                                inventory_table_data.append({
                                                    "Sản phẩm": product_class,
                                                    "Kế hoạch": status["expected"],
                                                    "Thực tế": status["actual"],
                                                    "Chênh lệch": status["difference"],
                                                    "Trạng thái": status["status"],
                                                    "Giá": price_formatted
                                                })
                                                
                                                
                                                total_expected += status["expected"]
                                                total_actual += status["actual"]
                                                total_difference += status["difference"]
                                                
                                                
                                                if status["alert"]:
                                                    alerts.append(f"{product_class}: {status['status']} (còn {status['actual']}/{status['expected']})")
                                            
                                            
                                            inventory_table.dataframe(inventory_table_data)
                                            
                                            
                                            summary_md = f
                                            
                                            
                                            all_products = []
                                            for product_class, status in inventory_status.items():
                                                status_icon = "🔴" if status["alert"] else "🟢"
                                                product_info = f"{status_icon} **{product_class}:** {status['status']} (còn {status['actual']}/{status['expected']})"
                                                all_products.append((product_class, status["alert"], product_info))
                                            
                                            
                                            all_products.sort(key=lambda x: (not x[1], x[0]))
                                            
                                            
                                            for _, _, product_info in all_products:
                                                summary_md += f"- {product_info}\n"
                                            
                                            inventory_summary.markdown(summary_md)
                                            
                                            
                                            if debug_mode:
                                                detection_info.write(f"Phát hiện {len(stable_objects)} đối tượng ổn định")
                                            
                                            
                                            last_update_time = current_time
                                    else:
                                        
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                        st.warning("Không thể xử lý frame - hiển thị frame gốc")
                                except Exception as e:
                                    st.error(f"Lỗi khi xử lý frame: {str(e)}")
                                    if debug_mode:
                                        import traceback
                                        st.code(traceback.format_exc())
                                    
                                    try:
                                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                                    except:
                                        pass
                                
                                
                                time.sleep(0.01)
                                
                                
                                stop_key = str(time.time())
                                if stop_button_placeholder.button("Dừng theo dõi", key=stop_key):
                                    st.session_state.inventory_running = False
                                    break
                            
                            
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