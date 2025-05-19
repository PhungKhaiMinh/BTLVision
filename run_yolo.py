


import cv2
import numpy as np
import torch
import argparse
import time
import os
import glob
from ultralytics import YOLO
import tensorrt
from collections import defaultdict, deque
import sys


class BoundingBoxRefiner:
    def __init__(self, history_length=10, similarity_iou=0.3, height_ratio_threshold=0.9, width_margin=0.1,
                edge_threshold=100, color_threshold=45, min_confidence_ratio=0.8):
        self.split_history = {}  
        self.box_id_counter = 0  
        self.history_length = history_length  
        self.similarity_iou = similarity_iou  
        self.height_ratio_threshold = height_ratio_threshold  
        self.width_margin = width_margin  
        self.split_decisions = {}  
        self.last_frames_detections = deque(maxlen=history_length)  
        self.edge_threshold = edge_threshold  
        self.color_threshold = color_threshold  
        self.min_confidence_ratio = min_confidence_ratio  
        self.verified_boxes = {}  
        
    def _calculate_iou(self, box1, box2):
        
        
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        
        return area_i / (area_1 + area_2 - area_i)
    
    def _find_matching_box_id(self, box, class_id, detections):
        
        for box_id, (c_id, _, saved_box) in self.split_decisions.items():
            if c_id == class_id and self._calculate_iou(box, saved_box) > self.similarity_iou:
                return box_id
        return None
    
    def _should_split_box(self, box, class_id, detections):
        
        
        x, y, w, h = box
        similar_detections = []
        
        for det_cls_id, conf, det_box in detections:
            
            if det_cls_id != class_id:
                continue
                
            det_x, det_y, det_w, det_h = det_box
            
            
            
            expanded_x = x - w * self.width_margin
            expanded_y = y - h * self.width_margin
            expanded_w = w + 2 * w * self.width_margin
            expanded_h = h + 2 * h * self.width_margin
            
            
            if (det_x + det_w > expanded_x and 
                det_x < expanded_x + expanded_w and
                det_y + det_h > expanded_y and 
                det_y < expanded_y + expanded_h):
                
                
                height_ratio = det_h / h
                if height_ratio > self.height_ratio_threshold:
                    similar_detections.append((det_cls_id, conf, det_box))
        
        return len(similar_detections) >= 2, similar_detections
    
    def _analyze_detections_history(self):
        
        
        box_counts = defaultdict(lambda: defaultdict(int))
        
        
        for frame_detections in self.last_frames_detections:
            for cls_id, conf, box in frame_detections:
                box_id = self._find_matching_box_id(box, cls_id, frame_detections)
                
                if box_id is not None:
                    
                    should_split, similar_dets = self._should_split_box(box, cls_id, frame_detections)
                    if should_split:
                        box_counts[box_id][len(similar_dets)] += 1
        
        
        for box_id, counts in box_counts.items():
            if box_id in self.split_decisions:
                cls_id, _, box = self.split_decisions[box_id]
                
                
                most_common_count = max(counts, key=counts.get)
                if counts[most_common_count] >= 3 and most_common_count >= 2:  
                    
                    self.split_decisions[box_id] = (cls_id, most_common_count, box)
    
    def _verify_split_with_image_analysis(self, image, box, num_objects):
        
        if image is None:
            return False, 1
        
        
        x, y, w, h = [int(v) for v in box]
        
        height, width = image.shape[:2]
        x = max(0, min(x, width-1))
        y = max(0, min(y, height-1))
        w = min(w, width-x)
        h = min(h, height-y)
        
        
        if w < 20 or h < 20:
            return False, 1
            
        
        roi = image[y:y+h, x:x+w]
        
        
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(255 * sobelx / np.max(sobelx))
        
        
        hist_x = np.sum(sobelx, axis=0)
        
        
        if np.max(hist_x) > 0:
            hist_x = hist_x / np.max(hist_x) * 255
        
        
        peaks = []
        for i in range(1, len(hist_x)-1):
            if hist_x[i] > self.edge_threshold and hist_x[i] > hist_x[i-1] and hist_x[i] > hist_x[i+1]:
                peaks.append(i)
        
        
        
        middle_row = roi[h//2]
        
        
        color_diff = []
        for i in range(1, w):
            
            diff = np.sqrt(np.sum((middle_row[i].astype(int) - middle_row[i-1].astype(int))**2))
            color_diff.append(diff)
        
        
        color_peaks = []
        for i in range(1, len(color_diff)-1):
            if (color_diff[i] > self.color_threshold and 
                color_diff[i] > color_diff[i-1] and 
                color_diff[i] > color_diff[i+1]):
                color_peaks.append(i)
        
        
        
        combined_peaks = sorted(set(peaks + color_peaks))
        min_peak_distance = max(3, int(w * 0.05))
        
        filtered_peaks = []
        if combined_peaks:
            filtered_peaks = [combined_peaks[0]]
            for peak in combined_peaks[1:]:
                if peak - filtered_peaks[-1] >= min_peak_distance:
                    filtered_peaks.append(peak)
        
        
        
        detected_objects = len(filtered_peaks) + 1 if filtered_peaks else 1
        
        
        verified = False
        
        
        if detected_objects >= num_objects and num_objects > 1:
            
            strong_peaks = [p for p in filtered_peaks if 
                          (hist_x[p] > self.edge_threshold * 1.5 or 
                           (p < len(color_diff) and color_diff[p] > self.color_threshold * 1.5))]
            
            
            if len(strong_peaks) >= num_objects - 1:
                verified = True
        
        
        
        if detected_objects == 1 and num_objects > 1:
            verified = False
        
        return verified, detected_objects
    
    def update(self, detections, image=None):
        
        
        self.last_frames_detections.append(detections.copy())
        
        
        self._analyze_detections_history()
        
        
        refined_detections = []
        processed_boxes = set()
        
        for cls_id, conf, box in detections:
            
            box_id = self._find_matching_box_id(box, cls_id, detections)
            
            if box_id is not None and box_id in self.split_decisions and box_id not in processed_boxes:
                
                _, num_objects, saved_box = self.split_decisions[box_id]
                
                if num_objects >= 2:
                    
                    verified = True  
                    detected_objects = num_objects
                    
                    
                    if box_id in self.verified_boxes:
                        verified = self.verified_boxes[box_id]
                    elif image is not None:
                        
                        verified, detected_objects = self._verify_split_with_image_analysis(
                            image, box, num_objects)
                        
                        self.verified_boxes[box_id] = verified
                    
                    if verified and detected_objects >= 2:
                        
                        actual_objects = min(detected_objects, num_objects)
                        box_width = box[2]
                        obj_width = box_width / actual_objects
                        
                        for i in range(actual_objects):
                            
                            new_x = box[0] + i * obj_width
                            new_box = np.array([new_x, box[1], obj_width, box[3]])
                            
                            new_conf = conf * self.min_confidence_ratio
                            refined_detections.append((cls_id, new_conf, new_box))
                        
                        processed_boxes.add(box_id)
                    else:
                        
                        refined_detections.append((cls_id, conf, box))
                else:
                    
                    refined_detections.append((cls_id, conf, box))
            else:
                
                
                if box_id is None:
                    
                    should_split, similar_dets = self._should_split_box(box, cls_id, detections)
                    
                    if should_split:
                        
                        verified = True  
                        detected_objects = len(similar_dets)
                        
                        if image is not None:
                            
                            verified, detected_objects = self._verify_split_with_image_analysis(
                                image, box, len(similar_dets))
                        
                        if verified and detected_objects >= 2:
                            
                            new_box_id = self.box_id_counter
                            self.box_id_counter += 1
                            
                            
                            self.split_decisions[new_box_id] = (cls_id, detected_objects, box)
                            self.verified_boxes[new_box_id] = verified
                            
                            
                            box_width = box[2]
                            obj_width = box_width / detected_objects
                            
                            for i in range(detected_objects):
                                
                                new_x = box[0] + i * obj_width
                                new_box = np.array([new_x, box[1], obj_width, box[3]])
                                new_conf = conf * self.min_confidence_ratio
                                refined_detections.append((cls_id, new_conf, new_box))
                            
                            processed_boxes.add(new_box_id)
                        else:
                            
                            refined_detections.append((cls_id, conf, box))
                    else:
                        
                        refined_detections.append((cls_id, conf, box))
                else:
                    
                    refined_detections.append((cls_id, conf, box))
        
        
        for cls_id, conf, box in detections:
            box_id = self._find_matching_box_id(box, cls_id, detections)
            if box_id is not None and box_id in processed_boxes:
                continue  
            refined_detections.append((cls_id, conf, box))
        
        return refined_detections
    
    def refine_detections(self, detections, image=None):
        
        
        refined_detections = self.update(detections, image)
        
        
        filtered_detections = []
        for det in refined_detections:
            
            is_duplicate = False
            for existing_det in filtered_detections:
                if det[0] == existing_det[0]:  
                    iou = self._calculate_iou(det[2], existing_det[2])
                    if iou > 0.7:  
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_detections.append(det)
        
        return filtered_detections

def apply_custom_nms(detections, iou_threshold=0.5, same_class_only=True, center_distance_threshold=0.2):
    
    if not detections:
        return []
    
    
    def calculate_iou(box1, box2):
        
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        
        return area_i / (area_1 + area_2 - area_i)
    
    
    def calculate_overlap_ratio(box1, box2):
        
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        
        smaller_area = min(area_1, area_2)
        return area_i / smaller_area if smaller_area > 0 else 0
    
    
    def center_distance(box1, box2):
        
        cx1 = box1[0] + box1[2] / 2
        cy1 = box1[1] + box1[3] / 2
        
        
        cx2 = box2[0] + box2[2] / 2
        cy2 = box2[1] + box2[3] / 2
        
        
        dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        
        
        norm_factor = max(box1[2], box1[3], box2[2], box2[3])
        return dist / norm_factor if norm_factor > 0 else float('inf')
    
    def is_similar_size(box1, box2, threshold=0.7):
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        return ratio > threshold
    
    def is_aligned(box1, box2, vertical_threshold=0.3, horizontal_threshold=0.3):
        
        cx1 = box1[0] + box1[2] / 2
        cy1 = box1[1] + box1[3] / 2
        
        cx2 = box2[0] + box2[2] / 2
        cy2 = box2[1] + box2[3] / 2
        
        
        h_diff = abs(cx1 - cx2) / max(box1[2], box2[2])
        v_diff = abs(cy1 - cy2) / max(box1[3], box2[3])
        
        
        h_aligned = h_diff < horizontal_threshold
        v_aligned = v_diff < vertical_threshold
        
        return h_aligned or v_aligned
    
    
    def is_beverage_can(box):
        width, height = box[2], box[3]
        aspect_ratio = height / width if width > 0 else 0
        
        
        return 1.7 <= aspect_ratio <= 3.2
    
    
    detections_sorted = sorted(detections, key=lambda x: x[1], reverse=True)
    
    
    nms_result = []
    
    
    for i in range(len(detections_sorted)):
        
        if detections_sorted[i] is None:
                continue
                
        
        cls_id_i, conf_i, box_i = detections_sorted[i]
        
        
        nms_result.append(detections_sorted[i])
        
        
        for j in range(i + 1, len(detections_sorted)):
            if detections_sorted[j] is None:
                continue
                
            cls_id_j, conf_j, box_j = detections_sorted[j]
            
            
            if same_class_only and cls_id_i != cls_id_j:
                continue
                
            
            iou = calculate_iou(box_i, box_j)
            
            
            dist = center_distance(box_i, box_j)
            
            
            if iou > iou_threshold:
                
                if is_beverage_can(box_i) and is_beverage_can(box_j):
                    
                    overlap_ratio = calculate_overlap_ratio(box_i, box_j)
                    
                    
                    if (overlap_ratio > 0.7 or
                        (is_similar_size(box_i, box_j) and dist < center_distance_threshold)):
                        detections_sorted[j] = None
                else:
                    
                    detections_sorted[j] = None
    
    return nms_result


class ObjectTracker:
    def __init__(self, buffer_size=10, iou_threshold=0.5, stability_threshold=0.5, confidence_threshold=0.3):
        self.tracked_objects = defaultdict(lambda: deque(maxlen=buffer_size))
        self.object_ids = {}
        self.next_id = 0
        self.buffer_size = buffer_size
        self.iou_threshold = iou_threshold
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
    
    def calculate_iou(self, box1, box2):
        
        
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        
        
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area
    
    def update(self, detections):
        
        
        
        
        if not self.object_ids:
            for cls_id, conf, box in detections:
                if conf >= self.confidence_threshold:
                    obj_id = self.next_id
                    self.next_id += 1
                    self.object_ids[(cls_id, obj_id)] = box
                    self.tracked_objects[(cls_id, obj_id)].append((conf, box))
            return
        
        
        matched_detections = set()
        for (cls_id, obj_id), last_box in self.object_ids.items():
            best_iou = self.iou_threshold
            best_detection = None
            
            for i, (det_cls_id, conf, box) in enumerate(detections):
                if i in matched_detections or det_cls_id != cls_id:
                    continue
                
                iou = self.calculate_iou(last_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_detection = (i, det_cls_id, conf, box)
            
            
            if best_detection:
                i, det_cls_id, conf, box = best_detection
                matched_detections.add(i)
                self.object_ids[(cls_id, obj_id)] = box
                self.tracked_objects[(cls_id, obj_id)].append((conf, box))
            else:
                
                self.tracked_objects[(cls_id, obj_id)].append((0.0, None))
        
        
        for i, (cls_id, conf, box) in enumerate(detections):
            if i not in matched_detections and conf >= self.confidence_threshold:
                obj_id = self.next_id
                self.next_id += 1
                self.object_ids[(cls_id, obj_id)] = box
                self.tracked_objects[(cls_id, obj_id)].append((conf, box))
    
    def get_stable_objects(self):
        
        stable_objects = []
        
        
        to_remove = []
        for (cls_id, obj_id), buffer in self.tracked_objects.items():
            
            appearances = sum(1 for conf, box in buffer if box is not None)
            stability = appearances / len(buffer)
            
            
            if stability >= self.stability_threshold and len(buffer) >= self.buffer_size * 0.5:
                
                valid_boxes = [box for conf, box in buffer if box is not None]
                valid_confs = [conf for conf, box in buffer if box is not None]
                
                if valid_boxes:
                    avg_box = np.mean(valid_boxes, axis=0)
                    avg_conf = np.mean(valid_confs)
                    stable_objects.append((cls_id, obj_id, avg_conf, avg_box))
                    
                    self.object_ids[(cls_id, obj_id)] = avg_box
            
            
            if appearances == 0 and len(buffer) >= self.buffer_size:
                to_remove.append((cls_id, obj_id))
        
        
        for key in to_remove:
            del self.tracked_objects[key]
            del self.object_ids[key]
        
        return stable_objects

class InputSource:
    
    def __init__(self, source_type, source_path=None, droidcam_ip='192.168.1.8', droidcam_port='4747'):
        self.source_type = source_type  
        self.source_path = source_path
        self.droidcam_ip = droidcam_ip
        self.droidcam_port = droidcam_port
        self.cap = None
        self.current_frame = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_image = (source_type == 'image')
        
    def open(self):
        
        if self.source_type == 'droidcam':
            
            url = f"http://{self.droidcam_ip}:{self.droidcam_port}/video"
            print(f"Kết nối đến DroidCam: {url}")
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                raise Exception(f"Không thể kết nối đến DroidCam tại {url}")
            print("Đã kết nối DroidCam thành công")
            
        elif self.source_type == 'video':
            
            if not self.source_path or not os.path.exists(self.source_path):
                raise Exception(f"File video không tồn tại: {self.source_path}")
                
            print(f"Mở file video: {self.source_path}")
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                raise Exception(f"Không thể mở file video: {self.source_path}")
                
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video có tổng cộng {self.total_frames} frames")
            
        elif self.source_type == 'webcam':
            
            cam_index = 0
            if isinstance(self.source_path, int):
                cam_index = self.source_path
            elif isinstance(self.source_path, str) and self.source_path.isdigit():
                cam_index = int(self.source_path)
                
            print(f"Mở webcam với index: {cam_index}")
            self.cap = cv2.VideoCapture(cam_index)
            if not self.cap.isOpened():
                raise Exception(f"Không thể mở webcam với index: {cam_index}")
            
            
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Đã mở webcam với kích thước: {width}x{height}")
            self.total_frames = 0  
            
        elif self.source_type == 'image':
            
            if not self.source_path or not os.path.exists(self.source_path):
                raise Exception(f"File ảnh không tồn tại: {self.source_path}")
            
            
            if os.path.isdir(self.source_path):
                self.image_files = []
                for ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                    self.image_files.extend(glob.glob(os.path.join(self.source_path, f'*.{ext}')))
                    self.image_files.extend(glob.glob(os.path.join(self.source_path, f'*.{ext.upper()}')))
                
                if not self.image_files:
                    raise Exception(f"Không tìm thấy file ảnh trong thư mục: {self.source_path}")
                
                print(f"Tìm thấy {len(self.image_files)} file ảnh trong thư mục")
                self.total_frames = len(self.image_files)
                self.current_frame = cv2.imread(self.image_files[0])
            else:
                
                self.current_frame = cv2.imread(self.source_path)
                if self.current_frame is None:
                    raise Exception(f"Không thể đọc file ảnh: {self.source_path}")
                self.total_frames = 1
                self.image_files = [self.source_path]
                
            print(f"Đã tải ảnh thành công: {self.source_path}")
        else:
            raise Exception(f"Loại nguồn đầu vào không hợp lệ: {self.source_type}")
            
        return True
        
    def read(self):
        
        if self.source_type in ['droidcam', 'video', 'webcam']:
            if self.cap is None:
                return False, None
                
            ret, frame = self.cap.read()
            if not ret and self.source_type == 'video':
                
                print("Đã đọc hết video, quay lại từ đầu")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                self.current_frame_idx = 0
            
            if ret:
                self.current_frame_idx += 1
                
            return ret, frame
        
        elif self.source_type == 'image':
            
            if hasattr(self, 'image_files') and len(self.image_files) > 1:
                if self.current_frame_idx < len(self.image_files):
                    self.current_frame = cv2.imread(self.image_files[self.current_frame_idx])
                    self.current_frame_idx += 1
                    return True, self.current_frame
                else:
                    
                    self.current_frame_idx = 0
                    self.current_frame = cv2.imread(self.image_files[0])
                    return True, self.current_frame
            else:
                
                return True, self.current_frame
        
        return False, None
    
    def get_progress(self):
        
        if self.total_frames > 0:
            return self.current_frame_idx / self.total_frames
        return 0
    
    def release(self):
        
        if self.cap is not None:
            self.cap.release()
            
    def get_dimensions(self):
        
        if self.source_type in ['droidcam', 'video']:
            if self.cap is not None:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return width, height
        elif self.source_type == 'image' and self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
            return width, height
            
        return 0, 0

def get_user_input_source():
    
    print("\n=== CHỌN NGUỒN ĐẦU VÀO ===")
    print("1. DroidCam")
    print("2. Video")
    print("3. Ảnh")
    
    while True:
        try:
            choice = int(input("Chọn loại nguồn đầu vào (1-3): "))
            if choice < 1 or choice > 3:
                print("Vui lòng chọn số từ 1-3")
                continue
                
            if choice == 1:
                
                droidcam_ip = input("Nhập địa chỉ IP DroidCam (mặc định 192.168.1.8): ") or "192.168.1.8"
                droidcam_port = input("Nhập cổng DroidCam (mặc định 4747): ") or "4747"
                return InputSource('droidcam', droidcam_ip=droidcam_ip, droidcam_port=droidcam_port)
                
            elif choice == 2:
                
                while True:
                    video_path = input("Nhập đường dẫn đến file video: ")
                    if os.path.exists(video_path):
                        return InputSource('video', source_path=video_path)
                    else:
                        print(f"File video không tồn tại: {video_path}")
                        
            elif choice == 3:
                
                while True:
                    image_path = input("Nhập đường dẫn đến file ảnh hoặc thư mục chứa ảnh: ")
                    if os.path.exists(image_path):
                        return InputSource('image', source_path=image_path)
                    else:
                        print(f"File/thư mục ảnh không tồn tại: {image_path}")
                        
        except ValueError:
            print("Vui lòng nhập một số")

def convert_model(model_path, format='engine', half=True, workspace=4, device=0):
    
    try:
        base_path = os.path.splitext(model_path)[0]
        target_path = f"{base_path}.{format}"
        
        
        if format == 'engine':
            
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Phát hiện GPU: {gpu_name}")
                
                
                if "1050" in gpu_name or "GTX" in gpu_name:
                    print("GPU này có thể không được hỗ trợ bởi TensorRT. Chuyển sang ONNX...")
                    format = 'onnx'
                    target_path = f"{base_path}.{format}"
            else:
                
                print("Không phát hiện GPU CUDA. Chuyển sang ONNX...")
                format = 'onnx'
                target_path = f"{base_path}.{format}"
        
        print(f"Đang chuyển đổi {model_path} sang {format.upper()}...")
        model = YOLO(model_path)
        
        
        if os.path.exists(target_path):
            print(f"File {format.upper()} đã tồn tại: {target_path}")
            return target_path
        
        
        try:
            success = model.export(format=format, half=half, workspace=workspace, device=device)
            
            if success:
                print(f"Đã chuyển đổi thành công! File {format.upper()} được lưu tại: {target_path}")
                return target_path
            else:
                print(f"Chuyển đổi {format.upper()} không thành công!")
                if format == 'engine':
                    print("Thử chuyển sang ONNX...")
                    return convert_model(model_path, format='onnx', half=half, device=device)
                return None
        except Exception as e:
            print(f"Lỗi khi chuyển đổi sang {format.upper()}: {e}")
            
            if format == 'engine':
                print("Thử chuyển sang ONNX...")
                return convert_model(model_path, format='onnx', half=half, device=device)
            return None
    except Exception as e:
        print(f"Lỗi trong quá trình chuyển đổi sang {format.upper()}: {e}")
        
        if "No module named 'tensorrt'" in str(e) and format == 'engine':
            print("TensorRT không khả dụng. Thử chuyển sang ONNX...")
            return convert_model(model_path, format='onnx', half=half, device=device)
        return None

def load_model(model_path, use_optimization=False, opt_format="engine", half=False, verbose=False):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    optimized_model_path = None
    if use_optimization and device == 'cuda':
        converted_path = os.path.splitext(model_path)[0] + f".{opt_format}"
        if os.path.exists(converted_path):
            optimized_model_path = converted_path
            if verbose:
                print(f"Sử dụng model {opt_format.upper()} đã tồn tại: {converted_path}")
        else:
            if verbose:
                print(f"Đang chuyển đổi model sang {opt_format.upper()}...")
            converted_path = convert_model(model_path, format=opt_format, half=half)
            if converted_path:
                optimized_model_path = converted_path
                if verbose:
                    print(f"Đã chuyển đổi model sang {opt_format.upper()}")
            else:
                if verbose:
                    print(f"Không thể chuyển đổi sang {opt_format.upper()}, sử dụng PyTorch")
    
    
    model = YOLO(model_path)
    if verbose:
        print(f"Đã tải mô hình PyTorch lên {device}")
    
    return {
        "model": model,
        "device": device,
        "optimized_path": optimized_model_path,
        "classes": model.names
    }

def process_image(image, model_data, conf_threshold=0.25, input_size=640, apply_nms=True, 
                 nms_threshold=0.5, center_distance_threshold=0.2, return_image_with_boxes=True, 
                 debug=False, use_box_refiner=False, box_refiner=None, use_box_merger=False, box_merger=None):
    
    if image is None:
        return None, []
    
    
    model = model_data['model']
    class_names = model_data.get('class_names', [])
    
    
    original_height, original_width = image.shape[:2]
    
    
    results = model(image, conf=conf_threshold, verbose=False)
    
    
    detections = []
    for r in results:
        if len(r.boxes.xywh) > 0 and r.boxes.conf.numel() > 0:
            for box_idx, (box, conf, cls) in enumerate(zip(r.boxes.xywh, r.boxes.conf, r.boxes.cls)):
                cls_id = int(cls.item())
                confidence = conf.item()
                x_center, y_center, w, h = box.tolist()
                
                
                x = x_center - w / 2
                y = y_center - h / 2
                
                
                if confidence >= conf_threshold:
                    detections.append((cls_id, confidence, [x, y, w, h]))
    
    
    if apply_nms and detections:
        
        detections = apply_custom_nms(detections, nms_threshold, center_distance_threshold=center_distance_threshold)
        
    
    if return_image_with_boxes:
        image_with_boxes = image.copy()
        
        
        grouped_detections = {}
        for cls_id, confidence, box in detections:
            if cls_id not in grouped_detections:
                grouped_detections[cls_id] = []
            grouped_detections[cls_id].append((confidence, box))
        
        
        for cls_id, class_detections in grouped_detections.items():
            
            best_detection = max(class_detections, key=lambda x: x[0])
            confidence, (x, y, w, h) = best_detection
            
            
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            
            
            color = tuple([(cls_id * 50 + i * 70) % 256 for i in range(3)])
            
            
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
            
            
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_with_boxes, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
            cv2.putText(image_with_boxes, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image_with_boxes, detections
    else:
        return None, detections

def split_wide_detections(detections, aspect_ratio_threshold=2.0):
    
    result = []
    
    for cls_id, conf, box in detections:
        x, y, w, h = box
        aspect_ratio = w / h
        
        
        if aspect_ratio > aspect_ratio_threshold and w > 100:
            
            estimated_objects = max(2, int(aspect_ratio / 1.2))
            
            
            new_width = w / estimated_objects
            for i in range(estimated_objects):
                new_x = x + i * new_width
                new_box = np.array([new_x, y, new_width, h])
                result.append((cls_id, conf * 0.9, new_box))  
        else:
            
            result.append((cls_id, conf, box))
    
    return result

def refine_detections_with_contours(image, detections, aspect_ratio_threshold=2.0):
    
    refined_detections = []
    
    for cls_id, conf, box in detections:
        x, y, w, h = box.astype(int)
        aspect_ratio = w / h
        
        
        if aspect_ratio > aspect_ratio_threshold and w > 100:
            
            roi = image[y:y+h, x:x+w]
            
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            
            
            if len(valid_contours) > 1:
                for cnt in valid_contours:
                    
                    x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    
                    
                    x_new = x + x_cnt
                    y_new = y + y_cnt
                    
                    
                    new_box = np.array([x_new, y_new, w_cnt, h_cnt])
                    refined_detections.append((cls_id, conf * 0.95, new_box))
            else:
                
                refined_detections.append((cls_id, conf, box))
        else:
            
            refined_detections.append((cls_id, conf, box))
    
    return refined_detections

class DuplicateBoxMerger:
    
    def __init__(self, overlap_threshold=0.4, containment_threshold=0.75, 
                center_dist_threshold=0.35, confidence_boost=1.05,
                color_similarity_threshold=30, texture_threshold=0.65,
                min_area_ratio=0.6, area_overlap_threshold=0.7):
        self.overlap_threshold = overlap_threshold  
        self.containment_threshold = containment_threshold  
        self.center_dist_threshold = center_dist_threshold  
        self.confidence_boost = confidence_boost  
        self.color_similarity_threshold = color_similarity_threshold  
        self.texture_threshold = texture_threshold  
        self.min_area_ratio = min_area_ratio  
        self.area_overlap_threshold = area_overlap_threshold  
        self.merge_history = {}  
    
    def _calculate_iou(self, box1, box2):
        
        
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0, 0.0, 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        
        
        smaller_box_area = min(box1_area, box2_area)
        containment = inter_area / smaller_box_area if smaller_box_area > 0 else 0
        
        
        area_ratio = smaller_box_area / max(box1_area, box2_area) if max(box1_area, box2_area) > 0 else 0
        
        return iou, containment, area_ratio
    
    def _calculate_overlap_area(self, box1, box2):
        
        
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        
        smaller_area = min(box1_area, box2_area)
        return inter_area / smaller_area if smaller_area > 0 else 0
    
    def _calculate_center_distance(self, box1, box2):
        
        center1_x = box1[0] + box1[2] / 2
        center1_y = box1[1] + box1[3] / 2
        
        center2_x = box2[0] + box2[2] / 2
        center2_y = box2[1] + box2[3] / 2
        
        
        max_dim = max(box1[2], box1[3], box2[2], box2[3])
        
        
        dist = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        normalized_dist = dist / max_dim if max_dim > 0 else float('inf')
        
        return normalized_dist
    
    def _compare_color_histograms(self, roi1, roi2):
        
        if roi1 is None or roi2 is None:
            return 0.0
            
        
        h, w = min(roi1.shape[0], roi2.shape[0]), min(roi1.shape[1], roi2.shape[1])
        if h < 5 or w < 5:  
            return 0.0
            
        roi1 = cv2.resize(roi1, (w, h))
        roi2 = cv2.resize(roi2, (w, h))
        
        
        hist1_b = cv2.calcHist([roi1], [0], None, [32], [0, 256])
        hist1_g = cv2.calcHist([roi1], [1], None, [32], [0, 256])
        hist1_r = cv2.calcHist([roi1], [2], None, [32], [0, 256])
        
        hist2_b = cv2.calcHist([roi2], [0], None, [32], [0, 256])
        hist2_g = cv2.calcHist([roi2], [1], None, [32], [0, 256])
        hist2_r = cv2.calcHist([roi2], [2], None, [32], [0, 256])
        
        
        cv2.normalize(hist1_b, hist1_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_g, hist1_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_r, hist1_r, 0, 1, cv2.NORM_MINMAX)
        
        cv2.normalize(hist2_b, hist2_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_g, hist2_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_r, hist2_r, 0, 1, cv2.NORM_MINMAX)
        
        
        similarity_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
        similarity_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
        similarity_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)
        
        
        similarity = (similarity_b + similarity_g + similarity_r) / 3
        return max(0, similarity)  
    
    def _compare_texture(self, roi1, roi2):
        
        if roi1 is None or roi2 is None:
            return 0.0
            
        
        h, w = min(roi1.shape[0], roi2.shape[0]), min(roi1.shape[1], roi2.shape[1])
        if h < 10 or w < 10:  
            return 0.0
            
        roi1 = cv2.resize(roi1, (w, h))
        roi2 = cv2.resize(roi2, (w, h))
        
        
        if len(roi1.shape) > 2:
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = roi1
            
        if len(roi2.shape) > 2:
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = roi2
        
        
        sobel_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        
        sobel_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        
        
        mag1 = cv2.magnitude(sobel_x1, sobel_y1)
        mag2 = cv2.magnitude(sobel_x2, sobel_y2)
        
        
        hist1 = cv2.calcHist([mag1.astype(np.float32)], [0], None, [32], [0, 100])
        hist2 = cv2.calcHist([mag2.astype(np.float32)], [0], None, [32], [0, 100])
        
        
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        
        texture_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, texture_similarity)  
    
    def _verify_with_image_analysis(self, box1, box2, image):
        
        if image is None:
            return False
            
        
        h, w = image.shape[:2]
        
        
        x1, y1, w1, h1 = [int(v) for v in box1]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        w1 = min(w1, w-x1)
        h1 = min(h1, h-y1)
        
        x2, y2, w2, h2 = [int(v) for v in box2]
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        w2 = min(w2, w-x2)
        h2 = min(h2, h-y2)
        
        
        if w1 < 5 or h1 < 5 or w2 < 5 or h2 < 5:
            return False
            
        
        roi1 = image[y1:y1+h1, x1:x1+w1]
        roi2 = image[y2:y2+h2, x2:x2+w2]
        
        
        center1_x = x1 + w1/2
        center2_x = x2 + w2/2
        horizontal_distance = abs(center1_x - center2_x)
        avg_width = (w1 + w2) / 2
        
        
        aspect_ratio1 = h1 / w1 if w1 > 0 else 0
        aspect_ratio2 = h2 / w2 if w2 > 0 else 0
        is_can = aspect_ratio1 > 1.5 and aspect_ratio2 > 1.5
        
        
        if is_can and horizontal_distance > avg_width * 0.8:
            return False
        
        
        color_similarity = self._compare_color_histograms(roi1, roi2)
        
        
        texture_similarity = self._compare_texture(roi1, roi2)
        
        
        is_can_case = self._check_if_beverage_can(roi1, roi2)
        
        
        if is_can_case:
            
            if horizontal_distance > avg_width * 0.8:
                return False
                
            
            
            should_merge = color_similarity > 0.7 and horizontal_distance < avg_width * 0.6
        else:
            
            should_merge = (color_similarity > self.texture_threshold and 
                           texture_similarity > self.texture_threshold)
        
        return should_merge
    
    def _check_if_beverage_can(self, roi1, roi2):
        
        
        if roi1 is None or roi2 is None:
            return False
            
        h1, w1 = roi1.shape[:2]
        h2, w2 = roi2.shape[:2]
        
        
        aspect_ratio1 = h1 / w1 if w1 > 0 else 0
        aspect_ratio2 = h2 / w2 if w2 > 0 else 0
        
        is_tall = aspect_ratio1 > 1.3 and aspect_ratio2 > 1.3
        
        
        is_colorful = False
        
        
        try:
            hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
            
            
            sat1 = np.mean(hsv1[:,:,1])
            sat2 = np.mean(hsv2[:,:,1])
            
            
            val_std1 = np.std(hsv1[:,:,2])
            val_std2 = np.std(hsv2[:,:,2])
            
            
            is_colorful = (sat1 > 50 and sat2 > 50) or (val_std1 > 40 and val_std2 > 40)
        except:
            pass
            
        
        has_edges = False
        try:
            
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            
            
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            
            edge_ratio1 = np.sum(edges1 > 0) / (h1 * w1) if h1 * w1 > 0 else 0
            edge_ratio2 = np.sum(edges2 > 0) / (h2 * w2) if h2 * w2 > 0 else 0
            
            
            has_edges = edge_ratio1 > 0.05 and edge_ratio2 > 0.05
        except:
            pass
            
        
        return is_tall and (is_colorful or has_edges)
    
    def _is_beverage_can(self, box):
        
        aspect_ratio = box[3] / box[2]  
        return aspect_ratio > 1.5  
    
    def _should_merge(self, box1, conf1, box2, conf2, image=None):
        
        
        iou, containment, area_ratio = self._calculate_iou(box1, box2)
        
        
        center_dist = self._calculate_center_distance(box1, box2)
        
        
        overlap_area_ratio = self._calculate_overlap_area(box1, box2)
        
        
        is_beverage_can = self._is_beverage_can(box1) and self._is_beverage_can(box2)
        
        
        if is_beverage_can:
            
            center1_x = box1[0] + box1[2]/2
            center2_x = box2[0] + box2[2]/2
            horizontal_distance = abs(center1_x - center2_x)
            
            
            avg_width = (box1[2] + box2[2]) / 2
            if horizontal_distance > avg_width * 0.8:  
                return False
        
        
        should_merge_geometry = False
        
        
        if iou > self.overlap_threshold:
            should_merge_geometry = True
            
        
        elif containment > self.containment_threshold:
            should_merge_geometry = True
            
        
        elif center_dist < self.center_dist_threshold and iou > 0.05:
            should_merge_geometry = True
            
        
        elif area_ratio > self.min_area_ratio and center_dist < self.center_dist_threshold * 1.5 and iou > 0.05:
            should_merge_geometry = True
            
        
        elif overlap_area_ratio > self.area_overlap_threshold:
            should_merge_geometry = True
            
        
        
        elif (abs(box1[1] - box2[1]) < 0.2 * max(box1[3], box2[3]) and  
              abs((box1[0] + box1[2]/2) - (box2[0] + box2[2]/2)) < max(box1[2], box2[2]) * 0.9):  
            
            height_ratio = min(box1[3], box2[3]) / max(box1[3], box2[3])
            width_ratio = min(box1[2], box2[2]) / max(box1[2], box2[2])
            if height_ratio > 0.8 and width_ratio > 0.8:  
                should_merge_geometry = True
                
        
        elif (abs(box1[1] - box2[1]) < 0.2 * max(box1[3], box2[3]) and  
              abs((box1[0] + box1[2]/2) - (box2[0] + box2[2]/2)) < max(box1[2], box2[2]) * 0.9):  
            
            height_ratio = min(box1[3], box2[3]) / max(box1[3], box2[3])
            aspect_ratio1 = box1[3] / box1[2]  
            aspect_ratio2 = box2[3] / box2[2]  
            
            
            if (height_ratio > 0.9 and aspect_ratio1 > 1.5 and aspect_ratio2 > 1.5 and iou > 0.2):
                should_merge_geometry = True
        
        
        if not should_merge_geometry:
            return False
        
        
        if image is not None:
            image_verification = self._verify_with_image_analysis(box1, box2, image)
            
            
            if iou > self.overlap_threshold * 1.5 or containment > self.containment_threshold * 1.2 or overlap_area_ratio > 0.8:
                return True
            return image_verification
        
        
        return should_merge_geometry
    
    def _is_beverage_can(self, box):
        
        aspect_ratio = box[3] / box[2]  
        return aspect_ratio > 1.5  
    
    def _merge_boxes(self, box1, conf1, box2, conf2):
        
        
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[0] + box1[2], box2[0] + box2[2])
        y2 = max(box1[1] + box1[3], box2[1] + box2[3])
        
        
        w = x2 - x1
        h = y2 - y1
        
        
        merged_box = np.array([x1, y1, w, h])
        
        
        merged_conf = max(conf1, conf2) * self.confidence_boost
        
        return merged_box, merged_conf
    
    def _count_unique_objects(self, boxes, confidences, min_distance=50):
        
        if not boxes:
            return 0
            
        
        centers = []
        for box in boxes:
            center_x = box[0] + box[2] / 2
            center_y = box[1] + box[3] / 2
            centers.append((center_x, center_y))
            
        
        from scipy.spatial.distance import pdist, squareform
        if len(centers) > 1:
            distances = squareform(pdist(centers))
            
            
            visited = [False] * len(centers)
            count = 0
            
            for i in range(len(centers)):
                if not visited[i]:
                    count += 1
                    visited[i] = True
                    
                    
                    for j in range(i+1, len(centers)):
                        if not visited[j] and distances[i, j] < min_distance:
                            visited[j] = True
            
            return count
        else:
            return 1
    
    def _is_reasonable_count(self, original_count, merged_count, image_size):
        
        
        if original_count > 0 and merged_count / original_count < 0.25:
            return False
            
        
        if image_size is not None:
            img_area = image_size[0] * image_size[1]
            
            if img_area < 640*480 and merged_count > 10:
                return False
                
        return True
    
    def merge_duplicates(self, detections, image=None):
        
        if not detections:
            return []
        
        
        image_size = None
        if image is not None:
            image_size = image.shape[:2]  
        
        
        grouped_by_class = defaultdict(list)
        for cls_id, conf, box in detections:
            grouped_by_class[cls_id].append((conf, box))
        
        
        merged_detections = []
        
        
        for cls_id, class_detections in grouped_by_class.items():
            original_boxes = [box for _, box in class_detections]
            original_confidences = [conf for conf, _ in class_detections]
            
            
            estimated_objects = self._count_unique_objects(original_boxes, original_confidences)
            
            
            remaining_detections = class_detections.copy()
            processed_detections = []
            
            
            while remaining_detections:
                current_conf, current_box = remaining_detections.pop(0)
                merged = False
                
                
                i = 0
                while i < len(processed_detections):
                    other_conf, other_box = processed_detections[i]
                    if self._should_merge(current_box, current_conf, other_box, other_conf, image):
                        
                        new_box, new_conf = self._merge_boxes(current_box, current_conf, other_box, other_conf)
                        processed_detections[i] = (new_conf, new_box)
                        merged = True
                        break
                    i += 1
                
                
                if not merged:
                    processed_detections.append((current_conf, current_box))
            
            
            if not self._is_reasonable_count(len(class_detections), len(processed_detections), image_size):
                
                processed_detections = class_detections
            
            
            for conf, box in processed_detections:
                merged_detections.append((cls_id, conf, box))
        
        return merged_detections

def main():
    
    parser = argparse.ArgumentParser(description='Chạy YOLO trên ảnh/video/webcam')
    parser.add_argument('--source', '-s', type=str, default='gui', help='Nguồn đầu vào (gui=hiển thị menu chọn, 0=webcam, đường dẫn tới ảnh hoặc video)')
    parser.add_argument('--model', '-m', type=str, default='last(1)-can-n.pt', help='Đường dẫn tới file model')
    parser.add_argument('--conf-threshold', '-c', type=float, default=0.25, help='Ngưỡng độ tin cậy')
    parser.add_argument('--iou-threshold', '-i', type=float, default=0.3, help='Ngưỡng IoU cho NMS')
    parser.add_argument('--center-distance', '-d', type=float, default=0.2, help='Ngưỡng khoảng cách tâm cho NMS')
    parser.add_argument('--save-dir', type=str, default='', help='Thư mục lưu kết quả')
    parser.add_argument('--optimize', action='store_true', help='Tối ưu hóa model')
    parser.add_argument('--opt-format', type=str, default='engine', help='Định dạng tối ưu (engine, onnx)')
    parser.add_argument('--half', action='store_true', help='Sử dụng FP16')
    parser.add_argument('--verbose', '-v', action='store_true', help='Hiển thị chi tiết')
    parser.add_argument('--fps', type=int, default=0, help='Giới hạn FPS (0 = không giới hạn)')
    parser.add_argument('--no-track', action='store_true', help='Không sử dụng object tracking')
    parser.add_argument('--classes', nargs='+', type=int, help='Lọc theo class')
    parser.add_argument('--resolution', type=str, default='', help='Độ phân giải đầu ra (WxH)')
    parser.add_argument('--no-nms', action='store_true', help='Không áp dụng NMS')
    parser.add_argument('--droidcam', action='store_true', help='Sử dụng DroidCam làm nguồn đầu vào')
    parser.add_argument('--droidcam-ip', type=str, default='192.168.1.8', help='IP của DroidCam')
    parser.add_argument('--droidcam-port', type=str, default='4747', help='Port của DroidCam')
    
    args = parser.parse_args()
    
    
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    center_distance = args.center_distance
    save_dir = args.save_dir
    fps_limit = args.fps
    verbose = args.verbose
    use_tracking = not args.no_track
    filtered_classes = args.classes
    use_nms = not args.no_nms
    
    
    output_width, output_height = None, None
    if args.resolution:
        try:
            output_width, output_height = map(int, args.resolution.lower().split('x'))
        except:
            print(f"Định dạng độ phân giải không hợp lệ: {args.resolution}, sử dụng độ phân giải gốc")
    
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    
    if args.source == 'gui':
        source = get_user_input_source()
        if not source:
            print("Không có nguồn đầu vào. Thoát.")
            return
    else:
        if args.droidcam:
            source = InputSource('droidcam', droidcam_ip=args.droidcam_ip, droidcam_port=args.droidcam_port)
        else:
            
            source_type = 'webcam'
            source_path = args.source
            
            
            if args.source.isdigit() or (args.source.startswith('-') and args.source[1:].isdigit()):
                source_type = 'webcam'
                source_path = int(args.source)
            
            elif os.path.exists(args.source):
                
                ext = os.path.splitext(args.source)[1].lower()
                if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
                    source_type = 'video'
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    source_type = 'image'
                else:
                    print(f"Không hỗ trợ định dạng file: {ext}")
                    return
            else:
                print(f"Đường dẫn không tồn tại: {args.source}")
                return
            
            source = InputSource(source_type, source_path)
    
    
    try:
        if not source.open():
            print(f"Không thể mở nguồn đầu vào: {args.source}")
            return
    except Exception as e:
        print(f"Lỗi khi mở nguồn đầu vào: {str(e)}")
        print("Thử sử dụng webcam...")
        source = InputSource('webcam', 0)  
        if not source.open():
            print("Không thể mở webcam. Thoát.")
            return
    
    
    try:
        model_data = load_model(args.model, args.optimize, args.opt_format, args.half, verbose)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        source.release()
        return
    
    if model_data is None:
        print("Không thể tải mô hình")
        source.release()
        return
    
    
    tracker = None
    if use_tracking:
        tracker = ObjectTracker(buffer_size=10, iou_threshold=0.5, stability_threshold=0.5)
    
    
    window_name = f"YOLO - {os.path.basename(args.model)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    
    src_width, src_height = source.get_dimensions()
    if src_width and src_height:
        
        screen_width, screen_height = 1280, 720  
        scale = min(screen_width / src_width, screen_height / src_height) * 0.8
        window_width, window_height = int(src_width * scale), int(src_height * scale)
        cv2.resizeWindow(window_name, window_width, window_height)
    
    
    video_writer = None
    if save_dir and source.source_type in ['video', 'webcam', 'droidcam']:
        output_file = os.path.join(save_dir, f"output_{int(time.time())}.mp4")
        
        
        if output_width and output_height:
            frame_width, frame_height = output_width, output_height
        elif src_width and src_height:
            frame_width, frame_height = src_width, src_height
    else:
            
            ret, frame = source.read()
            if ret:
                frame_height, frame_width = frame.shape[:2]
                
                source.release()
                source.open()
            else:
                frame_width, frame_height = 640, 480
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))
    
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    
    if fps_limit > 0:
        frame_time = 1.0 / fps_limit
    else:
        frame_time = 0
    
    last_frame_time = time.time()
    
    
    while True:
        
        if frame_time > 0:
            elapsed = time.time() - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        last_frame_time = time.time()
        
        
        ret, frame = source.read()
        
        if not ret:
            
            if source.source_type == 'video':
                print("Video đã kết thúc")
                cv2.putText(
                    np.zeros((400, 600, 3), dtype=np.uint8),
                    "Video da ket thuc. Nhan phim bat ky de thoat.",
                    (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                cv2.imshow(window_name, np.zeros((400, 600, 3), dtype=np.uint8))
                cv2.waitKey(0)
            break
        
        
        if output_width and output_height:
            frame = cv2.resize(frame, (output_width, output_height))
        
        
        processed_image, detections = process_image(
            frame, 
            model_data,
            conf_threshold=conf_threshold,
            apply_nms=use_nms,
            nms_threshold=iou_threshold,
            center_distance_threshold=center_distance,
            return_image_with_boxes=False,  
            use_box_refiner=False,  
            box_refiner=None,
            use_box_merger=False,  
            box_merger=None
        )
        
        
        if filtered_classes:
            detections = [d for d in detections if d[0] in filtered_classes]
        
        
        if tracker:
            tracker.update(detections)
            stable_detections = tracker.get_stable_objects()
            
            processed_image = frame.copy()
            if stable_detections:
                for cls_id, obj_id, confidence, (x, y, w, h) in stable_detections:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    class_names = model_data.get('classes', {})
                    if isinstance(class_names, dict):
                        class_name = class_names.get(cls_id, f"Class {cls_id}")
                    elif isinstance(class_names, list):
                        class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                    else:
                        class_name = f"Class {cls_id}"
                    color = (0, 0, 255)
                    cv2.rectangle(processed_image, (x, y), (x + w, y + h), color, 3)
                    label = f"{class_name} | ID:{obj_id} | {confidence:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(processed_image, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
                    cv2.putText(processed_image, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                processed_image = frame.copy()
        else:
            processed_image = frame.copy()
        
        
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(processed_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        if source.source_type == 'video':
            progress = source.get_progress()
            if progress > 0:
                
                width = processed_image.shape[1]
                progress_bar_width = int(width * progress)
                cv2.rectangle(processed_image, (0, 0), (progress_bar_width, 5), (0, 255, 0), -1)
        
        
        cv2.imshow(window_name, processed_image)
        
        
        if video_writer is not None:
            
            if processed_image.shape[1] != frame_width or processed_image.shape[0] != frame_height:
                output_frame = cv2.resize(processed_image, (frame_width, frame_height))
            else:
                output_frame = processed_image
            
            video_writer.write(output_frame)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  
            break
        elif key == ord('s') and save_dir:  
            timestamp = int(time.time())
            save_path = os.path.join(save_dir, f"image_{timestamp}.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"Đã lưu ảnh tại: {save_path}")
    
    
    source.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

def test_box_refinement():
    
    
    parser = argparse.ArgumentParser(description='Kiểm tra thuật toán tách box')
    parser.add_argument('--test-box-refinement', action='store_true', help='Chạy chế độ kiểm tra tách box')
    parser.add_argument('--image-path', type=str, help='Đường dẫn đến file ảnh cần kiểm tra')
    parser.add_argument('--model', type=str, default='last(1)-can-n.pt', help='Đường dẫn đến model YOLOv8')
    parser.add_argument('--conf', type=float, default=0.25, help='Ngưỡng tin cậy')
    parser.add_argument('--disable-box-refiner', action='store_true', help='Tắt BoundingBoxRefiner (mặc định bật)')
    parser.add_argument('--disable-box-merger', action='store_true', help='Tắt DuplicateBoxMerger (mặc định bật)')
    args, _ = parser.parse_known_args()
    
    
    if not args.test_box_refinement:
        return
    
    if not args.image_path:
        print("Vui lòng chỉ định đường dẫn ảnh với --image-path")
        return
    
    print(f"Đang kiểm tra thuật toán tách bounding box trên ảnh: {args.image_path}")
    
    
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ: {args.image_path}")
        return
    
    
    model_data = load_model(args.model, verbose=True)
    
    
    box_refiner = None
    if not args.disable_box_refiner:
        box_refiner = BoundingBoxRefiner(
            history_length=5,
            similarity_iou=0.3,
            height_ratio_threshold=0.9,
            width_margin=0.1,
            edge_threshold=100,
            color_threshold=45,
            min_confidence_ratio=0.8
        )
    
    
    box_merger = None
    if not args.disable_box_merger:
        box_merger = DuplicateBoxMerger(
            overlap_threshold=0.3,
            containment_threshold=0.8,
            center_dist_threshold=0.5,
            confidence_boost=1.05
        )
    
    print("Đang xử lý ảnh với model YOLO...")
    
    
    
    result_img1, detections1 = process_image(
        image, 
        model_data, 
        conf_threshold=args.conf,
        use_box_refiner=False,
        box_refiner=None,
        use_box_merger=False,
        box_merger=None,
        debug=True
    )
    
    
    result_img2, detections2 = process_image(
        image, 
        model_data, 
        conf_threshold=args.conf,
        use_box_refiner=not args.disable_box_refiner,
        box_refiner=box_refiner,
        use_box_merger=False,
        box_merger=None,
        debug=True
    )
    
    
    result_img3, detections3 = process_image(
        image, 
        model_data, 
        conf_threshold=args.conf,
        use_box_refiner=False,
        box_refiner=None,
        use_box_merger=not args.disable_box_merger,
        box_merger=box_merger,
        debug=True
    )
    
    
    result_img4, detections4 = process_image(
        image, 
        model_data, 
        conf_threshold=args.conf,
        use_box_refiner=not args.disable_box_refiner,
        box_refiner=box_refiner,
        use_box_merger=not args.disable_box_merger,
        box_merger=box_merger,
        debug=True
    )
    
    
    cv2.imshow("Original YOLO", result_img1)
    cv2.imshow("BoxRefiner Only", result_img2)
    cv2.imshow("BoxMerger Only", result_img3)
    cv2.imshow("BoxRefiner + BoxMerger", result_img4)
    
    print(f"1. Original: {len(detections1)} đối tượng")
    print(f"2. BoxRefiner: {len(detections2)} đối tượng")
    print(f"3. BoxMerger: {len(detections3)} đối tượng")
    print(f"4. BoxRefiner + BoxMerger: {len(detections4)} đối tượng")
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Đã hoàn tất kiểm tra")

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Kiểm tra chế độ chạy')
    parser.add_argument('--test-box-refinement', action='store_true', help='Chạy chế độ kiểm tra tách box')
    args, _ = parser.parse_known_args()
    
    if args.test_box_refinement:
        test_box_refinement()
    else:
        pass
    main()