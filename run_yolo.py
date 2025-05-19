#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# Lớp tinh chỉnh bounding box
class BoundingBoxRefiner:
    def __init__(self, history_length=10, similarity_iou=0.3, height_ratio_threshold=0.9, width_margin=0.1,
                edge_threshold=100, color_threshold=45, min_confidence_ratio=0.8):
        self.split_history = {}  # Lưu lịch sử phân tách box {box_id: [số lượng đối tượng, số lần phát hiện]}
        self.box_id_counter = 0  # ID duy nhất cho mỗi box
        self.history_length = history_length  # Số frame giữ lại lịch sử
        self.similarity_iou = similarity_iou  # Ngưỡng IoU để xác định box tương tự
        self.height_ratio_threshold = height_ratio_threshold  # Tỷ lệ chiều cao hợp lệ để tách box
        self.width_margin = width_margin  # Biên độ cho phép khi xác định vị trí box
        self.split_decisions = {}  # Lưu quyết định phân tách {box_id: số đối tượng cần tách}
        self.last_frames_detections = deque(maxlen=history_length)  # Lưu các phát hiện từ các frame gần đây
        self.edge_threshold = edge_threshold  # Ngưỡng để xác định cạnh khi phân tích hình ảnh
        self.color_threshold = color_threshold  # Ngưỡng độ chênh lệch màu sắc để xác định vùng phân cách
        self.min_confidence_ratio = min_confidence_ratio  # Tỷ lệ độ tin cậy tối thiểu cho box được tách so với box gốc
        self.verified_boxes = {}  # Lưu kết quả xác minh {box_id: True/False} để không phải xác minh lại
        
    def _calculate_iou(self, box1, box2):
        """Tính IoU giữa hai bounding box"""
        # Chuyển đổi từ [x, y, w, h] sang [x1, y1, x2, y2]
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Tính diện tích giao nhau
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Tính diện tích hai box
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Tính IoU
        return area_i / (area_1 + area_2 - area_i)
    
    def _find_matching_box_id(self, box, class_id, detections):
        """Tìm ID của box đã tồn tại tương tự với box hiện tại"""
        for box_id, (c_id, _, saved_box) in self.split_decisions.items():
            if c_id == class_id and self._calculate_iou(box, saved_box) > self.similarity_iou:
                return box_id
        return None
    
    def _should_split_box(self, box, class_id, detections):
        """Kiểm tra xem một box có nên được tách không dựa trên lịch sử và các phát hiện hiện tại"""
        # Tìm các detections với cùng class_id có bounding box nằm gần hoặc trong box hiện tại
        x, y, w, h = box
        similar_detections = []
        
        for det_cls_id, conf, det_box in detections:
            # Chỉ xét các đối tượng cùng loại
            if det_cls_id != class_id:
                continue
                
            det_x, det_y, det_w, det_h = det_box
            
            # Kiểm tra nếu detection nằm gần hoặc trong box hiện tại
            # Mở rộng box hiện tại ra một chút để xét
            expanded_x = x - w * self.width_margin
            expanded_y = y - h * self.width_margin
            expanded_w = w + 2 * w * self.width_margin
            expanded_h = h + 2 * h * self.width_margin
            
            # Kiểm tra phần giao nhau
            if (det_x + det_w > expanded_x and 
                det_x < expanded_x + expanded_w and
                det_y + det_h > expanded_y and 
                det_y < expanded_y + expanded_h):
                
                # Kiểm tra tỷ lệ chiều cao - phải tương tự
                height_ratio = det_h / h
                if height_ratio > self.height_ratio_threshold:
                    similar_detections.append((det_cls_id, conf, det_box))
        
        return len(similar_detections) >= 2, similar_detections
    
    def _analyze_detections_history(self):
        """Phân tích lịch sử các phát hiện để tìm các box cần tách"""
        # Tạo dict tạm để đếm số lượng đối tượng được phát hiện trong vùng box
        box_counts = defaultdict(lambda: defaultdict(int))
        
        # Phân tích các phát hiện từ các frame gần đây
        for frame_detections in self.last_frames_detections:
            for cls_id, conf, box in frame_detections:
                box_id = self._find_matching_box_id(box, cls_id, frame_detections)
                
                if box_id is not None:
                    # Tìm xem có bao nhiêu đối tượng tương tự trong vùng box này
                    should_split, similar_dets = self._should_split_box(box, cls_id, frame_detections)
                    if should_split:
                        box_counts[box_id][len(similar_dets)] += 1
        
        # Cập nhật split_decisions dựa trên phân tích
        for box_id, counts in box_counts.items():
            if box_id in self.split_decisions:
                cls_id, _, box = self.split_decisions[box_id]
                
                # Nếu nhiều lần phát hiện cùng số lượng đối tượng trong box
                most_common_count = max(counts, key=counts.get)
                if counts[most_common_count] >= 3 and most_common_count >= 2:  # Nếu phát hiện >= 3 lần và có >= 2 đối tượng
                    # Cập nhật số lượng đối tượng cần tách
                    self.split_decisions[box_id] = (cls_id, most_common_count, box)
    
    def _verify_split_with_image_analysis(self, image, box, num_objects):
        """
        Xác minh việc tách box bằng cách phân tích hình ảnh để kiểm tra xem có thực sự có nhiều đối tượng hay không
        
        Args:
            image: Ảnh gốc
            box: Bounding box [x, y, w, h]
            num_objects: Số đối tượng dự kiến trong box
        
        Returns:
            (verified, suggested_count): Đã xác minh hay chưa và số đối tượng đề xuất sau phân tích
        """
        if image is None:
            return False, 1
        
        # Trích xuất vùng ảnh cần phân tích
        x, y, w, h = [int(v) for v in box]
        # Đảm bảo tọa độ nằm trong giới hạn ảnh
        height, width = image.shape[:2]
        x = max(0, min(x, width-1))
        y = max(0, min(y, height-1))
        w = min(w, width-x)
        h = min(h, height-y)
        
        # Nếu vùng ảnh quá nhỏ, không thể phân tích đáng tin cậy
        if w < 20 or h < 20:
            return False, 1
            
        # Lấy vùng ảnh
        roi = image[y:y+h, x:x+w]
        
        # Phương pháp 1: Phân tích gradient theo chiều ngang để tìm ranh giới giữa các đối tượng
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Làm mờ ảnh để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Tính toán gradient theo chiều ngang (Sobel X)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        sobelx = np.uint8(255 * sobelx / np.max(sobelx))
        
        # Tạo histogram theo chiều ngang
        hist_x = np.sum(sobelx, axis=0)
        
        # Chuẩn hóa histogram
        if np.max(hist_x) > 0:
            hist_x = hist_x / np.max(hist_x) * 255
        
        # Tìm các đỉnh lớn trong histogram - đây có thể là ranh giới giữa các đối tượng
        peaks = []
        for i in range(1, len(hist_x)-1):
            if hist_x[i] > self.edge_threshold and hist_x[i] > hist_x[i-1] and hist_x[i] > hist_x[i+1]:
                peaks.append(i)
        
        # Phương pháp 2: Phân tích sự thay đổi màu sắc theo chiều ngang
        # Lấy dòng ở giữa vùng
        middle_row = roi[h//2]
        
        # Tính sự thay đổi màu sắc giữa các pixel liền kề
        color_diff = []
        for i in range(1, w):
            # Tính độ khác biệt màu sắc Euclidean
            diff = np.sqrt(np.sum((middle_row[i].astype(int) - middle_row[i-1].astype(int))**2))
            color_diff.append(diff)
        
        # Tìm các điểm có sự thay đổi màu sắc lớn
        color_peaks = []
        for i in range(1, len(color_diff)-1):
            if (color_diff[i] > self.color_threshold and 
                color_diff[i] > color_diff[i-1] and 
                color_diff[i] > color_diff[i+1]):
                color_peaks.append(i)
        
        # Kết hợp kết quả từ hai phương pháp
        # Gộp các đỉnh gần nhau (khoảng cách < 5% chiều rộng)
        combined_peaks = sorted(set(peaks + color_peaks))
        min_peak_distance = max(3, int(w * 0.05))
        
        filtered_peaks = []
        if combined_peaks:
            filtered_peaks = [combined_peaks[0]]
            for peak in combined_peaks[1:]:
                if peak - filtered_peaks[-1] >= min_peak_distance:
                    filtered_peaks.append(peak)
        
        # Phân tích kết quả
        # Số đỉnh phát hiện + 1 có thể là số đối tượng
        detected_objects = len(filtered_peaks) + 1 if filtered_peaks else 1
        
        # Để xác minh, chúng ta cần ít nhất 1 đỉnh mạnh nếu có 2 đối tượng
        verified = False
        
        # Kiểm tra xem số đối tượng phát hiện có phù hợp với số đối tượng dự kiến không
        if detected_objects >= num_objects and num_objects > 1:
            # Cần ít nhất 1 đỉnh mạnh cho mỗi ranh giới
            strong_peaks = [p for p in filtered_peaks if 
                          (hist_x[p] > self.edge_threshold * 1.5 or 
                           (p < len(color_diff) and color_diff[p] > self.color_threshold * 1.5))]
            
            # Nếu có đủ đỉnh mạnh, ta có thể xác minh việc tách
            if len(strong_peaks) >= num_objects - 1:
                verified = True
        
        # Nếu chỉ có 1 đối tượng được phát hiện, nhưng model muốn tách thành nhiều
        # Ta sẽ không tách nếu không tìm thấy đủ bằng chứng
        if detected_objects == 1 and num_objects > 1:
            verified = False
        
        return verified, detected_objects
    
    def update(self, detections, image=None):
        """Cập nhật bộ tinh chỉnh với các phát hiện mới"""
        # Lưu các phát hiện hiện tại vào lịch sử
        self.last_frames_detections.append(detections.copy())
        
        # Phân tích lịch sử phát hiện
        self._analyze_detections_history()
        
        # Kiểm tra các phát hiện mới xem có cần tách không
        refined_detections = []
        processed_boxes = set()
        
        for cls_id, conf, box in detections:
            # Tìm box_id tương ứng
            box_id = self._find_matching_box_id(box, cls_id, detections)
            
            if box_id is not None and box_id in self.split_decisions and box_id not in processed_boxes:
                # Đã có quyết định tách box này
                _, num_objects, saved_box = self.split_decisions[box_id]
                
                if num_objects >= 2:
                    # Xác minh việc tách bằng phân tích hình ảnh nếu có ảnh đầu vào
                    verified = True  # Mặc định cho trường hợp không có hình ảnh
                    detected_objects = num_objects
                    
                    # Nếu đã xác minh trước đó, sử dụng kết quả đã lưu
                    if box_id in self.verified_boxes:
                        verified = self.verified_boxes[box_id]
                    elif image is not None:
                        # Phân tích hình ảnh để xác minh
                        verified, detected_objects = self._verify_split_with_image_analysis(
                            image, box, num_objects)
                        # Lưu kết quả xác minh
                        self.verified_boxes[box_id] = verified
                    
                    if verified and detected_objects >= 2:
                        # Tách box thành nhiều đối tượng - số đối tượng theo phân tích nếu có
                        actual_objects = min(detected_objects, num_objects)
                        box_width = box[2]
                        obj_width = box_width / actual_objects
                        
                        for i in range(actual_objects):
                            # Tạo box mới cho mỗi đối tượng
                            new_x = box[0] + i * obj_width
                            new_box = np.array([new_x, box[1], obj_width, box[3]])
                            # Giảm độ tin cậy một chút để phản ánh sự không chắc chắn
                            new_conf = conf * self.min_confidence_ratio
                            refined_detections.append((cls_id, new_conf, new_box))
                        
                        processed_boxes.add(box_id)
                    else:
                        # Không đủ bằng chứng để tách, giữ nguyên box
                        refined_detections.append((cls_id, conf, box))
                else:
                    # Không tách
                    refined_detections.append((cls_id, conf, box))
            else:
                # Box chưa có trong lịch sử hoặc không cần tách
                # Kiểm tra xem có cần tạo box_id mới không
                if box_id is None:
                    # Kiểm tra xem box này có nên được tách không dựa trên detections hiện tại
                    should_split, similar_dets = self._should_split_box(box, cls_id, detections)
                    
                    if should_split:
                        # Xác minh bằng phân tích hình ảnh nếu có
                        verified = True  # Mặc định cho trường hợp không có hình ảnh
                        detected_objects = len(similar_dets)
                        
                        if image is not None:
                            # Phân tích hình ảnh để xác minh
                            verified, detected_objects = self._verify_split_with_image_analysis(
                                image, box, len(similar_dets))
                        
                        if verified and detected_objects >= 2:
                            # Tạo ID mới cho box này
                            new_box_id = self.box_id_counter
                            self.box_id_counter += 1
                            
                            # Lưu quyết định tách và kết quả xác minh
                            self.split_decisions[new_box_id] = (cls_id, detected_objects, box)
                            self.verified_boxes[new_box_id] = verified
                            
                            # Tách box
                            box_width = box[2]
                            obj_width = box_width / detected_objects
                            
                            for i in range(detected_objects):
                                # Tạo box mới cho mỗi đối tượng
                                new_x = box[0] + i * obj_width
                                new_box = np.array([new_x, box[1], obj_width, box[3]])
                                new_conf = conf * self.min_confidence_ratio
                                refined_detections.append((cls_id, new_conf, new_box))
                            
                            processed_boxes.add(new_box_id)
                        else:
                            # Không đủ bằng chứng để tách, giữ nguyên box
                            refined_detections.append((cls_id, conf, box))
                    else:
                        # Không tách
                        refined_detections.append((cls_id, conf, box))
                else:
                    # Đã có box_id nhưng không cần tách
                    refined_detections.append((cls_id, conf, box))
        
        # Thêm các phát hiện còn lại chưa được xử lý
        for cls_id, conf, box in detections:
            box_id = self._find_matching_box_id(box, cls_id, detections)
            if box_id is not None and box_id in processed_boxes:
                continue  # Đã xử lý rồi
            refined_detections.append((cls_id, conf, box))
        
        return refined_detections
    
    def refine_detections(self, detections, image=None):
        """Tinh chỉnh các detections, tách các box chứa nhiều đối tượng"""
        # Cập nhật bộ tinh chỉnh
        refined_detections = self.update(detections, image)
        
        # Lọc bỏ các detections trùng lặp
        filtered_detections = []
        for det in refined_detections:
            # Kiểm tra xem detection này có trùng với detection nào đã thêm vào chưa
            is_duplicate = False
            for existing_det in filtered_detections:
                if det[0] == existing_det[0]:  # Cùng class
                    iou = self._calculate_iou(det[2], existing_det[2])
                    if iou > 0.7:  # IoU cao => trùng lặp
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_detections.append(det)
        
        return filtered_detections

def apply_custom_nms(detections, iou_threshold=0.5, same_class_only=True, center_distance_threshold=0.2):
    """
    Áp dụng Non-Maximum Suppression (NMS) tùy chỉnh cho danh sách các phát hiện.
    
    Args:
        detections: List các detection [class_id, confidence, [x, y, w, h]]
        iou_threshold: Ngưỡng IoU để so sánh các box
        same_class_only: Chỉ áp dụng NMS cho các box cùng class
        center_distance_threshold: Ngưỡng khoảng cách tâm cho các box được xem là gần nhau
    
    Returns:
        List các detection sau khi áp dụng NMS
    """
    if not detections:
        return []
    
    # Hàm tính IoU giữa hai bounding box
    def calculate_iou(box1, box2):
        # Tính tọa độ giao nhau
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Tính diện tích giao nhau
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Tính diện tích hai box
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Tính IoU
        return area_i / (area_1 + area_2 - area_i)
    
    # Hàm tính tỷ lệ diện tích giao nhau trên diện tích box nhỏ hơn
    def calculate_overlap_ratio(box1, box2):
        # Tính tọa độ giao nhau
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
        
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Tính diện tích giao nhau
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Tính diện tích hai box
        area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Tính tỷ lệ overlap
        smaller_area = min(area_1, area_2)
        return area_i / smaller_area if smaller_area > 0 else 0
    
    # Hàm tính khoảng cách giữa tâm hai box
    def center_distance(box1, box2):
        # Tính tâm box1
        cx1 = box1[0] + box1[2] / 2
        cy1 = box1[1] + box1[3] / 2
        
        # Tính tâm box2
        cx2 = box2[0] + box2[2] / 2
        cy2 = box2[1] + box2[3] / 2
        
        # Tính khoảng cách Euclidean
        dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        
        # Chuẩn hóa khoảng cách theo kích thước box để so sánh tương đối
        norm_factor = max(box1[2], box1[3], box2[2], box2[3])
        return dist / norm_factor if norm_factor > 0 else float('inf')
    
    def is_similar_size(box1, box2, threshold=0.7):
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        return ratio > threshold
    
    def is_aligned(box1, box2, vertical_threshold=0.3, horizontal_threshold=0.3):
        # Tính tâm
        cx1 = box1[0] + box1[2] / 2
        cy1 = box1[1] + box1[3] / 2
        
        cx2 = box2[0] + box2[2] / 2
        cy2 = box2[1] + box2[3] / 2
        
        # Tính độ lệch tương đối
        h_diff = abs(cx1 - cx2) / max(box1[2], box2[2])
        v_diff = abs(cy1 - cy2) / max(box1[3], box2[3])
        
        # Kiểm tra xếp thẳng hàng
        h_aligned = h_diff < horizontal_threshold
        v_aligned = v_diff < vertical_threshold
        
        return h_aligned or v_aligned
    
    # Kiểm tra có phải lon nước hay không dựa trên tỷ lệ kích thước
    def is_beverage_can(box):
        width, height = box[2], box[3]
        aspect_ratio = height / width if width > 0 else 0
        
        # Lon nước thường có tỷ lệ cao/rộng từ khoảng 1.8 đến 3.0
        return 1.7 <= aspect_ratio <= 3.2
    
    # Sắp xếp detections theo độ tin cậy giảm dần
    detections_sorted = sorted(detections, key=lambda x: x[1], reverse=True)
    
    # Kết quả sau NMS
    nms_result = []
    
    # Kiểm tra từng detection
    for i in range(len(detections_sorted)):
        # Nếu detection đã được đánh dấu loại bỏ, bỏ qua
        if detections_sorted[i] is None:
                continue
                
        # Lấy thông tin của detection hiện tại
        cls_id_i, conf_i, box_i = detections_sorted[i]
        
        # Thêm detection vào kết quả
        nms_result.append(detections_sorted[i])
        
        # So sánh với các detection còn lại để loại bỏ các detection chồng lấp
        for j in range(i + 1, len(detections_sorted)):
            if detections_sorted[j] is None:
                continue
                
            cls_id_j, conf_j, box_j = detections_sorted[j]
            
            # Nếu yêu cầu chỉ áp dụng NMS trong cùng class và hai detection khác class, bỏ qua
            if same_class_only and cls_id_i != cls_id_j:
                continue
                
            # Tính toán IoU
            iou = calculate_iou(box_i, box_j)
            
            # Tính khoảng cách tâm (chuẩn hóa theo kích thước box)
            dist = center_distance(box_i, box_j)
            
            # Kiểm tra xem hai box có chồng lấp nhiều không
            if iou > iou_threshold:
                # Xử lý đặc biệt cho lon nước - thường hay bị nhận diện trùng lặp
                if is_beverage_can(box_i) and is_beverage_can(box_j):
                    # Lấy tỷ lệ overlap
                    overlap_ratio = calculate_overlap_ratio(box_i, box_j)
                    
                    # Nếu là lon nước và có overlap lớn, có thể là cùng một đối tượng
                    if (overlap_ratio > 0.7 or
                        (is_similar_size(box_i, box_j) and dist < center_distance_threshold)):
                        detections_sorted[j] = None
                else:
                    # Các đối tượng khác, dùng IoU để quyết định
                    detections_sorted[j] = None
    
    return nms_result

# Lớp theo dõi đối tượng qua thời gian
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
        """Tính toán IoU giữa hai box"""
        # Chuyển đổi từ [x, y, w, h] sang [x1, y1, x2, y2]
        box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        
        # Tính diện tích hai box
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        
        # Tìm tọa độ giao nhau
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        # Tính diện tích phần giao nhau
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Tính IoU
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area
    
    def update(self, detections):
        """Cập nhật theo dõi với các phát hiện mới"""
        # Detections là danh sách các tuple (class_id, confidence, box)
        
        # Nếu không có đối tượng nào đang theo dõi, khởi tạo với detections đầu tiên
        if not self.object_ids:
            for cls_id, conf, box in detections:
                if conf >= self.confidence_threshold:
                    obj_id = self.next_id
                    self.next_id += 1
                    self.object_ids[(cls_id, obj_id)] = box
                    self.tracked_objects[(cls_id, obj_id)].append((conf, box))
            return
        
        # Khớp các phát hiện mới với các đối tượng đang theo dõi
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
            
            # Cập nhật đối tượng với phát hiện tốt nhất
            if best_detection:
                i, det_cls_id, conf, box = best_detection
                matched_detections.add(i)
                self.object_ids[(cls_id, obj_id)] = box
                self.tracked_objects[(cls_id, obj_id)].append((conf, box))
            else:
                # Không có khớp, thêm None vào buffer
                self.tracked_objects[(cls_id, obj_id)].append((0.0, None))
        
        # Thêm phát hiện mới chưa khớp với đối tượng nào
        for i, (cls_id, conf, box) in enumerate(detections):
            if i not in matched_detections and conf >= self.confidence_threshold:
                obj_id = self.next_id
                self.next_id += 1
                self.object_ids[(cls_id, obj_id)] = box
                self.tracked_objects[(cls_id, obj_id)].append((conf, box))
    
    def get_stable_objects(self):
        """Lấy các đối tượng ổn định"""
        stable_objects = []
        
        # Duyệt qua các đối tượng đang theo dõi
        to_remove = []
        for (cls_id, obj_id), buffer in self.tracked_objects.items():
            # Đếm số lần xuất hiện trong buffer
            appearances = sum(1 for conf, box in buffer if box is not None)
            stability = appearances / len(buffer)
            
            # Nếu đối tượng xuất hiện ổn định
            if stability >= self.stability_threshold and len(buffer) >= self.buffer_size * 0.5:
                # Tính trung bình vị trí và kích thước
                valid_boxes = [box for conf, box in buffer if box is not None]
                valid_confs = [conf for conf, box in buffer if box is not None]
                
                if valid_boxes:
                    avg_box = np.mean(valid_boxes, axis=0)
                    avg_conf = np.mean(valid_confs)
                    stable_objects.append((cls_id, obj_id, avg_conf, avg_box))
                    # Cập nhật vị trí cuối cùng đã biết
                    self.object_ids[(cls_id, obj_id)] = avg_box
            
            # Loại bỏ đối tượng đã biến mất quá lâu
            if appearances == 0 and len(buffer) >= self.buffer_size:
                to_remove.append((cls_id, obj_id))
        
        # Loại bỏ các đối tượng không còn theo dõi
        for key in to_remove:
            del self.tracked_objects[key]
            del self.object_ids[key]
        
        return stable_objects

class InputSource:
    """Lớp đại diện cho nguồn đầu vào."""
    def __init__(self, source_type, source_path=None, droidcam_ip='10.229.161.17', droidcam_port='4747'):
        self.source_type = source_type  # 'droidcam', 'video', 'image', 'webcam'
        self.source_path = source_path
        self.droidcam_ip = droidcam_ip
        self.droidcam_port = droidcam_port
        self.cap = None
        self.current_frame = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.is_image = (source_type == 'image')
        
    def open(self):
        """Mở nguồn đầu vào"""
        if self.source_type == 'droidcam':
            # Mở kết nối DroidCam
            url = f"http://{self.droidcam_ip}:{self.droidcam_port}/video"
            print(f"Kết nối đến DroidCam: {url}")
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                raise Exception(f"Không thể kết nối đến DroidCam tại {url}")
            print("Đã kết nối DroidCam thành công")
            
        elif self.source_type == 'video':
            # Mở file video
            if not self.source_path or not os.path.exists(self.source_path):
                raise Exception(f"File video không tồn tại: {self.source_path}")
                
            print(f"Mở file video: {self.source_path}")
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                raise Exception(f"Không thể mở file video: {self.source_path}")
                
            # Lấy tổng số frame
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video có tổng cộng {self.total_frames} frames")
            
        elif self.source_type == 'webcam':
            # Mở webcam
            cam_index = 0
            if isinstance(self.source_path, int):
                cam_index = self.source_path
            elif isinstance(self.source_path, str) and self.source_path.isdigit():
                cam_index = int(self.source_path)
                
            print(f"Mở webcam với index: {cam_index}")
            self.cap = cv2.VideoCapture(cam_index)
            if not self.cap.isOpened():
                raise Exception(f"Không thể mở webcam với index: {cam_index}")
            
            # Lấy kích thước
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Đã mở webcam với kích thước: {width}x{height}")
            self.total_frames = 0  # Không có tổng số frame cho webcam
            
        elif self.source_type == 'image':
            # Đối với ảnh, kiểm tra xem file có tồn tại không
            if not self.source_path or not os.path.exists(self.source_path):
                raise Exception(f"File ảnh không tồn tại: {self.source_path}")
            
            # Nếu source_path là thư mục, lấy danh sách các file ảnh
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
                # Nếu là file ảnh đơn lẻ
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
        """Đọc frame tiếp theo từ nguồn đầu vào"""
        if self.source_type in ['droidcam', 'video', 'webcam']:
            if self.cap is None:
                return False, None
                
            ret, frame = self.cap.read()
            if not ret and self.source_type == 'video':
                # Nếu là video và đã đọc hết, quay lại từ đầu
                print("Đã đọc hết video, quay lại từ đầu")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                self.current_frame_idx = 0
            
            if ret:
                self.current_frame_idx += 1
                
            return ret, frame
        
        elif self.source_type == 'image':
            # Nếu có nhiều ảnh
            if hasattr(self, 'image_files') and len(self.image_files) > 1:
                if self.current_frame_idx < len(self.image_files):
                    self.current_frame = cv2.imread(self.image_files[self.current_frame_idx])
                    self.current_frame_idx += 1
                    return True, self.current_frame
                else:
                    # Quay lại ảnh đầu tiên
                    self.current_frame_idx = 0
                    self.current_frame = cv2.imread(self.image_files[0])
                    return True, self.current_frame
            else:
                # Đối với ảnh đơn, luôn trả về cùng một ảnh
                return True, self.current_frame
        
        return False, None
    
    def get_progress(self):
        """Trả về tiến trình xử lý (cho video)"""
        if self.total_frames > 0:
            return self.current_frame_idx / self.total_frames
        return 0
    
    def release(self):
        """Giải phóng tài nguyên"""
        if self.cap is not None:
            self.cap.release()
            
    def get_dimensions(self):
        """Lấy kích thước của frame đầu vào"""
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
    """Hàm để người dùng chọn nguồn đầu vào"""
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
                # DroidCam
                droidcam_ip = input("Nhập địa chỉ IP DroidCam (mặc định 10.229.161.17): ") or "10.229.161.17"
                droidcam_port = input("Nhập cổng DroidCam (mặc định 4747): ") or "4747"
                return InputSource('droidcam', droidcam_ip=droidcam_ip, droidcam_port=droidcam_port)
                
            elif choice == 2:
                # Video
                while True:
                    video_path = input("Nhập đường dẫn đến file video: ")
                    if os.path.exists(video_path):
                        return InputSource('video', source_path=video_path)
                    else:
                        print(f"File video không tồn tại: {video_path}")
                        
            elif choice == 3:
                # Ảnh
                while True:
                    image_path = input("Nhập đường dẫn đến file ảnh hoặc thư mục chứa ảnh: ")
                    if os.path.exists(image_path):
                        return InputSource('image', source_path=image_path)
                    else:
                        print(f"File/thư mục ảnh không tồn tại: {image_path}")
                        
        except ValueError:
            print("Vui lòng nhập một số")

def convert_model(model_path, format='engine', half=True, workspace=4, device=0):
    """Chuyển đổi model PyTorch sang định dạng tối ưu (TensorRT hoặc ONNX)"""
    try:
        base_path = os.path.splitext(model_path)[0]
        target_path = f"{base_path}.{format}"
        
        # Kiểm tra GPU nếu format là engine (TensorRT)
        if format == 'engine':
            # Cố gắng lấy thông tin GPU 
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"Phát hiện GPU: {gpu_name}")
                
                # Nếu là GTX 1050 Ti hoặc các GPU cũ, chuyển sang ONNX
                if "1050" in gpu_name or "GTX" in gpu_name:
                    print("GPU này có thể không được hỗ trợ bởi TensorRT. Chuyển sang ONNX...")
                    format = 'onnx'
                    target_path = f"{base_path}.{format}"
            else:
                # Không có GPU, chuyển sang ONNX
                print("Không phát hiện GPU CUDA. Chuyển sang ONNX...")
                format = 'onnx'
                target_path = f"{base_path}.{format}"
        
        print(f"Đang chuyển đổi {model_path} sang {format.upper()}...")
        model = YOLO(model_path)
        
        # Nếu file đã tồn tại, trả về luôn
        if os.path.exists(target_path):
            print(f"File {format.upper()} đã tồn tại: {target_path}")
            return target_path
        
        # Thử chuyển đổi sang định dạng yêu cầu
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
            # Nếu là lỗi với TensorRT, chuyển sang ONNX
            if format == 'engine':
                print("Thử chuyển sang ONNX...")
                return convert_model(model_path, format='onnx', half=half, device=device)
            return None
    except Exception as e:
        print(f"Lỗi trong quá trình chuyển đổi sang {format.upper()}: {e}")
        # Nếu là lỗi với TensorRT, chuyển sang ONNX
        if "No module named 'tensorrt'" in str(e) and format == 'engine':
            print("TensorRT không khả dụng. Thử chuyển sang ONNX...")
            return convert_model(model_path, format='onnx', half=half, device=device)
        return None

def load_model(model_path, use_optimization=False, opt_format="engine", half=False, verbose=False):
    """Tải model YOLOv8"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Kiểm tra model tối ưu
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
    
    # Tải model PyTorch
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
    """
    Xử lý ảnh với mô hình YOLO để phát hiện đối tượng
    """
    if image is None:
        return None, []
    
    # Chuẩn bị dữ liệu
    model = model_data['model']
    class_names = model_data.get('class_names', [])
    
    # Lưu kích thước gốc
    original_height, original_width = image.shape[:2]
    
    # Thực hiện phát hiện
    results = model(image, conf=conf_threshold, verbose=False)
    
    # Lấy các phát hiện
    detections = []
    for r in results:
        if len(r.boxes.xywh) > 0 and r.boxes.conf.numel() > 0:
            for box_idx, (box, conf, cls) in enumerate(zip(r.boxes.xywh, r.boxes.conf, r.boxes.cls)):
                cls_id = int(cls.item())
                confidence = conf.item()
                x_center, y_center, w, h = box.tolist()
                
                # Chuyển đổi từ định dạng xywh (center) sang định dạng xyxy (top-left)
                x = x_center - w / 2
                y = y_center - h / 2
                
                # Nếu độ tin cậy vượt ngưỡng, thêm vào danh sách phát hiện
                if confidence >= conf_threshold:
                    detections.append((cls_id, confidence, [x, y, w, h]))
    
    # Áp dụng NMS nếu yêu cầu
    if apply_nms and detections:
        # Áp dụng custom NMS
        detections = apply_custom_nms(detections, nms_threshold, center_distance_threshold=center_distance_threshold)
        
    # Vẽ hộp và nhãn lên ảnh (nếu yêu cầu)
    if return_image_with_boxes:
        image_with_boxes = image.copy()
        
        # Nhóm các detection theo class_id
        grouped_detections = {}
        for cls_id, confidence, box in detections:
            if cls_id not in grouped_detections:
                grouped_detections[cls_id] = []
            grouped_detections[cls_id].append((confidence, box))
        
        # Vẽ box cuối cùng cho mỗi class
        for cls_id, class_detections in grouped_detections.items():
            # Lấy detection có độ tin cậy cao nhất
            best_detection = max(class_detections, key=lambda x: x[0])
            confidence, (x, y, w, h) = best_detection
            
            # Chuyển đổi sang int cho việc vẽ
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Lấy tên class
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            
            # Chọn màu ngẫu nhiên nhưng nhất quán cho mỗi class
            color = tuple([(cls_id * 50 + i * 70) % 256 for i in range(3)])
            
            # Vẽ hộp
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
            
            # Vẽ nhãn
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image_with_boxes, (x, y - label_height - baseline - 5), (x + label_width, y), color, -1)
            cv2.putText(image_with_boxes, label, (x, y - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image_with_boxes, detections
    else:
        return None, detections

def split_wide_detections(detections, aspect_ratio_threshold=2.0):
    """
    Phân tách bounding box có tỷ lệ chiều rộng/chiều cao lớn thành nhiều đối tượng nhỏ hơn
    
    Args:
        detections: Danh sách tuple (class_id, confidence, box)
        aspect_ratio_threshold: Ngưỡng tỷ lệ kích thước để xác định box bất thường
    
    Returns:
        Danh sách detections sau khi đã phân tách các box rộng
    """
    result = []
    
    for cls_id, conf, box in detections:
        x, y, w, h = box
        aspect_ratio = w / h
        
        # Nếu box quá rộng, có thể chứa nhiều đối tượng
        if aspect_ratio > aspect_ratio_threshold and w > 100:
            # Ước tính số lượng đối tượng dựa trên tỷ lệ kích thước
            estimated_objects = max(2, int(aspect_ratio / 1.2))
            
            # Phân tách thành nhiều box nhỏ
            new_width = w / estimated_objects
            for i in range(estimated_objects):
                new_x = x + i * new_width
                new_box = np.array([new_x, y, new_width, h])
                result.append((cls_id, conf * 0.9, new_box))  # Giảm độ tin cậy một chút
        else:
            # Giữ nguyên box
            result.append((cls_id, conf, box))
    
    return result

def refine_detections_with_contours(image, detections, aspect_ratio_threshold=2.0):
    """
    Sử dụng phát hiện đường viền để tinh chỉnh các đối tượng có tỷ lệ bất thường
    """
    refined_detections = []
    
    for cls_id, conf, box in detections:
        x, y, w, h = box.astype(int)
        aspect_ratio = w / h
        
        # Chỉ xử lý các box có tỷ lệ bất thường
        if aspect_ratio > aspect_ratio_threshold and w > 100:
            # Cắt vùng ảnh chứa đối tượng
            roi = image[y:y+h, x:x+w]
            
            # Chuyển sang ảnh xám và áp dụng threshold
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            
            # Tìm contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Lọc các contour có kích thước lớn
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
            
            # Nếu tìm được nhiều contour
            if len(valid_contours) > 1:
                for cnt in valid_contours:
                    # Tính bounding box cho contour
                    x_cnt, y_cnt, w_cnt, h_cnt = cv2.boundingRect(cnt)
                    
                    # Điều chỉnh tọa độ về hệ tọa độ gốc
                    x_new = x + x_cnt
                    y_new = y + y_cnt
                    
                    # Thêm detection mới
                    new_box = np.array([x_new, y_new, w_cnt, h_cnt])
                    refined_detections.append((cls_id, conf * 0.95, new_box))
            else:
                # Giữ nguyên detection
                refined_detections.append((cls_id, conf, box))
        else:
            # Giữ nguyên detection
            refined_detections.append((cls_id, conf, box))
    
    return refined_detections

class DuplicateBoxMerger:
    """
    Lớp xử lý và hợp nhất các bounding box trùng lặp (nhiều box trên cùng một đối tượng)
    """
    def __init__(self, overlap_threshold=0.4, containment_threshold=0.75, 
                center_dist_threshold=0.35, confidence_boost=1.05,
                color_similarity_threshold=30, texture_threshold=0.65,
                min_area_ratio=0.6, area_overlap_threshold=0.7):
        self.overlap_threshold = overlap_threshold  # Ngưỡng IoU để xác định các box có khả năng trùng lặp
        self.containment_threshold = containment_threshold  # Ngưỡng phần trăm diện tích box nhỏ nằm trong box lớn
        self.center_dist_threshold = center_dist_threshold  # Ngưỡng khoảng cách tâm (tỷ lệ theo kích thước)
        self.confidence_boost = confidence_boost  # Hệ số tăng độ tin cậy khi hợp nhất các box
        self.color_similarity_threshold = color_similarity_threshold  # Ngưỡng độ tương đồng màu sắc
        self.texture_threshold = texture_threshold  # Ngưỡng độ tương đồng kết cấu
        self.min_area_ratio = min_area_ratio  # Tỷ lệ diện tích tối thiểu giữa box nhỏ hơn/box lớn hơn
        self.area_overlap_threshold = area_overlap_threshold  # Ngưỡng phần trăm diện tích giao nhau / diện tích nhỏ hơn
        self.merge_history = {}  # Lưu lịch sử hợp nhất {box_id: [box_ids đã hợp nhất]}
    
    def _calculate_iou(self, box1, box2):
        """Tính IoU giữa hai bounding box"""
        # Chuyển đổi từ [x, y, w, h] sang [x1, y1, x2, y2]
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Tính diện tích giao nhau
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0, 0.0, 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Tính diện tích hai box
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        # Tính IoU
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        
        # Tính tỷ lệ containment - phần diện tích box nhỏ nằm trong box lớn
        smaller_box_area = min(box1_area, box2_area)
        containment = inter_area / smaller_box_area if smaller_box_area > 0 else 0
        
        # Tính tỷ lệ diện tích
        area_ratio = smaller_box_area / max(box1_area, box2_area) if max(box1_area, box2_area) > 0 else 0
        
        return iou, containment, area_ratio
    
    def _calculate_overlap_area(self, box1, box2):
        """Tính tỷ lệ diện tích giao nhau so với diện tích box nhỏ hơn"""
        # Chuyển đổi từ [x, y, w, h] sang [x1, y1, x2, y2]
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Tính diện tích giao nhau
        inter_x1 = max(box1_x1, box2_x1)
        inter_y1 = max(box1_y1, box2_y1)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Diện tích hai box
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        
        # Tỷ lệ diện tích giao nhau so với diện tích box nhỏ hơn
        smaller_area = min(box1_area, box2_area)
        return inter_area / smaller_area if smaller_area > 0 else 0
    
    def _calculate_center_distance(self, box1, box2):
        """Tính khoảng cách giữa tâm của hai box, chuẩn hóa theo kích thước box lớn nhất"""
        center1_x = box1[0] + box1[2] / 2
        center1_y = box1[1] + box1[3] / 2
        
        center2_x = box2[0] + box2[2] / 2
        center2_y = box2[1] + box2[3] / 2
        
        # Kích thước lớn nhất giữa hai box
        max_dim = max(box1[2], box1[3], box2[2], box2[3])
        
        # Tính khoảng cách Euclidean và chuẩn hóa
        dist = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        normalized_dist = dist / max_dim if max_dim > 0 else float('inf')
        
        return normalized_dist
    
    def _compare_color_histograms(self, roi1, roi2):
        """So sánh histogram màu của hai vùng ảnh"""
        if roi1 is None or roi2 is None:
            return 0.0
            
        # Chuyển về cùng kích thước
        h, w = min(roi1.shape[0], roi2.shape[0]), min(roi1.shape[1], roi2.shape[1])
        if h < 5 or w < 5:  # Vùng ảnh quá nhỏ
            return 0.0
            
        roi1 = cv2.resize(roi1, (w, h))
        roi2 = cv2.resize(roi2, (w, h))
        
        # Tính histogram cho các kênh màu
        hist1_b = cv2.calcHist([roi1], [0], None, [32], [0, 256])
        hist1_g = cv2.calcHist([roi1], [1], None, [32], [0, 256])
        hist1_r = cv2.calcHist([roi1], [2], None, [32], [0, 256])
        
        hist2_b = cv2.calcHist([roi2], [0], None, [32], [0, 256])
        hist2_g = cv2.calcHist([roi2], [1], None, [32], [0, 256])
        hist2_r = cv2.calcHist([roi2], [2], None, [32], [0, 256])
        
        # Chuẩn hóa các histogram
        cv2.normalize(hist1_b, hist1_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_g, hist1_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist1_r, hist1_r, 0, 1, cv2.NORM_MINMAX)
        
        cv2.normalize(hist2_b, hist2_b, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_g, hist2_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2_r, hist2_r, 0, 1, cv2.NORM_MINMAX)
        
        # So sánh các histogram
        similarity_b = cv2.compareHist(hist1_b, hist2_b, cv2.HISTCMP_CORREL)
        similarity_g = cv2.compareHist(hist1_g, hist2_g, cv2.HISTCMP_CORREL)
        similarity_r = cv2.compareHist(hist1_r, hist2_r, cv2.HISTCMP_CORREL)
        
        # Tính trung bình độ tương đồng
        similarity = (similarity_b + similarity_g + similarity_r) / 3
        return max(0, similarity)  # Đảm bảo giá trị không âm
    
    def _compare_texture(self, roi1, roi2):
        """So sánh kết cấu (texture) của hai vùng ảnh"""
        if roi1 is None or roi2 is None:
            return 0.0
            
        # Chuyển về cùng kích thước
        h, w = min(roi1.shape[0], roi2.shape[0]), min(roi1.shape[1], roi2.shape[1])
        if h < 10 or w < 10:  # Vùng ảnh quá nhỏ
            return 0.0
            
        roi1 = cv2.resize(roi1, (w, h))
        roi2 = cv2.resize(roi2, (w, h))
        
        # Chuyển sang ảnh xám
        if len(roi1.shape) > 2:
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = roi1
            
        if len(roi2.shape) > 2:
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = roi2
        
        # Tính gradient (Sobel) cho cả hai ảnh
        sobel_x1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y1 = cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)
        
        sobel_x2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y2 = cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)
        
        # Tính độ lớn gradient
        mag1 = cv2.magnitude(sobel_x1, sobel_y1)
        mag2 = cv2.magnitude(sobel_x2, sobel_y2)
        
        # Tạo histogram của gradient
        hist1 = cv2.calcHist([mag1.astype(np.float32)], [0], None, [32], [0, 100])
        hist2 = cv2.calcHist([mag2.astype(np.float32)], [0], None, [32], [0, 100])
        
        # Chuẩn hóa
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # So sánh histogram
        texture_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, texture_similarity)  # Đảm bảo giá trị không âm
    
    def _verify_with_image_analysis(self, box1, box2, image):
        """Xác minh việc hợp nhất bằng phân tích hình ảnh"""
        if image is None:
            return False
            
        # Lấy vùng ảnh tương ứng với hai box
        h, w = image.shape[:2]
        
        # Đảm bảo tọa độ nằm trong giới hạn ảnh
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
        
        # Kiểm tra kích thước
        if w1 < 5 or h1 < 5 or w2 < 5 or h2 < 5:
            return False
            
        # Lấy vùng ảnh
        roi1 = image[y1:y1+h1, x1:x1+w1]
        roi2 = image[y2:y2+h2, x2:x2+w2]
        
        # Tính khoảng cách vị trí tuyệt đối
        center1_x = x1 + w1/2
        center2_x = x2 + w2/2
        horizontal_distance = abs(center1_x - center2_x)
        avg_width = (w1 + w2) / 2
        
        # Kiểm tra nếu là lon nước (dựa vào tỷ lệ)
        aspect_ratio1 = h1 / w1 if w1 > 0 else 0
        aspect_ratio2 = h2 / w2 if w2 > 0 else 0
        is_can = aspect_ratio1 > 1.5 and aspect_ratio2 > 1.5
        
        # Nếu khoảng cách lớn hơn chiều rộng và cả hai đều là lon -> không hợp nhất
        if is_can and horizontal_distance > avg_width * 0.8:
            return False
        
        # So sánh màu sắc
        color_similarity = self._compare_color_histograms(roi1, roi2)
        
        # So sánh kết cấu
        texture_similarity = self._compare_texture(roi1, roi2)
        
        # Kiểm tra đặc biệt cho lon nước
        is_can_case = self._check_if_beverage_can(roi1, roi2)
        
        # Đánh giá tổng thể
        if is_can_case:
            # Nếu là lon nước và khoảng cách lớn -> khả năng cao là 2 lon khác nhau
            if horizontal_distance > avg_width * 0.8:
                return False
                
            # Đối với lon nước, nếu màu sắc tương tự -> hợp nhất
            # Nhưng chỉ khi chúng thực sự rất gần nhau
            should_merge = color_similarity > 0.7 and horizontal_distance < avg_width * 0.6
        else:
            # Tiêu chuẩn thông thường
            should_merge = (color_similarity > self.texture_threshold and 
                           texture_similarity > self.texture_threshold)
        
        return should_merge
    
    def _check_if_beverage_can(self, roi1, roi2):
        """Kiểm tra xem có phải trường hợp lon nước không dựa trên đặc điểm hình ảnh"""
        # Kiểm tra tỷ lệ kích thước
        if roi1 is None or roi2 is None:
            return False
            
        h1, w1 = roi1.shape[:2]
        h2, w2 = roi2.shape[:2]
        
        # Lon nước thường có tỷ lệ chiều cao/chiều rộng > 1.5
        aspect_ratio1 = h1 / w1 if w1 > 0 else 0
        aspect_ratio2 = h2 / w2 if w2 > 0 else 0
        
        is_tall = aspect_ratio1 > 1.3 and aspect_ratio2 > 1.3
        
        # Lon nước thường có màu đặc trưng và độ tương phản cao
        is_colorful = False
        
        # Chuyển sang không gian màu HSV để phân tích màu sắc
        try:
            hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
            
            # Tính toán độ bão hòa trung bình - lon nước thường có độ bão hòa cao
            sat1 = np.mean(hsv1[:,:,1])
            sat2 = np.mean(hsv2[:,:,1])
            
            # Tính toán độ tương phản bằng độ lệch chuẩn của giá trị (V trong HSV)
            val_std1 = np.std(hsv1[:,:,2])
            val_std2 = np.std(hsv2[:,:,2])
            
            # Lon nước thường có độ bão hòa và độ tương phản cao
            is_colorful = (sat1 > 50 and sat2 > 50) or (val_std1 > 40 and val_std2 > 40)
        except:
            pass
            
        # Kiểm tra tỷ lệ đường viền/diện tích - lon nước thường có logo hoặc văn bản 
        has_edges = False
        try:
            # Chuyển đổi sang ảnh xám
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            
            # Áp dụng Canny edge detector
            edges1 = cv2.Canny(gray1, 50, 150)
            edges2 = cv2.Canny(gray2, 50, 150)
            
            # Tỷ lệ đường viền/diện tích
            edge_ratio1 = np.sum(edges1 > 0) / (h1 * w1) if h1 * w1 > 0 else 0
            edge_ratio2 = np.sum(edges2 > 0) / (h2 * w2) if h2 * w2 > 0 else 0
            
            # Lon nước thường có tỷ lệ đường viền cao do logo, nhãn, văn bản
            has_edges = edge_ratio1 > 0.05 and edge_ratio2 > 0.05
        except:
            pass
            
        # Kết hợp các điều kiện để xác định lon nước
        return is_tall and (is_colorful or has_edges)
    
    def _is_beverage_can(self, box):
        """Kiểm tra xem một box có phải là lon nước không dựa vào tỷ lệ khung"""
        aspect_ratio = box[3] / box[2]  # height/width
        return aspect_ratio > 1.5  # Lon nước thường cao hơn rộng nhiều
    
    def _should_merge(self, box1, conf1, box2, conf2, image=None):
        """Kiểm tra xem hai box có nên được hợp nhất không"""
        # Tính IoU và tỷ lệ containment
        iou, containment, area_ratio = self._calculate_iou(box1, box2)
        
        # Tính khoảng cách tâm
        center_dist = self._calculate_center_distance(box1, box2)
        
        # Tính tỷ lệ diện tích giao nhau
        overlap_area_ratio = self._calculate_overlap_area(box1, box2)
        
        # Kiểm tra nhanh xem có phải là lon nước không
        is_beverage_can = self._is_beverage_can(box1) and self._is_beverage_can(box2)
        
        # Nếu là lon nước, áp dụng quy tắc nghiêm ngặt hơn
        if is_beverage_can:
            # Tính khoảng cách vị trí tuyệt đối giữa các lon
            center1_x = box1[0] + box1[2]/2
            center2_x = box2[0] + box2[2]/2
            horizontal_distance = abs(center1_x - center2_x)
            
            # Nếu khoảng cách lớn hơn chiều rộng trung bình, đây là 2 lon khác nhau
            avg_width = (box1[2] + box2[2]) / 2
            if horizontal_distance > avg_width * 0.8:  # Giảm ngưỡng này để ngăn hợp nhất các lon riêng
                return False
        
        # Điều kiện ban đầu dựa trên phân tích hình học
        should_merge_geometry = False
        
        # Điều kiện 1: IoU cao -> Hai box có phần lớn trùng nhau
        if iou > self.overlap_threshold:
            should_merge_geometry = True
            
        # Điều kiện 2: Box nhỏ gần như hoàn toàn nằm trong box lớn
        elif containment > self.containment_threshold:
            should_merge_geometry = True
            
        # Điều kiện 3: Hai tâm rất gần nhau và có overlap
        elif center_dist < self.center_dist_threshold and iou > 0.05:
            should_merge_geometry = True
            
        # Điều kiện 4: Box có tỷ lệ kích thước tương tự, tâm gần nhau, và có overlap
        elif area_ratio > self.min_area_ratio and center_dist < self.center_dist_threshold * 1.5 and iou > 0.05:
            should_merge_geometry = True
            
        # Điều kiện 5: Diện tích giao nhau chiếm phần lớn box nhỏ hơn
        elif overlap_area_ratio > self.area_overlap_threshold:
            should_merge_geometry = True
            
        # Điều kiện 6: Các box nằm gần nhau theo chiều ngang (cho trường hợp Pepsi)
        # Hạn chế hơn để tránh hợp nhất các lon riêng biệt
        elif (abs(box1[1] - box2[1]) < 0.2 * max(box1[3], box2[3]) and  # Nghiêm ngặt hơn (0.3->0.2)
              abs((box1[0] + box1[2]/2) - (box2[0] + box2[2]/2)) < max(box1[2], box2[2]) * 0.9):  # Nghiêm ngặt hơn (1.5->0.9)
            # Kiểm tra thêm về kích thước tương đồng 
            height_ratio = min(box1[3], box2[3]) / max(box1[3], box2[3])
            width_ratio = min(box1[2], box2[2]) / max(box1[2], box2[2])
            if height_ratio > 0.8 and width_ratio > 0.8:  # Nghiêm ngặt hơn (0.6->0.8)
                should_merge_geometry = True
                
        # Điều kiện 7: Xử lý trường hợp lon nước đặc biệt - làm nghiêm ngặt hơn
        elif (abs(box1[1] - box2[1]) < 0.2 * max(box1[3], box2[3]) and  # Nghiêm ngặt hơn (0.4->0.2)
              abs((box1[0] + box1[2]/2) - (box2[0] + box2[2]/2)) < max(box1[2], box2[2]) * 0.9):  # Nghiêm ngặt hơn (2.0->0.9)
            # Kiểm tra xem có phải trường hợp đặc biệt của lon nước không
            height_ratio = min(box1[3], box2[3]) / max(box1[3], box2[3])
            aspect_ratio1 = box1[3] / box1[2]  # Tỷ lệ chiều cao/chiều rộng của box1
            aspect_ratio2 = box2[3] / box2[2]  # Tỷ lệ chiều cao/chiều rộng của box2
            
            # Lon nước thường có tỷ lệ chiều cao/chiều rộng > 1.5
            if (height_ratio > 0.9 and aspect_ratio1 > 1.5 and aspect_ratio2 > 1.5 and iou > 0.2):
                should_merge_geometry = True
        
        # Nếu không thỏa mãn điều kiện hình học, không hợp nhất
        if not should_merge_geometry:
            return False
        
        # Nếu có ảnh, xác minh thêm bằng phân tích hình ảnh
        if image is not None:
            image_verification = self._verify_with_image_analysis(box1, box2, image)
            # Kết hợp kết quả từ hình học và phân tích hình ảnh
            # Nếu điều kiện hình học rất mạnh (IoU cao hoặc containment cao) thì không cần phân tích hình ảnh
            if iou > self.overlap_threshold * 1.5 or containment > self.containment_threshold * 1.2 or overlap_area_ratio > 0.8:
                return True
            return image_verification
        
        # Nếu không có ảnh, dựa trên hình học
        return should_merge_geometry
    
    def _is_beverage_can(self, box):
        """Kiểm tra xem một box có phải là lon nước không dựa vào tỷ lệ khung"""
        aspect_ratio = box[3] / box[2]  # height/width
        return aspect_ratio > 1.5  # Lon nước thường cao hơn rộng nhiều
    
    def _merge_boxes(self, box1, conf1, box2, conf2):
        """Hợp nhất hai box thành một box mới"""
        # Tìm bounding box bao quanh cả hai box
        x1 = min(box1[0], box2[0])
        y1 = min(box1[1], box2[1])
        x2 = max(box1[0] + box1[2], box2[0] + box2[2])
        y2 = max(box1[1] + box1[3], box2[1] + box2[3])
        
        # Kích thước mới
        w = x2 - x1
        h = y2 - y1
        
        # Box mới
        merged_box = np.array([x1, y1, w, h])
        
        # Độ tin cậy mới - tăng độ tin cậy để phản ánh việc hợp nhất
        merged_conf = max(conf1, conf2) * self.confidence_boost
        
        return merged_box, merged_conf
    
    def _count_unique_objects(self, boxes, confidences, min_distance=50):
        """Ước tính số lượng đối tượng duy nhất dựa trên phân phối không gian"""
        if not boxes:
            return 0
            
        # Danh sách các tâm
        centers = []
        for box in boxes:
            center_x = box[0] + box[2] / 2
            center_y = box[1] + box[3] / 2
            centers.append((center_x, center_y))
            
        # Tính khoảng cách giữa các tâm
        from scipy.spatial.distance import pdist, squareform
        if len(centers) > 1:
            distances = squareform(pdist(centers))
            
            # Đếm số nhóm (mỗi nhóm là một đối tượng)
            visited = [False] * len(centers)
            count = 0
            
            for i in range(len(centers)):
                if not visited[i]:
                    count += 1
                    visited[i] = True
                    
                    # Tìm các tâm gần nhau
                    for j in range(i+1, len(centers)):
                        if not visited[j] and distances[i, j] < min_distance:
                            visited[j] = True
            
            return count
        else:
            return 1
    
    def _is_reasonable_count(self, original_count, merged_count, image_size):
        """Đánh giá xem số lượng đối tượng sau khi hợp nhất có hợp lý không"""
        # Heuristic đơn giản: sau khi merge, số lượng không nên giảm quá 75%
        if original_count > 0 and merged_count / original_count < 0.25:
            return False
            
        # Đánh giá dựa trên kích thước ảnh
        if image_size is not None:
            img_area = image_size[0] * image_size[1]
            # Đối với ảnh nhỏ, không nên có quá nhiều đối tượng
            if img_area < 640*480 and merged_count > 10:
                return False
                
        return True
    
    def merge_duplicates(self, detections, image=None):
        """
        Hợp nhất các bounding box trùng lặp trong danh sách detections
        
        Args:
            detections: Danh sách các tuple (class_id, confidence, box)
            image: Ảnh gốc để phân tích (tùy chọn)
            
        Returns:
            Danh sách các detections sau khi đã hợp nhất các box trùng lặp
        """
        if not detections:
            return []
        
        # Kích thước ảnh nếu có
        image_size = None
        if image is not None:
            image_size = image.shape[:2]  # (height, width)
        
        # Nhóm các detection theo class_id
        grouped_by_class = defaultdict(list)
        for cls_id, conf, box in detections:
            grouped_by_class[cls_id].append((conf, box))
        
        # Kết quả sau khi xử lý
        merged_detections = []
        
        # Xử lý cho từng class_id
        for cls_id, class_detections in grouped_by_class.items():
            original_boxes = [box for _, box in class_detections]
            original_confidences = [conf for conf, _ in class_detections]
            
            # Kiểm tra cơ bản để ước tính số đối tượng thực tế
            estimated_objects = self._count_unique_objects(original_boxes, original_confidences)
            
            # Sao chép danh sách detections cho class này
            remaining_detections = class_detections.copy()
            processed_detections = []
            
            # Lặp cho đến khi xử lý hết các detection
            while remaining_detections:
                current_conf, current_box = remaining_detections.pop(0)
                merged = False
                
                # So sánh với các detection đã xử lý
                i = 0
                while i < len(processed_detections):
                    other_conf, other_box = processed_detections[i]
                    if self._should_merge(current_box, current_conf, other_box, other_conf, image):
                        # Hợp nhất và cập nhật box đã xử lý
                        new_box, new_conf = self._merge_boxes(current_box, current_conf, other_box, other_conf)
                        processed_detections[i] = (new_conf, new_box)
                        merged = True
                        break
                    i += 1
                
                # Nếu không hợp nhất với box đã xử lý nào, thêm vào danh sách
                if not merged:
                    processed_detections.append((current_conf, current_box))
            
            # Kiểm tra số lượng đối tượng sau khi hợp nhất
            if not self._is_reasonable_count(len(class_detections), len(processed_detections), image_size):
                # Nếu số lượng không hợp lý, quay lại kết quả ban đầu
                processed_detections = class_detections
            
            # Thêm vào kết quả cuối cùng
            for conf, box in processed_detections:
                merged_detections.append((cls_id, conf, box))
        
        return merged_detections

def main():
    # Phân tích tham số dòng lệnh
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
    parser.add_argument('--droidcam-ip', type=str, default='10.229.161.17', help='IP của DroidCam')
    parser.add_argument('--droidcam-port', type=str, default='4747', help='Port của DroidCam')
    
    args = parser.parse_args()
    
    # Chuẩn bị các tham số
    conf_threshold = args.conf_threshold
    iou_threshold = args.iou_threshold
    center_distance = args.center_distance
    save_dir = args.save_dir
    fps_limit = args.fps
    verbose = args.verbose
    use_tracking = not args.no_track
    filtered_classes = args.classes
    use_nms = not args.no_nms
    
    # Xử lý độ phân giải đầu ra nếu được chỉ định
    output_width, output_height = None, None
    if args.resolution:
        try:
            output_width, output_height = map(int, args.resolution.lower().split('x'))
        except:
            print(f"Định dạng độ phân giải không hợp lệ: {args.resolution}, sử dụng độ phân giải gốc")
    
    # Tạo thư mục lưu kết quả nếu cần
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Lấy nguồn đầu vào
    if args.source == 'gui':
        source = get_user_input_source()
        if not source:
            print("Không có nguồn đầu vào. Thoát.")
            return
    else:
        if args.droidcam:
            source = InputSource('droidcam', droidcam_ip=args.droidcam_ip, droidcam_port=args.droidcam_port)
        else:
            # Xác định loại nguồn đầu vào
            source_type = 'webcam'
            source_path = args.source
            
            # Nếu nguồn là số (0, 1, 2,...) thì là webcam
            if args.source.isdigit() or (args.source.startswith('-') and args.source[1:].isdigit()):
                source_type = 'webcam'
                source_path = int(args.source)
            # Nếu nguồn là đường dẫn file
            elif os.path.exists(args.source):
                # Kiểm tra phần mở rộng của file
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
    
    # Mở nguồn đầu vào
    try:
        if not source.open():
            print(f"Không thể mở nguồn đầu vào: {args.source}")
            return
    except Exception as e:
        print(f"Lỗi khi mở nguồn đầu vào: {str(e)}")
        print("Thử sử dụng webcam...")
        source = InputSource('webcam', 0)  # Thử với webcam mặc định
        if not source.open():
            print("Không thể mở webcam. Thoát.")
            return
    
    # Chuẩn bị mô hình
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
    
    # Khởi tạo object tracker nếu cần
    tracker = None
    if use_tracking:
        tracker = ObjectTracker(buffer_size=10, iou_threshold=0.5, stability_threshold=0.5)
    
    # Khởi tạo cửa sổ hiển thị
    window_name = f"YOLO - {os.path.basename(args.model)}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Lấy kích thước của nguồn nếu có
    src_width, src_height = source.get_dimensions()
    if src_width and src_height:
        # Điều chỉnh kích thước cửa sổ để phù hợp với màn hình
        screen_width, screen_height = 1280, 720  # Kích thước mặc định
        scale = min(screen_width / src_width, screen_height / src_height) * 0.8
        window_width, window_height = int(src_width * scale), int(src_height * scale)
        cv2.resizeWindow(window_name, window_width, window_height)
    
    # Chuẩn bị đầu ra video nếu cần
    video_writer = None
    if save_dir and source.source_type in ['video', 'webcam', 'droidcam']:
        output_file = os.path.join(save_dir, f"output_{int(time.time())}.mp4")
        
        # Xác định kích thước đầu ra
        if output_width and output_height:
            frame_width, frame_height = output_width, output_height
        elif src_width and src_height:
            frame_width, frame_height = src_width, src_height
    else:
            # Đọc frame đầu tiên để lấy kích thước
            ret, frame = source.read()
            if ret:
                frame_height, frame_width = frame.shape[:2]
                # Đặt lại nguồn để bắt đầu từ đầu
                source.release()
                source.open()
            else:
                frame_width, frame_height = 640, 480
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 30, (frame_width, frame_height))
    
    # Biến để tính FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Limit FPS
    if fps_limit > 0:
        frame_time = 1.0 / fps_limit
    else:
        frame_time = 0
    
    last_frame_time = time.time()
    
    # Xử lý từng frame
    while True:
        # Kiểm tra giới hạn FPS
        if frame_time > 0:
            elapsed = time.time() - last_frame_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        
        last_frame_time = time.time()
        
        # Đọc frame từ nguồn
        ret, frame = source.read()
        
        if not ret:
            # Nếu là video, hiển thị thông báo kết thúc và chờ nhấn phím
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
        
        # Điều chỉnh kích thước nếu cần
        if output_width and output_height:
            frame = cv2.resize(frame, (output_width, output_height))
        
        # Xử lý frame với model YOLO
        processed_image, detections = process_image(
            frame, 
            model_data,
            conf_threshold=conf_threshold,
            apply_nms=use_nms,
            nms_threshold=iou_threshold,
            center_distance_threshold=center_distance,
            return_image_with_boxes=False,  # Không vẽ bounding box tạm thời
            use_box_refiner=False,  # Không sử dụng box refiner
            box_refiner=None,
            use_box_merger=False,  # Không sử dụng box merger
            box_merger=None
        )
        
        # Lọc theo class nếu cần
        if filtered_classes:
            detections = [d for d in detections if d[0] in filtered_classes]
        
        # Nếu sử dụng object tracking
        if tracker:
            tracker.update(detections)
            stable_detections = tracker.get_stable_objects()
            # Vẽ các đối tượng ổn định lên ảnh
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
        
        # Tính và hiển thị FPS
        frame_count += 1
        if frame_count >= 10:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
        
        cv2.putText(processed_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị tiến trình nếu là video
        if source.source_type == 'video':
            progress = source.get_progress()
            if progress > 0:
                # Vẽ thanh tiến trình
                width = processed_image.shape[1]
                progress_bar_width = int(width * progress)
                cv2.rectangle(processed_image, (0, 0), (progress_bar_width, 5), (0, 255, 0), -1)
        
        # Hiển thị ảnh
        cv2.imshow(window_name, processed_image)
        
        # Lưu video nếu cần
        if video_writer is not None:
            # Đảm bảo kích thước khớp với VideoWriter
            if processed_image.shape[1] != frame_width or processed_image.shape[0] != frame_height:
                output_frame = cv2.resize(processed_image, (frame_width, frame_height))
            else:
                output_frame = processed_image
            # Ghi frame ra video
            video_writer.write(output_frame)
        
        # Kiểm tra phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC hoặc 'q' để thoát
            break
        elif key == ord('s') and save_dir:  # 's' để lưu ảnh
            timestamp = int(time.time())
            save_path = os.path.join(save_dir, f"image_{timestamp}.jpg")
            cv2.imwrite(save_path, processed_image)
            print(f"Đã lưu ảnh tại: {save_path}")
    
    # Giải phóng tài nguyên
    source.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

def test_box_refinement():
    """
    Chức năng demo để trực tiếp kiểm tra thuật toán tách bounding box
    Chạy với: python run_yolo.py --test-box-refinement --image-path duong_dan_anh.jpg
    """
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Kiểm tra thuật toán tách box')
    parser.add_argument('--test-box-refinement', action='store_true', help='Chạy chế độ kiểm tra tách box')
    parser.add_argument('--image-path', type=str, help='Đường dẫn đến file ảnh cần kiểm tra')
    parser.add_argument('--model', type=str, default='last(1)-can-n.pt', help='Đường dẫn đến model YOLOv8')
    parser.add_argument('--conf', type=float, default=0.25, help='Ngưỡng tin cậy')
    parser.add_argument('--disable-box-refiner', action='store_true', help='Tắt BoundingBoxRefiner (mặc định bật)')
    parser.add_argument('--disable-box-merger', action='store_true', help='Tắt DuplicateBoxMerger (mặc định bật)')
    args, _ = parser.parse_known_args()
    
    # Kiểm tra nếu chức năng test được yêu cầu
    if not args.test_box_refinement:
        return
    
    if not args.image_path:
        print("Vui lòng chỉ định đường dẫn ảnh với --image-path")
        return
    
    print(f"Đang kiểm tra thuật toán tách bounding box trên ảnh: {args.image_path}")
    
    # Đọc ảnh
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ: {args.image_path}")
        return
    
    # Tải model
    model_data = load_model(args.model, verbose=True)
    
    # Khởi tạo bộ tinh chỉnh bounding box
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
    
    # Khởi tạo bộ hợp nhất box trùng lặp
    box_merger = None
    if not args.disable_box_merger:
        box_merger = DuplicateBoxMerger(
            overlap_threshold=0.3,
            containment_threshold=0.8,
            center_dist_threshold=0.5,
            confidence_boost=1.05
        )
    
    print("Đang xử lý ảnh với model YOLO...")
    
    # Xử lý ảnh với cả 4 trường hợp: có/không refiner, có/không merger
    # 1. Không refiner, không merger
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
    
    # 2. Có refiner, không merger
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
    
    # 3. Không refiner, có merger
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
    
    # 4. Có refiner, có merger
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
    
    # Hiển thị kết quả
    cv2.imshow("Original YOLO", result_img1)
    cv2.imshow("BoxRefiner Only", result_img2)
    cv2.imshow("BoxMerger Only", result_img3)
    cv2.imshow("BoxRefiner + BoxMerger", result_img4)
    
    print(f"1. Original: {len(detections1)} đối tượng")
    print(f"2. BoxRefiner: {len(detections2)} đối tượng")
    print(f"3. BoxMerger: {len(detections3)} đối tượng")
    print(f"4. BoxRefiner + BoxMerger: {len(detections4)} đối tượng")
    
    # Chờ người dùng nhấn phím bất kỳ
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Đã hoàn tất kiểm tra")

if __name__ == "__main__":
    # Chạy chức năng test nếu được yêu cầu, nếu không chạy chương trình chính
    # Phân tích tham số để kiểm tra --test-box-refinement
    parser = argparse.ArgumentParser(description='Kiểm tra chế độ chạy')
    parser.add_argument('--test-box-refinement', action='store_true', help='Chạy chế độ kiểm tra tách box')
    args, _ = parser.parse_known_args()
    
    if args.test_box_refinement:
        test_box_refinement()
    else:
        pass
    main()