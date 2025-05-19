#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File test để kiểm tra lỗi debug_mode trong process_video
"""

import streamlit as st
import sys
import os

# Đảm bảo đường dẫn chuẩn để import các module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import từ app.py
from app import process_video, load_model_wrapper

def test_debug_mode():
    """Test debug_mode trong process_video"""
    st.title("Test Debug Mode")
    
    # Đường dẫn tới model và video test
    model_path = st.text_input("Đường dẫn đến model", "last(1)-can-n.pt")
    video_path = st.text_input("Đường dẫn đến video test", "test.mp4")
    
    # Debug mode checkbox
    debug_mode = st.checkbox("Debug Mode", False)
    
    if st.button("Test Process Video"):
        try:
            # Tải model
            st.write("Đang tải model...")
            model_data = load_model_wrapper(model_path)
            
            if model_data:
                # Gọi process_video với debug_mode
                st.write(f"Xử lý video với debug_mode={debug_mode}")
                output_video = process_video(
                    video_path,
                    model_data,
                    conf_threshold=0.25,
                    input_size=640,
                    apply_nms=True,
                    nms_threshold=0.5,
                    center_distance_threshold=0.2,
                    buffer_size=10,
                    iou_threshold=0.5,
                    stability_threshold=0.6,
                    debug_mode=debug_mode
                )
                
                if output_video:
                    st.success("Video được xử lý thành công!")
                    st.video(output_video)
        except Exception as e:
            st.error(f"Lỗi: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    test_debug_mode() 