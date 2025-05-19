#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper để chạy app Streamlit với tùy chọn tắt file watching
để tránh lỗi với torch._classes trên Python 3.13
"""

import os
import sys
import subprocess

def main():
    """Chạy ứng dụng Streamlit với các tùy chọn cần thiết"""
    # Đường dẫn tới app.py
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Kiểm tra app.py tồn tại
    if not os.path.exists(app_path):
        print(f"Không tìm thấy file app.py tại: {app_path}")
        return 1
    
    # Tìm đường dẫn tới Streamlit
    try:
        import streamlit
        streamlit_cmd = [sys.executable, "-m", "streamlit"]
    except ImportError:
        print("Streamlit không được cài đặt! Đang cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        streamlit_cmd = [sys.executable, "-m", "streamlit"]
    
    # Thiết lập biến môi trường để tắt file watching
    os.environ["STREAMLIT_SERVER_FILE_WATCHDOG"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCH_POLL"] = "false"
    # Đảm bảo biến môi trường PYTHONPATH chứa thư mục hiện tại để import run_yolo.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Xây dựng lệnh chạy Streamlit
    cmd = streamlit_cmd + [
        "run", 
        app_path, 
        "--server.fileWatcherType", "none",
        "--server.runOnSave", "false"
    ]
    
    print("====================================================")
    print("Chạy Streamlit với tùy chọn tắt file watching")
    print("Điều này giúp tránh lỗi trên Python 3.13 với torch._classes")
    print("====================================================")
    print(f"Lệnh: {' '.join(cmd)}")
    
    # Chạy Streamlit
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main()) 