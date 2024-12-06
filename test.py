import cv2
import streamlit as st

# Đường dẫn đến các tệp YOLO
classes_file = r"D:\AI\yolov3.cfg"  # Đường dẫn tệp chứa lớp
weights_file = r"D:\AI\yolov3.weights"  # Đường dẫn tệp trọng số
config_file = r"D:\AI\yolov3.txt"  # Đường dẫn tệp cấu hình

try:
    # Kiểm tra tệp cfg trước khi tải mô hình
    with open(config_file, 'r') as f:
        for idx, line in enumerate(f):
            if '=' not in line and not line.startswith('#') and line.strip():
                st.error(f"Lỗi ở dòng {idx + 1}: {line.strip()}")
    
    # Tải mô hình YOLO
    net = cv2.dnn.readNet(weights_file, config_file)
    st.success("Mô hình YOLO tải thành công!")
except cv2.error as e:
    st.error(f"OpenCV Error: {e}")
except Exception as ex:
    st.error(f"Unexpected Error: {ex}")
