import cv2
import numpy as np
import imutils
import pyglet
from flask import Flask, render_template, Response
import threading

# Tạo đối tượng Flask
app = Flask(__name__)

# Đọc tên các class
classes_file = "D:\AI\yolov3.cfg"  # Đường dẫn tới tệp chứa các lớp đối tượng
weights_file = "D:\AI\yolov3.weights"  # Đường dẫn tới tệp weights của YOLO
config_file = "D:\AI\yolov3.txt"  # Đường dẫn tới tệp cấu hình YOLO

classes = None
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(weights_file, config_file)

# Hàm trả về output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Hàm vẽ các hình chữ nhật và tên class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Cấu hình webcam
cap = cv2.VideoCapture(0)

# Hàm tạo video stream
def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = imutils.resize(frame, width=600)

        # Phát hiện đối tượng
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if (confidence > 0.5) and (classes[class_id] == 'cell phone'):  # Chỉnh lại tên đối tượng
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            if classes[class_ids[i]] == 'cell phone':
                draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))

        # Chuyển đổi ảnh thành định dạng mà Flask có thể sử dụng
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Trang chính
@app.route('/')
def index():
    return render_template('index.html')  # Tạo một template index.html để hiển thị webcam

# Route để stream video
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Khởi động server Flask
if __name__ == '__main__':
    app.run(debug=True)
