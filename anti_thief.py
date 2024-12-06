import cv2
import argparse
import numpy as np
import pyglet
import imutils

# Cài đặt tham số đọc weight, config và class name
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--object_name', required=True, help='path to yolo config file')
ap.add_argument('-f', '--frame', default=5, type=int, help='path to yolo config file')
ap.add_argument('-c', '--config', default='yolov3.cfg', help='path to yolo config file')
ap.add_argument('-w', '--weights', default='yolov3.weights', help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', default='yolov3.txt', help='path to text file containing class names')
args = ap.parse_args()

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

# Đọc từ webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Đọc tên các class
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(args.weights, args.config)

previous_count = 0  # Số lượng điện thoại nhận diện ban đầu

# Bắt đầu đọc từ webcam
try:
    while True:
        # Đọc frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize lại frame cho phù hợp
        image = imutils.resize(frame, width=600)

        # Biến theo dõi sự tồn tại của đối tượng trong khung hình
        isExist = False

        # Resize và đưa khung hình vào mạng predict
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # Lọc các đối tượng trong khung hình
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
                if (confidence > 0.5) and (classes[class_id] == args.object_name):
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

        # Vẽ các khung chữ nhật quanh đối tượng
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            if classes[class_ids[i]] == args.object_name:
                isExist = True
                draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))

        # Đếm số lượng điện thoại hiện diện
        current_count = sum([1 for i in indices if classes[class_ids[i[0]]]==args.object_name])

        # Lưu lại số lượng điện thoại ban đầu
        if previous_count == 0:
            previous_count = current_count

        # Nếu số lượng điện thoại thay đổi (mất 1 điện thoại), báo động
        if current_count < previous_count:
            cv2.putText(image, "Phone is missing!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            try:
                music = pyglet.resource.media('police.wav')
                music.play()
                pyglet.app.run()
            except Exception as e:
                print("Error playing sound:", e)

        previous_count = current_count  # Cập nhật số lượng điện thoại

        cv2.imshow("object detection", image)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()  # Giải phóng webcam
    cv2.destroyAllWindows()
