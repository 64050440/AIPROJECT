import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
import cv2
import datetime
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

yolo = Create_Yolo(input_size=416, CLASSES="person_names.txt")
yolo.load_weights("checkpoints/yolov3_custom")
height = 224
width = 224
last_detection_time = None  

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/tester')
def tester():
    return render_template('tester.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    global last_detection_time
    try:
        
        image_path = "static/img/image5.jpg"
        # โหลดรูปภาพด้วย OpenCV
        image = cv2.imread(image_path)
        # แปลงสีของรูปภาพเป็น RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ปรับขนาดรูปภาพให้เป็น 416x416
        resized_image = cv2.resize(image, (416, 416))
        resized_image = np.expand_dims(resized_image, axis=0)

        image = detect_image(yolo, image_path, "", input_size=416, show=False, CLASSES="person_names.txt", rectangle_colors=(255,0,0))

        #predictions = yolo.predict(resized_image)
        print("prediction data type : ",type(image))
        # Update last_detection_time with the current time


        last_detection_time = datetime.datetime.now()

        return str(image), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/num', methods=['GET'])
def num():
    global last_detection_time
    try:
        
        image_path = "static/img/image5.jpg"
        # โหลดรูปภาพด้วย OpenCV
        image = cv2.imread(image_path)
        # แปลงสีของรูปภาพเป็น RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ปรับขนาดรูปภาพให้เป็น 416x416
        resized_image = cv2.resize(image, (416, 416))
        resized_image = np.expand_dims(resized_image, axis=0)

    
        predictions = yolo.predict(resized_image)

        # อ่านชื่อคลาสจากไฟล์ person_names.txt
        with open('person_names.txt', 'r') as file:
            class_names_from_file = [line.strip() for line in file.readlines()]

        person_count = 0
        bag_count = 0
          # นับจำนวนวัตถุที่ตรวจจับได้ในแต่ละคลาส

        for class_name, confidence in zip(class_names_from_file, predictions):
            # ถ้าคลาสเป็น "person" และความมั่นใจมากกว่าค่าที่กำหนด
            if class_name == "person" and confidence.any() > 0.5:
            # เพิ่มจำนวนคนที่ตรวจจับได้
                person_count += 1
            elif class_name == "bag" and confidence.any() > 0.5:
                bag_count += 1

        # สร้าง JSON response ที่ระบุชื่อคลาสและจำนวนที่ตรวจจับได้
        result = [{"class": "person", "count": person_count},{"class": "bag", "count": bag_count}]

        # Update last_detection_time with the current time
        last_detection_time = datetime.datetime.now()


        return jsonify(result=result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400  


@app.route('/api/predict_pic', methods=['GET'])
def predict_pic():
    global last_detection_time
    try:
        
        image_path = "static/img/image5.jpg"
        # โหลดรูปภาพด้วย OpenCV
        image = cv2.imread(image_path)
        # แปลงสีของรูปภาพเป็น RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # ปรับขนาดรูปภาพให้เป็น 416x416
        resized_image = cv2.resize(image, (416, 416))
        resized_image = np.expand_dims(resized_image, axis=0)

        image = detect_image(yolo, image_path, "", input_size=416, show=False, CLASSES="person_names.txt", rectangle_colors=(255,0,0))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # บันทึกรูปภาพลงในไฟล์
        output_image_path = "static/img/detected_image.jpg"
        cv2.imwrite(output_image_path, image)

        #predictions = yolo.predict(resized_image)
        print("prediction data type : ",type(image))
        # Update last_detection_time with the current time

          # ส่งชื่อไฟล์ของรูปภาพที่ถูกบันทึกกลับ
        response_data = {
            "image_path": output_image_path
        }

        last_detection_time = datetime.datetime.now()

        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
        
        

def draw_boxes(image, predictions):
    # วาดกล่องสี่เหลี่ยมลงบนวัตถุที่ตรวจจับได้
    for prediction in predictions:
        class_name, confidence, (x, y, w, h) = ("person", 0.95, (100, 200, 50, 80))#prediction
        color = (255, 0, 0)  # สีของกล่องสี่เหลี่ยม (RGB: 255, 0, 0 คือสีแดง)
        thickness = 2  # ความหนาของเส้นของกล่องสี่เหลี่ยม
        font_scale = 1  # ขนาดของตัวหนังสือ

        # กำหนดข้อความที่จะแสดงในกล่องสี่เหลี่ยม
        label = f'{class_name}: {confidence:.2f}'
        # คำนวณขนาดของข้อความ
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # คำนวณตำแหน่งเริ่มต้นของกล่องสี่เหลี่ยม
        text_offset_x = x
        text_offset_y = y - 5
        # ถ้าตำแหน่ง y ติดกับขอบบนของรูปให้เลื่อนข้อความลงมา
        if text_offset_y < 0:
            text_offset_y = y + h + 15
        # วาดกล่องสี่เหลี่ยม
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        # วาดข้อความ
        cv2.putText(image, label, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image    

@app.route('/api/last_detection_time', methods=['GET'])
def get_last_detection_time():
    global last_detection_time
    if last_detection_time:
        return jsonify({'last_detection_time': last_detection_time.strftime('%Y-%m-%d %H:%M:%S')})
    else:
        return jsonify({'last_detection_time': 'ไม่มีการตรวจจับ'})

if __name__ == '__main__':
    app.run(debug=True)

