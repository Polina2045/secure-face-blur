import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_file, redirect
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загружаем уже имеющуюся рабочую модель YOLOv8
model = YOLO('yolov8n.pt')

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return False

    h_img, w_img, _ = img.shape
    
    # Запускаем распознавание объектов (ищем людей)
    results = model(input_path, verbose=False)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Класс 0 в модели YOLO — это человек (person)
            if int(box.cls) == 0:
                # Извлекаем координаты тела
                xyxy = box.xyxy.cpu().numpy()[0]
                bx1, by1, bx2, by2 = map(int, xyxy)
                
                person_w = bx2 - bx1
                person_h = by2 - by1
                
                # КОРРЕКТИРОВКА:
                # Делаем зону размытия чуть больше (30% от высоты), чтобы захватить подбородок
                face_h = int(person_h * 0.30)
                face_w = int(person_w * 0.80)
                
                # Центрируем по горизонтали
                face_x1 = bx1 + int((person_w - face_w) / 2)
                
                # ОПУСКАЕМ РАМКУ: смещаем верхнюю точку вниз на 8% от высоты силуэта, 
                # чтобы размытие ложилось ровно на глаза и нос, а не на волосы
                face_y1 = by1 + int(person_h * 0.08)
                
                # Защита: ограничиваем координаты краями фотографии
                face_x1, face_y1 = max(0, face_x1), max(0, face_y1)
                face_w = min(face_w, w_img - face_x1)
                face_h = min(face_h, h_img - face_y1)

                
                if face_w > 5 and face_h > 5:
                    # Вырезаем область лица
                    face_zone = img[face_y1:face_y1+face_h, face_x1:face_x1+face_w]
                    # Делаем аккуратное размытие по Гауссу
                    blurred_face = cv2.GaussianBlur(face_zone, (99, 99), 20)
                    img[face_y1:face_y1+face_h, face_x1:face_x1+face_w] = blurred_face

    # Сохраняем промежуточный результат обработки
    temp_path = os.path.join(UPLOAD_FOLDER, 'temp.jpg')
    cv2.imwrite(temp_path, img)

    # ИБ-очистка скрытых метаданных (EXIF) через Pillow
    image = Image.open(temp_path)
    data = list(image.getdata())
    clean_image = Image.new(image.mode, image.size)
    clean_image.putdata(data)
    clean_image.save(output_path)
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_user.jpg')
            file.save(input_path)
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'secure_user.jpg')
            
            success = process_image(input_path, output_path)
            if success:
                return send_file(output_path, as_attachment=True, download_name='secure_photo.jpg')
            else:
                return "Ошибка при обработке изображения."
    return render_template('index.html')

if __name__ == '__main__':
    # Локально приложение работает на порту 5000. На Hugging Face Spaces
    # платформа сама задаёт порт 7860 через переменную окружения PORT.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
