import cv2
from PIL import Image

def anonymize_image(image_path, output_path):
# 1. Обнаружение и размытие лица (Компьютерное зрение)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)

    if img is None:
        print("Ошибка: Не удалось загрузить изображение.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_zone = img[y:y+h, x:x+w]
        # Размываем лицо (коэффициент 99 определяет силу размытия)
        blurred_face = cv2.GaussianBlur(face_zone, (99, 99), 30)
        img[y:y+h, x:x+w] = blurred_face

    # Временно сохраняем обработанную картинку
    temp_path = "temp_result.jpg"
    cv2.imwrite(temp_path, img)

    # 2. Удаление метаданных/EXIF (Информационная безопасность)
    # Открываем изображение через Pillow и сохраняем БЕЗ метаданных
    image = Image.open(temp_path)

    # Очищаем данные exif
    data = list(image.getdata())
    clean_image = Image.new(image.mode, image.size)
    clean_image.putdata(data)

    clean_image.save(output_path)
    print(f"Готово! Лица размыты, метаданные удалены. Файл сохранен как: {output_path}")

# Запуск программы
if __name__ == "__main__":
    # Положите любое фото с лицом в папку проекта под именем 'input.jpg'
    anonymize_image('input.jpg', 'secure_result.jpg')
