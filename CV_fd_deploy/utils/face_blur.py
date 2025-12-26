# utils/face_blur.py
import cv2
import numpy as np

def blur_faces(image, model, blur_strength=25):
    """
    Размывает все лица на изображении
    
    Args:
        image: numpy array изображения (BGR или RGB)
        model: загруженная YOLO модель
        blur_strength: сила размытия (нечетное число)
    
    Returns:
        tuple: (размытое изображение, количество обнаруженных лиц)
    """
    try:
        # Убедимся, что blur_strength нечетное
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Копируем изображение чтобы не модифицировать оригинал
        result_image = image.copy()
        
        # Детекция лиц
        results = model(image)
        
        faces_count = 0
        
        # Размытие каждого обнаруженного лица
        for r in results:
            for box in r.boxes:
                # Получаем координаты bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Проверяем уверенность (опционально)
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box, 'conf') else 1.0
                
                # Пропускаем лица с низкой уверенностью (опционально)
                if conf < 0.5:
                    continue
                
                # Увеличиваем счетчик лиц
                faces_count += 1
                
                # Вырезаем область лица
                face_region = result_image[y1:y2, x1:x2]
                
                # Проверяем, что область не пустая
                if face_region.size > 0:
                    # Применяем Гауссово размытие
                    blurred = cv2.GaussianBlur(
                        face_region, 
                        (blur_strength, blur_strength), 
                        0
                    )
                    # Заменяем оригинальную область размытой
                    result_image[y1:y2, x1:x2] = blurred
        
        return result_image, faces_count
        
    except Exception as e:
        print(f"Ошибка в blur_faces: {e}")
        # В случае ошибки возвращаем оригинальное изображение
        return image, 0