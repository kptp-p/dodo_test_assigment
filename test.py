from ultralytics import YOLO

# Загрузка модели YOLO26
model = YOLO('yolo26n.pt')  # или yolo26s.pt и т.д.

# Запуск предсказания на изображении
results = model('2.mp4')
