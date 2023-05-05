import os
import random
import shutil
import zipfile

import requests
from imageai.Classification import ImageClassification

# Отримання поточної директорії
execution_path = os.getcwd()

# Ініціалізація моделі та її завантаження
prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
# Вказуємо абсолютний шлях до файлу з моделлю
model_path = os.path.join(execution_path, 'pretrained', 'resnet50-19c8e357.pth')
prediction.setModelPath(model_path)
prediction.loadModel()

# Завантажуємо датасет
dataset_path = os.path.join(execution_path, 'fruits-360.zip')
if not os.path.exists(dataset_path):
    # встановлюємо URL та ім'я файлу, яке хочемо зберегти
    dataset_url = 'https://www.kaggle.com/datasets/moltean/fruits/download?datasetVersionNumber=9'

    # робимо запит за файлом та зберігаємо його
    response = requests.get(dataset_url)
    with open(dataset_path, 'wb') as dataset_f:
        dataset_f.write(response.content)

# Розпаковуємо датасет
dataset_extracted_path = os.path.join(execution_path, 'fruits-360')
if not os.path.exists(dataset_extracted_path):
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_extracted_path)

# Використання абсолютного шляху до теки з зображеннями
image_path = os.path.join(dataset_extracted_path, 'fruits-360_dataset', 'fruits-360',
                          'test-multiple_fruits')
image_list = os.listdir(image_path)
# Перемішуємо список зображень
random.shuffle(image_list)

# Класифікація кількох зображень
for image in image_list[:5]:
    # Використовуємо абсолютний шлях до зображення
    predictions, probabilities = prediction.classifyImage(os.path.join(image_path, image), result_count=5)
    print(image, predictions, probabilities)

# Видаляємо розпакований датасет
shutil.rmtree(dataset_extracted_path)
