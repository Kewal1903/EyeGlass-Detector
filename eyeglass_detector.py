import os
import cv2 as cv
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout

base_dir = 'eyeglass_detector'
glasses_dir = os.path.join(base_dir, 'glasses')
no_glasses_dir = os.path.join(base_dir, 'no_glasses')
categories = ['glasses', 'no_glasses']

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = {'face': [], 'glasses': []}
    for obj in root.findall('object'):
        label = obj.find('name').text.lower()
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        if label == 'face':
            objects['face'].append((xmin, ymin, xmax, ymax))
        if label == 'glasses':
            objects['glasses'].append((xmin, ymin, xmax, ymax))
    return objects


def preprocess_images_with_bboxes(data_dir, categories, img_size):
    data = []
    labels = []
    bboxes = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        for file in os.listdir(path):
            if file.endswith(".jpg" or ".jpeg"):
                img_path = os.path.join(path, file)
                xml_path = os.path.join(path, file.replace(".jpg" or ".jpeg", ".xml"))
                if not os.path.exists(xml_path):
                    continue
                img_array = cv.imread(img_path)
                resized_array = cv.resize(img_array, (img_size, img_size))
                data.append(resized_array)
                labels.append(class_num)
                objects = parse_xml(xml_path)
                bboxes.append(objects)
    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels, bboxes

img_size = 64
data, labels, bboxes = preprocess_images_with_bboxes(base_dir, categories, img_size)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

train_datagen = ImageDataGenerator(
    rotation_range=80,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip = True,
    brightness_range = [0.2,1.0],
    channel_shift_range = 30,
    fill_mode='nearest')

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_and_classify(frame, model):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        resized_face = cv.resize(face, (img_size, img_size)).reshape(1, img_size, img_size, 3) / 255.0
        prediction = model.predict(resized_face)
        label = "Glasses" if np.argmax(prediction) == 0 else "No Glasses"

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, "Face", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if label == "Glasses":
            face_resized = cv.resize(face, (img_size, img_size))
            face_resized_gray = cv.cvtColor(face_resized, cv.COLOR_BGR2GRAY)
            glasses_bboxes = face_cascade.detectMultiScale(face_resized_gray, 1.1, 4)
            for (gx, gy, gw, gh) in glasses_bboxes:
                gx_min = int(gx * w / img_size)
                gy_min = int(gy * h / img_size)
                gx_max = int((gx + gw) * w / img_size)
                gy_max = int((gy + gh) * h / img_size)
                cv.rectangle(frame, (x + gx_min, y + gy_min), (x + gx_max, y + gy_max), (0, 255, 255), 2)
                cv.putText(frame, "Glasses", (x + gx_min, y + gy_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2)
    return frame


cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_and_classify(frame, model)
    cv.imshow('Webcam - Glasses Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
