import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau
import albumentations as A

# Config
img_dir = 'letters'
img_size = 64
num_aug = 10

augmenter = A.Compose([
    A.Rotate(limit=15, p=0.9),
    A.RandomScale(scale_limit=0.1, p=0.8),
    A.Perspective(scale=(0.02, 0.05), p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.RandomBrightnessContrast(p=0.5)
])

X, y = [], []
all_images = glob(os.path.join(img_dir, '*_blue.jpg'))

classes = sorted(list(set([os.path.basename(f)[0] for f in all_images])))
label_map = {cls: i for i, cls in enumerate(classes)}

for img_path in all_images:
    label_char = os.path.basename(img_path)[0]
    label = label_map[label_char]

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (img_size, img_size))
    X.append(img)
    y.append(label)

    for _ in range(num_aug):
        aug = augmenter(image=img)
        aug_img = aug['image']
        if aug_img.shape != (img_size, img_size):
            aug_img = cv2.resize(aug_img, (img_size, img_size))
        X.append(aug_img)
        y.append(label)

X = [cv2.resize(x, (img_size, img_size)) if x.shape[:2] != (img_size, img_size) else x for x in X]
X = np.array(X).reshape(-1, img_size, img_size, 1).astype('float32') / 255.0
y = to_categorical(y, num_classes=len(classes))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,1)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[lr_schedule])

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {acc * 100:.2f}%')
