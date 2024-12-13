import os
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from PIL import Image
import tensorflow as tf
import json

# Thiết lập các tham số
IMG_SIZE = (256, 256)  # Kích thước hình ảnh
BATCH_SIZE = 32  # Kích thước lô
EPOCHS = 20  # Số lượng epochs
DATASET_DIR = 'dataset'  # Thư mục chứa dữ liệu
NUM_CLASSES = 4  # Số lượng lớp

# Hàm xóa ảnh không hợp lệ
def remove_invalid_images(dataset_dir):
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError, Image.UnidentifiedImageError):
                print(f"Removing invalid image: {file_path}")
                os.remove(file_path)

# Lọc ảnh không hợp lệ
remove_invalid_images(DATASET_DIR)

# Hàm tạo dữ liệu

def create_data_generators(dataset_dir, img_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='reflect',
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data

# Hàm xây dựng mô hình
def build_model(img_size, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hàm Learning Rate Scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)  # Đảm bảo trả về kiểu float
    else:
        return float(lr * tf.math.exp(-0.1).numpy())  # Chuyển giá trị Tensor thành float


# Hàm huấn luyện mô hình
def train_model(model, train_data, val_data, epochs, checkpoint_dir='best_model.keras'):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_dir, save_best_only=True, monitor='val_loss')
    lr_schedule = LearningRateScheduler(lr_scheduler)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, lr_schedule]
    )
    return history

# Hàm vẽ biểu đồ lịch sử huấn luyện
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Mất mát
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

# Chạy toàn bộ quy trình
train_data, val_data = create_data_generators(DATASET_DIR, IMG_SIZE, BATCH_SIZE)
model = build_model(IMG_SIZE, NUM_CLASSES)
history = train_model(model, train_data, val_data, EPOCHS)
plot_history(history)
# Sau khi huấn luyện mô hình, lưu lịch sử vào file JSON
with open('history.json', 'w') as f:
    json.dump(history.history, f)