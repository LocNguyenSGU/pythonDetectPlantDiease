from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = (256, 256)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0/255,          # Chuẩn hóa pixel thành khoảng 0-1
    validation_split=0.2,     # Chia 80% dữ liệu cho huấn luyện và 20% cho kiểm tra
    rotation_range=20,        # Tăng cường dữ liệu
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    '.venv/dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'         # Sử dụng cho huấn luyện
)

val_data = datagen.flow_from_directory(
    '.venv/dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'       # Sử dụng cho kiểm tra
)
