import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Đọc ảnh và chuyển thành mảng numpy
img = cv2.imread('ImgTest/khoe_manh/0a68ef5a-027c-41ae-b227-159dae77d3dd___R.S_HL 7969 copy.jpg')
input_matrix = np.array(img)
print("Layer 0 (shape):", img.shape)
print("Layer 0:")
print(input_matrix)
width, height , sau = img.shape

# Reshape để phù hợp với đầu vào của Conv2D trong TensorFlow: (batch, height, width, channels)
input_matrix = input_matrix.reshape((1, width, height, 3))  # 1 là batch_size

# Xây dựng model với lớp đầu vào Input
inputs = tf.keras.layers.Input(shape=(width, height, 3))
conv_output = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv_layer')(inputs)
# relu_output = tf.keras.layers.ReLU()(conv_output)
# maxpool_output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(relu_output)
# conv_output_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv_layer_1')(maxpool_output)
# relu_output_1 = tf.keras.layers.ReLU()(conv_output_1)
# maxpool_output_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(relu_output_1)
# flatten_output = tf.keras.layers.Flatten()(maxpool_output_1)
# dense_1_output = tf.keras.layers.Dense(32, activation='relu', name='dense_1')(flatten_output)  # Thêm lớp Dense
# dense_2_output = tf.keras.layers.Dense(10, activation='softmax', name='dense_2')(dense_1_output)  # Lớp Dense cuối với 10 đầu ra

# Tạo model phụ để lấy đầu ra của conv_output
intermediate_model = tf.keras.Model(inputs=inputs, outputs=conv_output)

# Chạy tính toán và lấy đầu ra của conv_output
conv_output_result = intermediate_model.predict(input_matrix)

print("Conv layer output shape:", conv_output_result.shape)

# Vẽ 32 ảnh của conv_output
plt.figure(figsize=(15, 10))
for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(conv_output_result[0, :, :, i])  # Vẽ feature map i
    plt.title(f'Filter {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()


