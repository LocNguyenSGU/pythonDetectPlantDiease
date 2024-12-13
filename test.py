from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import cv2


def appTest(img_path):
    def resize_to_256x256(image):
        """
        Chuyển ảnh về kích thước 256x256. Nếu ảnh nhỏ hơn thì thêm viền,
        nếu lớn hơn thì cắt viền.

        Args:
            image: numpy.ndarray - Ảnh đầu vào.

        Returns:
            resized_image: numpy.ndarray - Ảnh kích thước 256x256.
        """
        target_size = 256
        h, w = image.shape[:2]

        if h == target_size and w == target_size:
            return image

        # Tính toán viền cần thêm
        top, bottom, left, right = 0, 0, 0, 0

        if h < target_size:
            top = (target_size - h) // 2
            bottom = target_size - h - top

        if w < target_size:
            left = (target_size - w) // 2
            right = target_size - w - left

        # Thêm viền nếu cần
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # Cắt viền nếu kích thước vượt quá target_size
        h, w = image.shape[:2]  # Cập nhật kích thước sau khi thêm viền
        if h > target_size:
            top = (h - target_size) // 2
            bottom = top + target_size
        else:
            top, bottom = 0, h

        if w > target_size:
            left = (w - target_size) // 2
            right = left + target_size
        else:
            left, right = 0, w

        if h > target_size or w > target_size:
            image = image[top:bottom, left:right]
        print(image.shape);
        return image

    # Tải mô hình đã lưu
    model = tf.keras.models.load_model('best_model.keras')

    # Đọc ảnh với OpenCV và resize
    # img_path = 'ImgTest/khoe_manh/0a72d779-4df2-4365-9251-1733a1a1085c___R.S_HL 7992 copy 2.jpg'
    img = cv2.imread(img_path)  # Đọc ảnh gốc
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh tại đường dẫn: {img_path}")

    img_resized = resize_to_256x256(img)  # Resize về 256x256

    # cv2.imshow("ádfasdfas" , img_resized);
    # cv2.waitKey(0);

    # Chuẩn hóa ảnh
    img_array = img_resized.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

    # Dự đoán
    prediction = model.predict(img_array)

    # Danh sách các lớp bệnh
    class_names = ['BẠC LÁ', 'ĐỐM LÁ', 'KHOẺ MẠNH', 'RỈ SÉT']

    # In phần trăm dự đoán cho từng loại bệnh
    print("Kết quả dự đoán:")
    prediction_percentages = [
        f'{class_name}: {prediction[0][i] * 100:.2f}%' for i, class_name in enumerate(class_names)
    ]
    for item in prediction_percentages:
        print(item)

    # Xác định lớp có xác suất cao nhất
    predicted_class = np.argmax(prediction)
    print(f'\nTình trạng lá ngô dự đoán: {class_names[predicted_class]}')

    return prediction_percentages , class_names[predicted_class] ;



