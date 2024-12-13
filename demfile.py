import cv2
import os

# Đường dẫn tới thư mục chứa hình ảnh đầu vào và thư mục lưu ảnh đầu ra
input_folder = 'dataset/dom_la'  # Thay đổi đường dẫn này

count = 0;
# Lặp qua tất cả các tệp trong thư mục đầu vào
for filename in os.listdir(input_folder):
    count = count + 1;

print("so tep la : " , count);
