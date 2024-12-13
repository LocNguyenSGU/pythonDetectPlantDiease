from flask import Flask, render_template, request
import os
import test

app = Flask(__name__)

# Thiết lập thư mục lưu trữ ảnh tải lên
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Kiểm tra định dạng ảnh hợp lệ
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route hiển thị form và xử lý upload ảnh
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Lưu ảnh vào thư mục uploads
            prediction_percentages , name = test.appTest(filepath);
            return render_template('index.html', filename=filename ,  percent = prediction_percentages , name = name)  # Hiển thị ảnh đã tải lên
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])  # Tạo thư mục uploads nếu chưa có
    app.run(debug=True)
