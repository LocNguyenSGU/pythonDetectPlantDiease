# Để load lịch sử huấn luyện từ file JSON
with open('history.json', 'r') as f:
    history_data = json.load(f)

# Sau đó bạn có thể vẽ biểu đồ
plot_accuracy_and_loss(history_data)
