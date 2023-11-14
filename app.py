from flask import Flask, render_template, request
import utils # Thay thế "your_ml_module" bằng tên module của bạn

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Gọi hàm dự đoán từ mô hình machine learning của bạn
        # result_text, result_number, result_table = your_ml_module.predict(user_input)

        return render_template('index.html', user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
