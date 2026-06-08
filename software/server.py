from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import io
import base64

# 导入模型
from model import CNN

app = Flask(__name__)

#启动
device = torch.device("cpu")
model = CNN().to(device)
# 加载训练好的权重
model.load_state_dict(torch.load("model/fhe_friendly_cnn.pth"))
model.eval() # 切换到推理模式

#转化
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# 路由设计

# 窗口1：在浏览器输入网址时，发出网页
@app.route('/')
def index():
    return render_template('index.html')

# 窗口2：点击“识别”时，接收图片并算答案
@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = ImageOps.invert(image)
        
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            prediction = output.argmax(dim=1, keepdim=True).item()
            
        return jsonify({'prediction': prediction, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

if __name__ == '__main__':
    print("Server is loading open: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)