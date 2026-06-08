from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFile
import io
import base64
import subprocess
import re
import time

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__, template_folder=r"C:\Users\jia\Desktop\CKKS-MNIST\software\templates")

# 图像预处理（对齐要求的 mean=0.1307, std=0.3081）
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


PROJECT_DIR = r"C:\Users\jia\Desktop\CKKS-MNIST\FHE\GPU\ckks-mnist"
DATA_TXT_PATH = r"C:\Users\jia\Desktop\CKKS-MNIST\FHE\GPU\ckks-mnist\data\images.txt"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. 接收并解析前端发来的本地上传或固定样例图片
        data = request.get_json()
        raw_b64 = data.get('image', '')
        if ',' in raw_b64:
            raw_b64 = raw_b64.split(',')[1]
            
        # 拿到纯净的 Base64 后进行安全解码
        image_bytes = base64.b64decode(raw_b64)
        
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        if data.get('isCanvas', False):
            image = ImageOps.invert(image)
            
        tensor = transform(image) # 得到 [1, 28, 28] 的张量
        
        # 2. 展平覆进data/images.txt
        # 将单张图片转化为文本
        pixel_values = tensor.flatten().tolist()
        img_line = "0 " + " ".join(f"{val:.6f}" for val in pixel_values)
        
        with open(DATA_TXT_PATH, "w", encoding="utf-8", newline='\n') as f:
            f.write("1\n")           
            f.write(img_line + "\n") 
            
        # 3. 调用WSL穿透指令执行 CUDA 密文推理
        start_time = time.time()
        
        # WSL2 启动命令
        wsl_command = (
            'wsl --cd "/mnt/c/Users/jia/Desktop/CKKS-MNIST/FHE/GPU/ckks-mnist" -e bash -lc '
            '"export PATH=\\"/usr/local/cuda-12.6/bin:\$PATH\\"; '
            'export CUDACXX=/usr/local/cuda-12.6/bin/nvcc; '
            'cd build && '
            'export LD_LIBRARY_PATH=\\"\$PWD/phantom-build/lib:/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH\\"; '
            './ckks_mnist ../data 1"' 
        )
        
        # 执行命令并捕获控制台输出
        process = subprocess.run(wsl_command, shell=True, capture_output=True, text=True, encoding='utf-8')
        
        real_duration_ms = int((time.time() - start_time) * 1000)
        console_output = process.stdout
        
        # 4. 正则表达式解析终端日志
        # 预期输出格式：img0: label=0 pred=7 logits=[...]
        match = re.search(r"pred=(\d+)", console_output)
        if match:
            prediction = int(match.group(1))
        else:
            # 如果没匹配到，C++ 层报错，控制台错误输出
            raise Exception(f"CUDA引擎未能成功输出预测值。日志：{console_output}")
            
        # 5. 返回真实密文数据给前端展示
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'real_time_ms': real_duration_ms,
            'log': f"WSL2 穿透成功。密文模数链层级: 9层, 硬件架构: RTX 4060 (sm_89)。\n终端回显:\n{console_output.strip()}"
        })

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)