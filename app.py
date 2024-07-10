#连续手语识别系统后端
from flask import Flask, render_template, request, Response
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from logits import translate  # 假设这是一个自定义模块，用于处理视频翻译

app = Flask(__name__)

def generate_frames(video_path, pred_result):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    position = (300, 630)  # 字幕位置
    textColor = (255, 255, 255)  # 字幕颜色
    textSize = 40  # 字幕大小

    while True:
        success, frame = video.read()  # 读取视频帧
        if not success:
            break
        else:
            # 将OpenCV图像转换为PIL图像，以使用PIL处理字幕
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            # 使用中文字体
            fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
            draw.text(position, pred_result, textColor, font=fontText)
            
            # 将PIL图像转换回OpenCV图像
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            
            # 将帧编码为JPEG格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # 生成MJPEG流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

@app.route('/')
def index():
    return render_template('index.html')  # 渲染首页

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        # 检查文件名是否为空
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file:
            # 保存上传的文件到服务器指定目录
            video_path = os.path.join('tmps', file.filename)
            file.save(video_path)
            return render_template('index.html', message='File uploaded successfully', video_path=video_path)

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path', None)  # 获取视频路径
    if video_path:
        pred_result = translate(video_path)  # 调用翻译函数处理视频
        pred_result = ' '.join(pred_result)  # 将结果拼接成字符串
        print(pred_result)
        return Response(generate_frames(video_path, pred_result), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Error: No video path provided."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)  # 启动服务器
