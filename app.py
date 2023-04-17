import sys
import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
import os
import tensorflow as tf
import numpy as np
import time

app = Flask(__name__)
upload_folder = 'static'
app.config['UPLOAD'] = upload_folder

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('homePage3.html', tensor_input='')

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = os.path.join(app.config['UPLOAD'], secure_filename(f.filename))
        f.save(path)
        img = Image.open(path)
        start = time.time()
        results = model(img, size=640)
        end = time.time()
        print(end - start)
        results.render()
        Image.fromarray(results.ims[0]).save(os.path.join('static',  f.filename))
        bboxes = results.xyxy[0].tolist()
        sorted_bboxes = sorted(bboxes, key=lambda x: x[0])
        numb_str = ''
        count = 0
        for bbox in sorted_bboxes:
            numb_str += str(int(bbox[-1]))
            count += 1
            if count % 4 == 0:
                numb_str += ' '
        path = os.path.join(upload_folder, f.filename)
        print(path)
    return render_template('homePage3.html', title=numb_str, image=path)

if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt', force_reload=True)
    app.run(host='127.0.0.1', debug=True)

