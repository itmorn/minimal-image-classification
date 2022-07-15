import os

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch
from flask import Flask
from flask import request, jsonify
import base64
import numpy as np
import cv2
import json
import base64
from gevent import pywsgi

app = Flask(__name__)


@app.route('/aa', methods=["POST"])
def post_demo():
    dic_client = request.json
    pic_str = dic_client.get("img")

    img_data = base64.b64decode(pic_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img2 = np.transpose(img_np, [2, 0, 1])[::-1].copy()
    img2 = torch.from_numpy(img2)

    return method_name(img2)


def method_name(img2):
    img2 = img2.to(device)
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img2).unsqueeze(0)
    # batch = batch.to(device)
    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")
    return jsonify({"ok": 1})


def init():
    img2 = torch.randn(3, 224, 224)
    img2 = img2.to(device)
    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img2).unsqueeze(0)
    # batch = batch.to(device)
    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")


def set_process_gpu():
    worker_id = int(os.environ.get('APP_WORKER_ID', 1))
    # devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')

    # if not devices:
    #     print('current environment did not get CUDA_VISIBLE_DEVICES env ,so use the default')

    # rand_max = 9527
    # print(worker_id)
    # gpu_index = (worker_id + rand_max) % torch.cuda.device_count()
    if worker_id<=4:
        gpu_index = 0
    else:
        gpu_index = 1
    print('current worker id  {} set the gpu id :{}'.format(worker_id, gpu_index))
    torch.cuda.set_device(int(gpu_index))

set_process_gpu()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = model.to(device)
model.eval()
preprocess = weights.transforms()
init()


print("done")
if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 7832), app)
    server.serve_forever()
    # app.run(host="0.0.0.0", port=7832, debug=False)
    # app.run(host="127.0.0.1", port=7832, debug=False)
