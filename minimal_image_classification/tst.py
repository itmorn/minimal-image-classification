import base64
import cv2
import numpy as np

np_img = cv2.imread("images/20715152805.png")

retval, buffer = cv2.imencode('.png', np_img)
pic_str = base64.b64encode(buffer)
pic_str = pic_str.decode()


img_data = base64.b64decode(pic_str)
nparr = np.frombuffer(img_data, np.uint8)
img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
print(np_img==img_np)
