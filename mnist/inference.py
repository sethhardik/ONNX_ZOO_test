from configparser import Interpolation
import sys
import json
import cv2
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper


model = 'mnist-12.onnx' #model file
path = sys.argv[1] # image path

# preprocessing the image
img = cv2.imread(path)
img = np.dot(img[...,:3], [0.299, 0.587, 0.114]) # convert to grayscale
img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
img.resize((1,1,28,28)) # batch of image

data = json.dumps({"data":img.tolist()})
data = np.array(json.loads(data)["data"]).astype(np.float32)
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

results = session.run([output_name], {input_name: data})
preds = int(np.argmax(np.array(results).squeeze(),axis = 0))
print("Predictions: {}".format(preds))