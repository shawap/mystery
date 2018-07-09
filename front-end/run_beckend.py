import onnx
import numpy as np
from onnx_tf.backend import prepare

model = onnx.load('mnist16x16x16.onnx')
tf_rep = prepare(model)

img = np.load("./assets/image.npz")
output = tf_rep.run(img.reshape([1, 784]))
print ("The digit is classified as {1} ".format(np.argmax(output)))