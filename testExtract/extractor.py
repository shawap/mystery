import onnx
from onnx_tf.backend import prepare

fpath = './input/mnist-perceptron.onnx'
model = onnx.load(fpath)
print(model)
