import tensorflow as tf
from onnx_tf.frontend import tensorflow_graph_to_onnx_model

with tf.gfile.GFile("ckpt/frozen_model.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    onnx_model = tensorflow_graph_to_onnx_model(graph_def,
                                     "output/add",
                                     opset=4)

    file = open("mnist.onnx", "wb")
    file.write(onnx_model.SerializeToString())
    file.close()

    print(onnx_model.graph.node[0])
