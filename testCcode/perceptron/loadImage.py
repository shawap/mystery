from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
x_label = mnist.train.labels

# 印出來看看
chooseImage = 9487
first_train_img = np.reshape(x_train[chooseImage, :], (28, 28))
label = -1
for i in range(0, 10):
    if x_label[chooseImage, i]:
        label = i
print("label is : {}".format(label))
#print(first_train_img)
with open('image_info.txt', 'w') as the_file:
    line = ''
    for i in first_train_img:
        for j in i:
            line += str(j) + ' '
        line += '\n'
    the_file.write(line)

