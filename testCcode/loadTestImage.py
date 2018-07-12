from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_test = mnist.test.images
x_label = mnist.test.labels

print(np.shape(x_test[1,:]))

select_num = input('Input a number for the number of test images\n')
print(select_num)
select_num = int(select_num)


img_list = ''
label_list = ''
for i in range(0, select_num):
    string = ''
    for num in x_test[i,:]:
        string += str(num) + ' '
    string += '\n'
    img_list += string
for i in range(0, select_num):
    label = 0
    for num in range(0, 10):
        if x_label[i, num] :
            label = num
    string = str(label) + '\n'
    label_list += string
with open("test_images.txt", "w") as f:
    f.write(img_list)
with open("test_label_list.txt", "w") as f:
    f.write(label_list)