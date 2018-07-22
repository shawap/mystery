import onnx
import numpy as np

fdir = '../testExtract/input/'
onnxfile = 'mnist64x64x64.onnx'
fpath = fdir + onnxfile
model = onnx.load(fpath)


inits = []
for init in model.graph.initializer :
    inits.append(init)

weights_bias = len(inits)
print("weights + bias : {}".format(weights_bias))


layerdim = []
for i in range(0, weights_bias):
    if i == weights_bias - 1:
        #print(inits[i].dims[0])
        layerdim.append(inits[i].dims[0])
    elif i % 2 == 0:
        #print(inits[i].dims[0])
        layerdim.append(inits[i].dims[0])

print("Structure of each layer : {}".format(layerdim))


numOfLayer = len(layerdim)
numOfGap = numOfLayer - 1
maxLayer = max(layerdim[1:])
dimOfLayer = str(layerdim).replace('[', '{').replace(']', '}')

coef = []

for i in inits:
    coef += (list(i.double_data))

coef = str(coef).replace('[','{ ').replace(']',' }')

with open('./__nn_template.c', 'r') as fp:
    ccode = fp.readlines()

outputs = ''
for cline in ccode:
    outputs += cline

outputs = outputs.format(numOfLayer, numOfGap, coef, maxLayer, '(int [])' + dimOfLayer).replace('lbrace', '{').replace('rbrace', '}')
with open('ver0-' + onnxfile.split('.')[0] + '.c', 'w') as fp:
    fp.write(outputs)