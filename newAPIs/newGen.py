import onnx
import numpy as np

fdir = '../testExtract/input/'
onnxfile = 'mnist16x16x16.onnx'
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

##sheep add
const_vars = '\n'
assgin_data = '\n'
##End sheep add

for idx, i in enumerate(inits):

    ##sheeep add
    n_lay = int(idx / 2)

    name = str(i.name)
    const_vars += 'double coef_' + name.split('/')[0] + name.split('/')[1] + '[] = '
    const_vars += str(list(i.double_data)).replace('[','{ ').replace(']',' }') + ';\n'

    if name.find('weights') >= 0:
        print(name)
        assgin_data += '\tnn->wei[{}] = coef_{};\n'.format(n_lay, name.split('/')[0] + name.split('/')[1])
    elif name.find('bias') >= 0:
        assgin_data += '\tnn->bia[{}] = coef_{};\n'.format(n_lay, name.split('/')[0] + name.split('/')[1])

    
    #END sheep add
   
   
    coef += (list(i.double_data))

##
const_vars += '\n'
assgin_data += '\n'
##


coef = str(coef).replace('[','{ ').replace(']',' }')

with open('./__nn_template.c', 'r') as fp:
    ccode = fp.readlines()

outputs = ''
for cline in ccode:
    outputs += cline

# set max Layer to 784 for M4
outputs = outputs.format(numOfLayer, numOfGap, const_vars, 784, '(int [])' + dimOfLayer, assgin_data).replace('lbrace', '{').replace('rbrace', '}')
with open('ver0.0.2-' + onnxfile.split('.')[0] + '.c', 'w') as fp:
    fp.write(outputs)