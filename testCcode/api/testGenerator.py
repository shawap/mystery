import onnx
import numpy as np

fdir = '../../testExtract/input/'
onnxfile = 'mnist-perceptron.onnx'
fpath = fdir + onnxfile
model = onnx.load(fpath)
#print(model)
print(type(model))

nodes = []

for node in model.graph.node:
    #print(node)
    nodes.append(node)

num_node = len(nodes)
print("Total node : {}".format(num_node))
inits = []

for init in model.graph.initializer :
    #print(init)
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

print("each layer tensor counts : {}".format(layerdim))
num_layer = len(layerdim)


headers = '#include "onnx2c.h"\n#include <stdio.h>\n#include <assert.h>\n#include <stdlib.h>\n#include <string.h>\n'
defines = '\n#define NUM_LAYER '+ str(num_layer) + '\n'
constvar = '\nconst int LAYER_DIM[NUM_LAYER] = { '
for idx, dim in enumerate(layerdim):
    constvar += str(dim)
    if idx != num_layer - 1:
        constvar += ', '
constvar += ' };\n'

macros = '\n__attribute__((always_inline)) inline double RELU(const double x){ return x > 0 ? x : 0; }\n'

run_model_function = '\nint RunModel(double *input, double *output)\n\
{\n\
    int err = 0;\n\
    _NN_ nn;\n\
    Init_NN(&nn);\n\
    \n\
    err = Test_NN(&nn, input, output);\n\
    if(err) goto error;\n\
\n\
\n\
    Free_NN(&nn);\n\
\n\
    return 0;\n\
\n\
error:\n\
    printf("Get something wrong...\\n");\n\
    return err;\n\
}'


init_nn_function = '\n\
int Init_NN(_NN_ *nn)\n\
{\n\
    printf("Start init\\n");\n\
    //scanf("%d", &(nn->numlay));\n\
    nn->numlay = NUM_LAYER;\n\
    assert(MAX_LAYER >= nn->numlay);\n\
\n\
    int i;\n\
    int j;\n\
    int err = 0;\n\
    nn->maxdim = -1;\n\
    // dimension infos\n\
    //printf("start Input2\\n");\n\
    for(i = 0; i < nn->numlay; i++)\n\
    {    \n\
        nn->laydim[i] = LAYER_DIM[i];\n\
        //scanf("%d", &(nn->laydim[i]));\n\
\n\
        nn->maxdim = nn->maxdim > nn->laydim[i] ? nn->maxdim : nn->laydim[i];\n\
    }\n\
    // extra space for storing results\n\
    nn->temp[0] = (double *) malloc(sizeof(double) * nn->maxdim);\n\
    assert(nn->temp[0] != NULL);\n\
\n\
    nn->temp[1] = (double *) malloc(sizeof(double) * nn->maxdim);\n\
    assert(nn->temp[1] != NULL);\n\
\n\
    // construct nns\n\
    //    printf("start Input3\\n");\n\
'
var_wei_bia = '\n\tdouble wei_bia[' + str(2 * (num_layer - 1)) + '][' + str(layerdim[0] * layerdim[1]) + '] = {'
for idx, init in enumerate(inits):
    layer_idx = int(idx / 2)
    arr = '{'
    size_of_doubledata = len(init.double_data)
    for idxx, doubles in enumerate(init.double_data):
        arr += str(doubles)
        if idxx != size_of_doubledata - 1:
            arr += ', '
        
    arr += ' }'
    if idx != weights_bias - 1:
        arr += ', '
    """
    if layer_idx == num_layer - 1:
        break
    if idx % 2 != 0:
        #arr = 'bias' + str(layer_idx) + '[' + str(layerdim[layer_idx]) + '] = { '
        #print(init.double_data)
        for idx, doubles in enumerate(init.double_data):
            arr += str(doubles)
            if idx != layerdim[layer_idx] - 1:
                arr += ', '
            
        arr += ' }\n\n'
        print(arr)
    else:
        arr = 'weight'+ str(layer_idx) + '[' + str(layerdim[layer_idx] * layerdim[layer_idx + 1]) + '] = { '
        
        for idx, doubles in enumerate(init.double_data):
            arr += str(doubles)
            if idx != layerdim[layer_idx] * layerdim[layer_idx + 1]- 1:
                arr += ', '
        arr += ' }\n\n'
    """

    var_wei_bia += arr

var_wei_bia += '};\n\n'
init_nn_function += var_wei_bia
init_nn_function += '\n\
    for(i = 0; i < nn->numlay - 1; i++)\n\
    {\n\
        int wdim = nn->laydim[i] * nn->laydim[i + 1];\n\
        int bdim = nn->laydim[i + 1];\n\
        nn->wei[i] = (double *) malloc(sizeof(double) * wdim);\n\
        assert(nn->wei[i] != NULL);\n\
        nn->bia[i] = (double *) malloc(sizeof(double) * bdim);\n\
        assert(nn->bia[i] != NULL);\n\
\n\
\n\
        /**     have to change to hard code     **/\n\
        for(j = 0; j < wdim; j++)\n\
        {\n\
            nn->wei[i][j] = wei_bia[i * 2][j];\n\
        }\n\
        for(j = 0; j < bdim; j++)\n\
        {\n\
            nn->bia[i][j] = wei_bia[i * 2 + 1][j];\n\
        }\n\
    }\n\
\n\
    return err;\n\
}\n\
'

size_function = '\n\
int GetNNInputSize(_NN_* nn)\n\
{\n\
    assert(nn != NULL);\n\
    return nn->laydim[0];\n\
}\n\
\n\
int GetNNOutputSize(_NN_* nn)\n\
{\n\
    assert(nn != NULL);\n\
    return nn->laydim[nn->numlay - 1];\n\
}\n\
\n\
'

test_function = '\n\
int Test_NN(_NN_ *nn, double *input, double *output)\n\
{\n\
    //FILE *fp = (FILE *) fopen(testFile, "r");\n\
   // assert(fp != NULL);\n\
    int i, t, m, n, k;\n\
    int iptdim, optdim;\n\
    int err = 0;\n\
    double temp;\n\
    iptdim = nn->laydim[0];\n\
\n\
    printf("start test\\n");\n\
    for(i = 0, t = 0; i < iptdim ; i++)\n\
    {\n\
        //fscanf(fp, "%lf", &temp);\n\
        //nn->temp[t & 1][i] = temp;\n\
        nn->temp[t & 1][i] = input[i];\n\
       // if(i != 0 && i % iptdim == 0) // every single test;\n\
        if((i + 1) % iptdim == 0)\n\
        {\n\
            //i = 0;\n\
            for(m = 0; m < nn->numlay - 1; m++)\n\
            {\n\
                t++;\n\
                memset(nn->temp[t & 1], 0, sizeof(double) * nn->maxdim);\n\
                // set bias here\n\
                memcpy(nn->temp[t & 1], nn->bia[m], sizeof(double) * nn->laydim[m + 1]); // should be m+1\n\
                \n\
                // weighting for normal i, j, k\n\
                for(k = 0; k < nn->laydim[m + 1]; k++)\n\
                {\n\
                    for(n = 0; n < nn->laydim[m]; n++)\n\
                    {\n\
                        nn->temp[t & 1][k] += nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];\n\
                    }\n\
                }\n\
                // activation functions here;\n\
                for(k = 0; k < nn->laydim[m + 1]; k++)\n\
                {\n\
                    nn->temp[t & 1][k] = RELU(nn->temp[t & 1][k]);\n\
                }\n\
            }\n\
            optdim = nn->laydim[nn->numlay - 1];\n\
            for(k = 0; k < optdim; k++)\n\
            {\n\
                //printf("[%d]:\\t%lf\\n", k, nn->temp[t & 1][k]);\n\
                output[k] = nn->temp[t & 1][k];\n\
            }\n\
        }\n\
    }\n\
    //fclose(fp);\n\
    return err;\n\
}\n\
'

free_function = '\n\
void Free_NN(_NN_ *nn)\n\
{\n\
    int i;\n\
    int numgap = nn->numlay - 1;\n\
    for(i = 0; i < numgap; i++)\n\
    {\n\
        free(nn->wei[i]);\n\
        free(nn->bia[i]);\n\
    }\n\
    free(nn->temp[0]);\n\
    free(nn->temp[1]);\n\
}'







with open("version0-onnx2c-"+ onnxfile.split('.')[0] + ".c", "w") as f:
    f.write(headers)
    f.write(defines)
    f.write(constvar)
    f.write(macros)
    f.write(run_model_function)
    f.write(init_nn_function)
    f.write(size_function)
    f.write(test_function)
    f.write(free_function)
    f.close()