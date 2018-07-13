#include "onnx2c.h"
#include <stdio.h>
#include <assert.h>

int main()
{
    printf("hello world\n");

    _nn nn;
    int input_size, i, output_size;
    double input[1005], output[15];

    printf("Start NN \n");
    init_nn(&nn);
    printf("End NN init \n");
    input_size = GetNNInputSize(&nn);
    output_size = GetNNOutputSize(&nn);
    printf("Input Layer is %d\n", input_size);

    
    FILE *fp = (FILE *) fopen("../general/td1.txt", "r");
    assert(fp != NULL);

    while(!feof(fp))
    {
        for(i = 0 ; i < input_size ; i++)
            fscanf(fp, "%lf", &input[i]);

        test_nn(&nn, input, output);
        for(i = 0 ; i < output_size ; i++)
            printf("[%d]:\t%lf\n", i, output[i]);
    }
    free_nn(&nn);

    
    return 0;
}