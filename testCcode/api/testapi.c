#include "onnx2c.h"
#include <stdio.h>
#include <assert.h>

int main()
{
    printf("hello world\n");

    _NN_ nn;
    int input_size, i, output_size;
    double input[1005], output[15], test2Darr[10][10];

    printf("Start NN \n");
    Init_NN(&nn);
    printf("End NN init \n");
    input_size = GetNNInputSize(&nn);
    output_size = GetNNOutputSize(&nn);
    printf("Input Layer is %d\n", input_size);

    FILE *fp = (FILE *) fopen("../general/td1.txt", "r");
    assert(fp != NULL);
    Norm_NN(&nn);

    while(!feof(fp))
    {
        for(i = 0 ; i < input_size ; i++)
            fscanf(fp, "%lf", &input[i]);

        Test_NN(&nn, input, output);
        
        for(i = 0 ; i < output_size ; i++)
            printf("[%d]:\t%lf\n", i, output[i]);
    }
    Free_NN(&nn);

    
    return 0;
}