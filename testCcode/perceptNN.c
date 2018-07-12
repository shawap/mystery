#include <stdio.h>
#include <stdlib.h>


int layerCnt, INPUT_SIZE, OUTPUT_SIZE;



int main()
{

    scanf("%d%d%d", &layerCnt, &INPUT_SIZE, &OUTPUT_SIZE);
    
    FILE *fptr;
    int i, j, k, maxValue;
    double input_value[INPUT_SIZE], output_weights[OUTPUT_SIZE][INPUT_SIZE], output_bias[OUTPUT_SIZE], predict[OUTPUT_SIZE];
    char c;

    printf("Total Layer is : \t%d\n", layerCnt);
    printf("Input Layer size is : \t%d\n", INPUT_SIZE);
    printf("Output Layer size is : \t%d\n", OUTPUT_SIZE);



    /*  get intput layer weight   */
    fptr = fopen("image_info.txt", "r");
    if(fptr != NULL)
        printf("\nOpen image_info success!\n");

    for(i = 0 ; i < INPUT_SIZE ; i++)
        fscanf(fptr, "%lf", &input_value[i]);
    printf("Finish \"input value\" input\n");
    fclose(fptr);

    /*  get output layer weights   */
    fptr = fopen("perceptWeight.txt", "r");
    if(fptr != NULL)
        printf("\nOpen perceptWeight success!\n");

    for(i = 0 ; i < OUTPUT_SIZE ; i++)
        for(j = 0 ; j < INPUT_SIZE ; j++)
            fscanf(fptr, "%lf", &output_weights[i][j]);
    printf("Finish \"output Weights\" input\n");
    fclose(fptr);

    // test weight
    /*
    for(i = 0 ; i < 30 ; i++)
        printf("%lf\n", output_weights[0][i]);
    */

    /*  get output layer bias   */
    fptr = fopen("perceptBias.txt", "r");
    if(fptr != NULL)
        printf("\nOpen perceptBias success!\n");
        
    for(i = 0 ; i < OUTPUT_SIZE ; i++)
        fscanf(fptr,"%lf", &output_bias[i]);
    printf("Finish \"output Bias\" input\n\n");
    fclose(fptr);
    
    
    // test weight
    /*
    for(i = 0 ; i < 10 ; i++)
        printf("%lf\n", output_bias[i]);
    */

    /*  compute   */
    for(i = 0 ; i < OUTPUT_SIZE ; i++)
    {
        predict[i] = 0;
        for(j = 0 ; j < INPUT_SIZE ; j++)
        {
            predict[i] +=  output_weights[i][j] * input_value[j] ;
        }
        predict[i] += output_bias[i];
    }


    maxValue = 0;
    for(i = 0 ; i < OUTPUT_SIZE ; i++)
    {
        if(predict[i] > predict[maxValue])
            maxValue = i;
        printf("Label %d : %lf\n", i, predict[i]);
    }
    printf("\nPredict is %d\n", maxValue);
    return 0;
}