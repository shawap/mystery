#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


int layerCnt, INPUT_SIZE, OUTPUT_SIZE;



int main()
{

    scanf("%d%d%d", &layerCnt, &INPUT_SIZE, &OUTPUT_SIZE);
    
    FILE *fptr;
    int i, j, k, maxValue, predict_value[10005], cnt = 0;
    int correct, wrong, correct_num;
    double input_value[INPUT_SIZE], output_weights[OUTPUT_SIZE][INPUT_SIZE], output_bias[OUTPUT_SIZE], predict[OUTPUT_SIZE];
    double correct_percent;
    char c;

    printf("Total Layer is : \t%d\n", layerCnt);
    printf("Input Layer size is : \t%d\n", INPUT_SIZE);
    printf("Output Layer size is : \t%d\n", OUTPUT_SIZE);



    /*  get output layer weights   */
    fptr = fopen("perceptWeight.txt", "r");
    assert(fptr != NULL );
    printf("\nOpen perceptWeight success!\n");

    for(j = 0 ; j < INPUT_SIZE ; j++)
        for(i = 0 ; i < OUTPUT_SIZE ; i++)
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
    assert(fptr != NULL );
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



    /*  get intput layer weight   */
    fptr = fopen("test_images.txt", "r");
    assert(fptr != NULL );
    printf("\nOpen image_info success!\n");

    cnt = 0;
    while(!feof(fptr))
    {    
        for(i = 0 ; i < INPUT_SIZE ; i++)
            fscanf(fptr, "%lf", &input_value[i]);
        printf("Finish \"input value\" input\n");
        
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
            //printf("Label %d : %lf\n", i, predict[i]);
        }
        //printf("\nPredict is %d\n", maxValue);

        predict_value[cnt++] = maxValue;
    }
    fclose(fptr);

    for(i = 0 ; i < cnt ; i++)
        printf("%d\n", predict_value[i]);
    


    fptr = fopen("test_label_list.txt", "r");
    assert(fptr != NULL );

    correct = wrong = 0;
    for(i = 0 ; i < cnt ; i++)
    {
        fscanf(fptr, "%d", &correct_num);
        if(predict_value[i] == correct_num)
            correct++;
        else
            wrong++;
    }
    fclose(fptr);
    correct_percent = (double) correct / (double ) (correct + wrong);
    printf("correct percentage : %.2lf% , correct number is %d.\n", correct_percent, correct);



    return 0;
}