#include "onnx2c.h"
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>


const int LAYER_DIM[NUM_LAYER] = {784, 16, 16, 16, 10};

__attribute__((always_inline)) inline double RELU(const double x){ return x > 0 ? x : 0; }

int RunModel(double *input, double *output)
{
    int err = 0;
    _nn nn;
    init_nn(&nn);
    
    err = test_nn(&nn, input, output);
    if(err) goto error;


    free_nn(&nn);

    return 0;

error:
    printf("Get something wrong...\n");
    return err;
}

int init_nn(_nn *nn)
{
    printf("start Input1\n");
    scanf("%d", &(nn->numlay));
    //nn->numlay = NUM_LAYER;
    assert(MAX_LAYER >= nn->numlay);

    int i;
    int j;
    int err = 0;
    nn->maxdim = -1;
    // dimension infos
        printf("start Input2\n");
    for(i = 0; i < nn->numlay; i++)
    {    
        //nn->laydim[i] = LAYER_DIM[i];
        scanf("%d", &(nn->laydim[i]));

        nn->maxdim = nn->maxdim > nn->laydim[i] ? nn->maxdim : nn->laydim[i];
    }
    // extra space for storing results
    nn->temp[0] = (double *) malloc(sizeof(double) * nn->maxdim);
    assert(nn->temp[0] != NULL);

    nn->temp[1] = (double *) malloc(sizeof(double) * nn->maxdim);
    assert(nn->temp[1] != NULL);

    // construct nns
        printf("start Input3\n");
    for(i = 0; i < nn->numlay - 1; i++)
    {
        int wdim = nn->laydim[i] * nn->laydim[i + 1];
        int bdim = nn->laydim[i + 1];
        nn->wei[i] = (double *) malloc(sizeof(double) * wdim);
        assert(nn->wei[i] != NULL);
        nn->bia[i] = (double *) malloc(sizeof(double) * bdim);
        assert(nn->bia[i] != NULL);


        /**     have to change to hard code     **/
        for(j = 0; j < wdim; j++)
        {
            scanf("%lf", &(nn->wei[i][j]));
        }
        for(j = 0; j < bdim; j++)
        {
            scanf("%lf", &(nn->bia[i][j]));
        }
    }

    return err;
}

int GetNNInputSize(_nn* nn)
{
    assert(nn != NULL);
    return nn->laydim[0];
}

int GetNNOutputSize(_nn* nn)
{
    assert(nn != NULL);
    return nn->laydim[nn->numlay - 1];
}


int test_nn(_nn *nn, double *input, double *output)
{
    //FILE *fp = (FILE *) fopen(testFile, "r");
   // assert(fp != NULL);
    int i;
    int t;
    int m;
    int n;
    int k;
    int iptdim;
    int optdim;
    int err = 0;
    double temp;
    iptdim = nn->laydim[0];

    printf("start test\n");
    for(i = 0, t = 0; i < iptdim ; i++)
    {
        //fscanf(fp, "%lf", &temp);
        //nn->temp[t & 1][i] = temp;
        nn->temp[t & 1][i] = input[i];
       // if(i != 0 && i % iptdim == 0) // every single test;
        if((i + 1) % iptdim == 0)
        {
            //i = 0;
            for(m = 0; m < nn->numlay - 1; m++)
            {
                t++;
                memset(nn->temp[t & 1], 0, sizeof(double) * nn->maxdim);
                // set bias here
                memcpy(nn->temp[t & 1], nn->bia[m], sizeof(double) * nn->laydim[m + 1]); // should be m+1
                
                // weighting for normal i, j, k
                for(k = 0; k < nn->laydim[m + 1]; k++)
                {
                    for(n = 0; n < nn->laydim[m]; n++)
                    {
                        nn->temp[t & 1][k] += nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];
                    }
                }
                // activation functions here;
                for(k = 0; k < nn->laydim[m + 1]; k++)
                {
                    nn->temp[t & 1][k] = RELU(nn->temp[t & 1][k]);
                }
            }
            optdim = nn->laydim[nn->numlay - 1];
            for(k = 0; k < optdim; k++)
            {
                //printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k]);
                output[k] = nn->temp[t & 1][k];
            }
        }
    }
    //fclose(fp);
    return err;
}



void free_nn(_nn *nn)
{
    int i;
    int numgap = nn->numlay - 1;
    for(i = 0; i < numgap; i++)
    {
        free(nn->wei[i]);
        free(nn->bia[i]);
    }
    free(nn->temp[0]);
    free(nn->temp[1]);
}