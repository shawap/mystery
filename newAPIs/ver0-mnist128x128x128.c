#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define NUMBER_OF_LAYER 5 
#define NUMBER_OF_GAP  4 



/*
typedef struct _NN_
{
    int numlay;
    int maxdim;
    int laydim[NUMBER_OF_LAYER];
    double *wei[NUMBER_OF_GAP];
    double *bia[NUMBER_OF_GAP];
    double *temp[2];
    // Future work, function pointers to activation functions
}   _NN_;

void Init_NN(_NN_ *nn);
void Test_NN(_NN_ *nn);
void Free_NN(_NN_ *nn);
*/
__attribute__((always_inline)) inline double RELU(const double x) { return x > 0 ? x : 0; }

int RunModel(double *input, double *output)
{
    int err = 0;
    _NN_ nn;
    Init_NN(&nn);
    
    err = Test_NN(&nn, input, output);
    if(err) goto error;


    Free_NN(&nn);

    return 0;

error:
    printf("Get something wrong...\n");
    return err;
}



int Init_NN(_NN_ *nn)
{
    // dimension infos
    nn->numlay = NUMBER_OF_LAYER;
    nn->maxdim = 128;
    int err = 0;
    memcpy(nn->laydim, (int []){784, 128, 128, 128, 10}, sizeof(int) * NUMBER_OF_LAYER);
    
 
    // extra space for storing results
    nn->temp[0] = (double *) malloc(sizeof(double) * nn->maxdim); assert(nn->temp[0] != NULL);
    nn->temp[1] = (double *) malloc(sizeof(double) * nn->maxdim); assert(nn->temp[1] != NULL);


    // construct nns
    int i, j, ofst = 0;
    for(i = 0; i < NUMBER_OF_GAP; i++)
    {
        int wdim = nn->laydim[i] * nn->laydim[i + 1];
        int bdim = nn->laydim[i + 1];
        nn->wei[i] = (double *) malloc(sizeof(double) * wdim); assert(nn->wei[i] != NULL);
        nn->bia[i] = (double *) malloc(sizeof(double) * bdim); assert(nn->bia[i] != NULL);
        memcpy(nn->wei[i], &coef[ofst], sizeof(double) * wdim);
        ofst += wdim;
        memcpy(nn->bia[i], &coef[ofst], sizeof(double) * bdim);
        ofst += bdim;
    }

    return err;
}


int GetNNInputSize(_NN_* nn)
{
    assert(nn != NULL);
    return nn->laydim[0];
}

int GetNNOutputSize(_NN_* nn)
{
    assert(nn != NULL);
    return nn->laydim[nn->numlay - 1];
}



int Test_NN(_NN_ *nn, double *input, double *output)
{
    //FILE *fp = (FILE *) fopen("td1.txt", "r");
    //assert(fp != NULL);
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

    for(i = 0, t = 0; i < iptdim ; i++)
    {
        //fscanf(fp, "%lf", &temp);
        nn->temp[t & 1][i] = input[i];
        if((i + 1) % iptdim == 0)  // every single test;
        {
            //i = 0;
            for(m = 0; m < nn->numlay - 1; m++)
            {
                t++;
                memset(nn->temp[t & 1], 0, sizeof(double) * nn->maxdim);
                // set bias here
                memcpy(nn->temp[t & 1], nn->bia[m], sizeof(double) * nn->laydim[m + 1]); // should be m+1
                // weighting
                for(n = 0; n < nn->laydim[m]; n++)
                {
                    for(k = 0; k < nn->laydim[m + 1]; k++)
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
                printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k]);
            }
        }
    }
    //fclose(fp);

    return err;
}



int Free_NN(_NN_ *nn)
{
    int i;
    int err = 0;
    int numgap = nn->numlay - 1;
    for(i = 0; i < numgap; i++)
    {
        free(nn->wei[i]);
        free(nn->bia[i]);
    }
    free(nn->temp[0]);
    free(nn->temp[1]);

    return err;
}