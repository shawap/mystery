#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define MAX_LAYER 16
// 測試檔路徑
const char testFile[] = "../storage/tdx10";

typedef struct _norm
{
    double nw;
    double na;
    double na2;
} _norm;


typedef struct _nn
{
    int numlay;
    int maxdim;
    int laydim[MAX_LAYER];
    double *wei[MAX_LAYER];
    double *bia[MAX_LAYER];
    double *temp[2]; // for calculating
    double compensate;
    struct _norm *factor;
    
    /*
     * future add,
     * func ptr to act func
    */
}   _nn;

void init_nn(_nn *nn);
void norm_nn(_nn *nn);
void test_nn(_nn *nn);
void free_nn(_nn *nn);
__attribute__((always_inline)) inline double RELU(const double x){ return x > 0 ? x : 0; }

int main()
{
    _nn nn;
    init_nn(&nn);
    norm_nn(&nn);
    test_nn(&nn);
    free_nn(&nn);
    return 0;
}


void init_nn(_nn *nn)
{
    nn->factor = NULL; // initialize value for ptr to factors
    scanf("%d", &(nn->numlay));
    assert(MAX_LAYER >= nn->numlay);
    int i;
    int j;
    nn->maxdim = -1;
    // dimension infos
    for(i = 0; i < nn->numlay; i++)
    {
        scanf("%d", &(nn->laydim[i]));
        nn->maxdim = nn->maxdim > nn->laydim[i] ? nn->maxdim : nn->laydim[i];
    }
    // extra space for storing results
    nn->temp[0] = (double *) malloc(sizeof(double) * nn->maxdim);
    assert(nn->temp[0] != NULL);
    nn->temp[1] = (double *) malloc(sizeof(double) * nn->maxdim);
    assert(nn->temp[1] != NULL);
    // construct nns
    for(i = 0; i < nn->numlay - 1; i++)
    {
        int wdim = nn->laydim[i] * nn->laydim[i + 1];
        int bdim = nn->laydim[i + 1];
        nn->wei[i] = (double *) malloc(sizeof(double) * wdim);
        assert(nn->wei[i] != NULL);
        nn->bia[i] = (double *) malloc(sizeof(double) * bdim);
        assert(nn->bia[i] != NULL);
        for(j = 0; j < wdim; j++)
        {
            scanf("%lf", &(nn->wei[i][j]));
        }
        for(j = 0; j < bdim; j++)
        {
            scanf("%lf", &(nn->bia[i][j]));
        }
    }
}


void norm_nn(_nn *nn)
{
    int numOfLay = nn->numlay; // Note that it includes the input layer now
    int numOfGap = numOfLay - 1;

    // memory allocation for normalization factors
    nn->factor = (_norm *) malloc(sizeof(_norm) * numOfGap);
    assert(nn->factor != NULL);

    int idxi, idxj, idxm, idxn, idxk;
    int bdim, wdim, numOfNeuron;
    double tempSum, tempMax, tempVal;

    for(idxi = 0; idxi < numOfGap; idxi++)
    {
        // get dimension infos at first
        bdim = nn->laydim[idxi + 1];
        wdim = nn->laydim[idxi] * nn->laydim[idxi + 1];
        // (1) normalize bias at the first palce (by previous layers' nas & na2s)
        // (1-1) from i=0 ~ current - 2, bias /= (na * nw)
        // (1-2) from i=current - 1 if i >= 0, bias /= (na2 * nw)
        for(idxj = 0; idxj < idxi - 1; idxj++)
        {
            for(idxk = 0; idxk < bdim; idxk++)
            {
                nn->bia[idxi][idxk] /= (nn->factor[idxj].na * nn->factor[idxj].nw);
            }
        }
        for(idxj = idxi - 1; idxj >= 0; idxj = -1)
        {
            for(idxk = 0; idxk < bdim; idxk++)
            {
                nn->bia[idxi][idxk] /= (nn->factor[idxj].na2 * nn->factor[idxj].nw);
            }
        }
        // (2) select the maximum among biases and weights
        // (2-1) initialize tempMax to an extreme small figure
        // (2-2) pick maximum of abs(BIASES, WEIGHTS), should avoid using abs()
    
        tempMax = -(1e9 + 10);
        // traverse weights
        for(idxk = 0; idxk < wdim; idxk++)
        {
            tempVal = nn->wei[idxi][idxk] >= 0 ? nn->wei[idxi][idxk] : -nn->wei[idxi][idxk];
            tempMax = tempMax > tempVal ? tempMax : tempVal;
        }

        // traverse biases
        for(idxk = 0; idxk < bdim; idxk++)
        {
            tempVal = nn->bia[idxi][idxk] >= 0 ? nn->bia[idxi][idxk] : -nn->bia[idxi][idxk];
            tempMax = tempMax > tempVal ? tempMax : tempVal;
        }

        // get maximum nw
        nn->factor[idxi].nw = tempMax;
        // change the original data directly for testing purposes
        // tuning weights
        for(idxk = 0; idxk < wdim; idxk++)
        {
            nn->wei[idxi][idxk] /= tempMax;
        }
        // tuning biases
        for(idxk = 0; idxk < bdim; idxk++)
        {
            nn->bia[idxi][idxk] /= tempMax;
        }



        // (3) find the na & na2 for each layers (based on idxi)
        // (3-1) get number of neurons at begin, which should be identical to length of current bias vector
        // (3-2) load the bias vector to tempVec, and do weights accumulation on it
        numOfNeuron = bdim;
        memset(nn->temp[0], 0, sizeof(double) * nn->maxdim);
        memcpy(nn->temp[0], nn->bia[idxi], sizeof(double) * bdim); // same as multiply by numOfNeurons

        // accumulations
        for(idxk = 0; idxk < wdim; idxk++)
        {
            nn->temp[0][idxk % bdim] += nn->wei[idxi][idxk] > 0 ? nn->wei[idxi][idxk] : 0; // add only those who are bigger than zero
        }
        
        // find the maximum as current na
        // remember to reset tempMax to an extreme small value
        tempMax = -(1e9 + 10);
        for(idxk = 0; idxk < bdim; idxk++)
        {
            tempMax = tempMax > nn->temp[0][idxk] ? tempMax : nn->temp[0][idxk];
        }
        nn->factor[idxi].na = tempMax;
        // find the current na2
        nn->factor[idxi].na2 = pow(2, ceil(log2(tempMax)));


        
    }
    // (4) get compensate factor
    nn->compensate = 1;
    for(idxk = 0; idxk < numOfGap - 1; idxk++)
    {
        nn->compensate *= (nn->factor[idxk].nw * nn->factor[idxk].na);
    }
    nn->compensate *= (nn->factor[idxk].nw * nn->factor[idxk].na2);
    /*
    FILE *fp = fopen("./factors", "w");
    for(idxi = 0; idxi < numOfGap; idxi++)
    {
        fprintf(fp, "%lf,\t%lf\t%lf\n", nn->factor[idxi].nw, nn->factor[idxi].na, nn->factor[idxi].na2);
    }
    fprintf(fp, "%lf\n", nn->compensate);
    fclose(fp);
    */
}




void test_nn(_nn *nn)
{
    FILE *fp = (FILE *) fopen(testFile, "r");
    assert(fp != NULL);
    int i;
    int t;
    int m;
    int n;
    int k;
    int iptdim;
    int optdim;
    double temp;
    iptdim = nn->laydim[0];

    for(i = 0, t = 0; !feof(fp); i++)
    {
        fscanf(fp, "%lf", &temp);
        nn->temp[t & 1][i] = temp;
        if((i + 1) % iptdim == 0) // every single test;
        {
            i = -1;
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

                // normalization testers, 0814
                if(m == 0)
                {
                    for(k = 0; k < nn->laydim[m + 1]; k++)
                    {
                        nn->temp[t & 1][k] /= nn->factor[m].na2;
                    }
                }
                else if(m > 0)
                {
                    for(k = 0; k < nn->laydim[m + 1]; k++)
                    {
                        nn->temp[t & 1][k] = nn->temp[t & 1][k] * nn->factor[m - 1].na2 / nn->factor[m - 1].na / nn->factor[m].na2;
                    }
                }
                // 0814 edition end



                // activation functions here;
                for(k = 0; k < nn->laydim[m + 1]; k++)
                {
                    nn->temp[t & 1][k] = RELU(nn->temp[t & 1][k]);
                    if(nn->temp[t & 1][k] > 1 || nn->temp[t & 1][k] < -1)
                    {
                        printf("fuck!\n");
                    }
                    else
                    {
                        printf("%lf, ",nn->temp[t & 1][k]);
                    }
                }
            }
            optdim = nn->laydim[nn->numlay - 1];
            for(k = 0; k < optdim; k++)
            {
                printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k] * nn->compensate);
            }
        }
    }
    fclose(fp);
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
    if(nn->factor != NULL)
    {
        free(nn->factor);
    }
  
}
