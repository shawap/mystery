#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define MAX_LAYER 16
// 測試檔路徑
const char testFile[] = "../storage/td1.txt";

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
    _norm *factor;
    /*
     * future add,
     * func ptr to act func
    */
}   _nn;

void init_nn(_nn *nn);
void test_nn(_nn *nn);
void free_nn(_nn *nn);
__attribute__((always_inline)) inline double RELU(const double x){ return x > 0 ? x : 0; }

int main()
{
    _nn nn;
    init_nn(&nn);
    test_nn(&nn);
    free_nn(&nn);
    return 0;
}


void init_nn(_nn *nn)
{
    scanf("%d", &(nn->numlay));
    assert(MAX_LAYER >= nn->numlay);
    // malloc for the factors of normalizations
    nn->factor = (_norm *) malloc(sizeof(_norm) * nn->numlay);
    assert(nn->factor != NULL);

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
        double tempMax = -1e9, tempNeuron, tempSum;
        int wdim = nn->laydim[i] * nn->laydim[i + 1], k, n;
        int bdim = nn->laydim[i + 1];
        nn->wei[i] = (double *) malloc(sizeof(double) * wdim);
        assert(nn->wei[i] != NULL);
        nn->bia[i] = (double *) malloc(sizeof(double) * bdim);
        assert(nn->bia[i] != NULL);
        // avoid using abs!
        for(j = 0; j < wdim; j++)
        {
            scanf("%lf", &(nn->wei[i][j]));
            tempNeuron = nn->wei[i][j] > 0 ? nn->wei[i][j] : -nn->wei[i][j];
            tempMax = tempMax > tempNeuron ? tempMax : tempNeuron;
        }
        for(j = 0; j < bdim; j++)
        {
            scanf("%lf", &(nn->bia[i][j]));

            //tempNeuron = nn->bia[i][j] > 0 ? nn->bia[i][j] : -nn->bia[i][j];
            //tempMax = tempMax > tempNeuron ? tempMax : tempNeuron;
        }

        //bias 要先normalize
        if(i > 0)
        {
            for(j = 0; j < bdim; j++)
            {
                for(k = 0 ; k <= i - 1 ; k++)
                {
                    nn->bia[i][j] /=  nn->factor[k].nw / nn->factor[k].na2;
                }
                tempNeuron = nn->bia[i][j] > 0 ? nn->bia[i][j] : -nn->bia[i][j];
                tempMax = tempMax > tempNeuron ? tempMax : tempNeuron;
            }
        }
        /*
        for(n = i - 2; n >= 0; n--)
        {
            for(j = 0; j < bdim; j++)
            {
                nn->bia[i][j] /=  nn->factor[n].nw / nn->factor[n].na; 
            }
        }
        */


        // max(max(BIAS), max(WEIGHT))
        nn->factor[i].nw = tempMax;
        

        for(j = 0; j < wdim; j++)
        {
            nn->wei[i][j] /= nn->factor[i].nw;
        }
        for(j = 0; j < bdim; j++)
        {
            nn->bia[i][j] /= nn->factor[i].nw;
        }


        //get na
        double tempNA = 0;
        for(j = 0 ; j < nn->laydim[i] ; j++)    //在這層neuron個數
        {
            tempSum = 0;
            for(k = 0 ; k < nn->laydim[i + 1] ; k++)    //前一層neuron個數
            {
                //add all positive weights
                tempSum += nn->wei[i][j * nn->laydim[i + 1] + k] > 0 ? nn->wei[i][j * nn->laydim[i + 1] + k] : 0 ; 
            }
            tempSum += nn->bia[i][j];
            if(i > 0)
            {
                for(k = 0 ; k <= i - 1 ; k++)
                {
                    tempSum = tempSum * nn->factor[k].na2 / nn->factor[k].na;
                }
            }
            tempNA = tempNA > tempSum ? tempNA : tempSum;
        }
        
        nn->factor[i].na = tempNA;
        // next layer's estimated domain
        nn->factor[i].na2 = pow(2, ceil(log2(tempNA)));




    }

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

                // set bias here, already diviede by 'nw'
                memcpy(nn->temp[t & 1], nn->bia[m], sizeof(double) * nn->laydim[m + 1]); // should be m+1
                
                // tuning BIAS according to normalize factor from layers ahead
               

                // weighting
                for(n = 0; n < nn->laydim[m]; n++)
                {
                    for(k = 0; k < nn->laydim[m + 1]; k++)
                    {
                        nn->temp[t & 1][k] += nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];
                    }
                }

                
                // normalize with na2 before activation
                for(k = 0; k < nn->laydim[m + 1]; k++)
                {
                    nn->temp[t & 1][k] /= nn->factor[m].na2;                   
                }

                // activation functions here;
                for(k = 0; k < nn->laydim[m + 1]; k++)
                {
                    nn->temp[t & 1][k] = RELU(nn->temp[t & 1][k]);
                }   
            }
            optdim = nn->laydim[nn->numlay - 1];
            double compensate = 1;
            for(k = 0; k < nn->numlay - 1; k++)
            {
                compensate *= (nn->factor[k].nw * nn->factor[k].na2);
            }
    
            //compensate *= (nn->factor[k].nw * nn->factor[k].na2);
            for(k = 0; k < optdim; k++)
            {
                printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k] * compensate);
            }
            /*
            printf("Compensate is %lf\n", compensate);
            for(k = 0; k < 4; k++)
            {
                printf("%lf, %lf, %lf\n", nn->factor[k].nw, nn->factor[k].na, nn->factor[k].na2);
            }
            */
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
   
    free(nn->factor);
}
