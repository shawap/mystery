#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define MAX_LAYER           16

#define FIXPOINT            0

#if FIXPOINT
int MAX_DECIMAL_BIT =    0;
int MAX_FLOAT_BIT   =   31;       
double fix(double x);  
#endif

// 測試檔路徑
const char testFile[] = "./test_images.txt";
const char labels[] = "./test_label_list.txt";

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
    int i;
#if FIXPOINT
    for(i = 1 ; i <= 31 ; i++)
    {
        MAX_FLOAT_BIT = i;
#endif
        _nn nn;
        init_nn(&nn);
        norm_nn(&nn);
        test_nn(&nn);
        free_nn(&nn);
#if FIXPOINT
    }
#endif
    return 0;
}


void init_nn(_nn *nn)
{
    FILE * fptr = (FILE*) fopen("../storage/16x16x16", "r");
    //scanf("%d", &(nn->numlay));
    fscanf(fptr, "%d", &(nn->numlay));
    assert(MAX_LAYER >= nn->numlay);
    int i;
    int j;
    nn->maxdim = -1;
    nn->factor = NULL;
    // dimension infos
    for(i = 0; i < nn->numlay; i++)
    {
        fscanf(fptr, "%d", &(nn->laydim[i]));
        //scanf("%d", &(nn->laydim[i]));
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
            fscanf(fptr, "%lf", &(nn->wei[i][j]));
            //scanf("%lf", &(nn->wei[i][j]));
        }
        for(j = 0; j < bdim; j++)
        {
            fscanf(fptr, "%lf", &(nn->bia[i][j]));
            //scanf("%lf", &(nn->bia[i][j]));
        }
    }

    fclose(fptr);
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

    // get the nw, na, na2
    FILE * fptr = (FILE *) fopen("factors", "r");

    for(idxi = 0 ; idxi < numOfGap ; idxi++)
    {
        fscanf(fptr, "%lf", &nn->factor[idxi].nw);
        fscanf(fptr, "%lf", &nn->factor[idxi].na);
        fscanf(fptr, "%lf", &nn->factor[idxi].na2);
        //printf("[%d] nw is %lf, na is %lf, na2 is %lf\n", idxi, nn->factor[idxi].nw, nn->factor[idxi].na, nn->factor[idxi].na2);
    }
    fscanf(fptr, "%lf", &nn->compensate);

    //printf("%lf\n", nn->compensate);

    // normalize the weight and bias
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
        
        // change the original data directly for testing purposes
        // tuning weights
        for(idxk = 0; idxk < wdim; idxk++)
        {
            nn->wei[idxi][idxk] /= nn->factor[idxi].nw;
        }
        // tuning biases
        for(idxk = 0; idxk < bdim; idxk++)
        {
            nn->bia[idxi][idxk] /= nn->factor[idxi].nw;
        }
        
    }


    // fix point
#if FIXPOINT
    for(idxi = 0 ; idxi < numOfGap ; idxi++)
    { 
        // get dimension infos at first
        bdim = nn->laydim[idxi + 1];
        wdim = nn->laydim[idxi] * nn->laydim[idxi + 1];

        for(idxj = 0 ; idxj < wdim ; idxj++)
            nn->wei[idxi][idxj] = fix(nn->wei[idxi][idxj]);
        for(idxj = 0 ; idxj < bdim ; idxj++)
            nn->bia[idxi][idxj] = fix(nn->bia[idxi][idxj]);
    }
#endif

    fclose(fptr);
}

void test_nn(_nn *nn)
{
    FILE *fp = (FILE *) fopen(testFile, "r");
    FILE *flab = (FILE *) fopen(labels, "r");
    assert(fp != NULL);
    int i;
    int t;
    int m;
    int n;
    int k;
    int iptdim;
    int optdim, preIdx, ans, nCorrect = 0, nTestcase = 0;
    double temp, preVal;
    iptdim = nn->laydim[0];

    for(i = 0, t = 0; !feof(fp); i++)
    {
        fscanf(fp, "%lf", &temp);
        nn->temp[t & 1][i] = temp;
        if((i + 1) % iptdim == 0) // every single test;
        {
            i = -1;

            nTestcase ++;
            preVal = -1e9+10;
            preIdx = -1;
            
            for(m = 0; m < nn->numlay - 1; m++)
            {
                t++;
                memset(nn->temp[t & 1], 0, sizeof(double) * nn->maxdim);
                // set bias here
                memcpy(nn->temp[t & 1], nn->bia[m], sizeof(double) * nn->laydim[m + 1]); // should be m+1

                /*** sheep move  ****/
                if(nn->factor != NULL)
                {
                    if(m == 0)
                    {
                        for(k = 0; k < nn->laydim[m + 1]; k++)
                        {
                            nn->temp[t & 1][k] /= nn->factor[m].na2;
                            //nn->temp[t & 1][k] = fix(nn->temp[t & 1][k]);
                        }
                    }
                    else if(m > 0)
                    {
                        for(k = 0; k < nn->laydim[m + 1]; k++)
                        {
                            nn->temp[t & 1][k] = nn->temp[t & 1][k] * nn->factor[m - 1].na2 / nn->factor[m - 1].na / nn->factor[m].na2;
                            //nn->temp[t & 1][k] = fix(fix(fix(nn->temp[t & 1][k]) * fix(nn->factor[m - 1].na2)) / fix(fix(nn->factor[m - 1].na) / fix(nn->factor[m].na2)));
                            //nn->temp[t & 1][k] = fix(nn->temp[t & 1][k]);
                        }
                    }
                }
                /********************/
                
                // weighting
                for(n = 0; n < nn->laydim[m]; n++)
                {
                    for(k = 0; k < nn->laydim[m + 1]; k++)
                    {
                         /*** sheep add  ****/
                        if(nn->factor != NULL)
                        {
                            double tmp = nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];
#if FIXPOINT
                            tmp = fix(tmp);
#endif                
                            assert(tmp < 1);
                            
                            if(m == 0)
                            {
                                tmp /= nn->factor[m].na2;
                                nn->temp[t & 1][k] += tmp;
                            }
                            else if(m > 0)
                            {
                                tmp = tmp * nn->factor[m - 1].na2 / nn->factor[m - 1].na / nn->factor[m].na2;
                                nn->temp[t & 1][k] += tmp;
                            }
                            assert(nn->temp[t & 1][k] < 1 );
                        }
                        /********************/

                        /** original **/
                        else    nn->temp[t & 1][k] += nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];
                        //printf("hello\n");
                    }
                }


                // activation functions here;
                for(k = 0; k < nn->laydim[m + 1]; k++)
                {
                    nn->temp[t & 1][k] = RELU(nn->temp[t & 1][k]);
                }
            }
            optdim = nn->laydim[nn->numlay - 1];
/*
            for(k = 0; k < optdim; k++)
            {
                nn->temp[t & 1][k] = nn->temp[t & 1][k] * nn->compensate;
                printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k]);
            }
*/
            for(k = 0; k < optdim; k++)
            {
                
                if(nn->temp[t & 1][k] > preVal)
                {
                    preVal = nn->temp[t & 1][k];
                    preIdx = k;
                    //printf("%lf\n", preVal);
                }
               
                //printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k] * nn->compensate);
            }
            
            fscanf(flab, "%d", &ans);
            nCorrect = ans == preIdx ? nCorrect + 1 : nCorrect;
        }
    }
    fclose(fp);
    fclose(flab);
    //printf("Accuracy: %.3f\n", (float)nCorrect / nTestcase);
    printf("%.3f ", (float)nCorrect / nTestcase);
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
    if(nn->factor != NULL)
    {
        free(nn->factor);
    }
    free(nn->temp[0]);
    free(nn->temp[1]);
}


#if FIXPOINT
double fix(double x)
{
    int i, j, sign = 1;
    double ret, k, ori = x;
    if(x >= 0) sign = 1;
    else sign = -1;

    x = sign == 1 ? x : x * -1;
    if(MAX_DECIMAL_BIT == 0 && x >= 1)
    {
        double tmp = (double)powf(2, - MAX_FLOAT_BIT - 1);
        x -= tmp;
        //printf("%lf\n", tmp);
    }
    ret = 0;
    ret += (int) x > ((1 << MAX_DECIMAL_BIT) - 1) ? ((1 << MAX_DECIMAL_BIT) - 1) : (int) x;
    for(i = 0, j = -1, k = 2 * (x - (int) x); i < MAX_FLOAT_BIT - 1; i++, j--, k += k)
    {
      ret += k - 1 >= 0 ? powf(2, j) : 0;
      k -= (int)k;
    }
    ret += k - 1 > 0 ? powf(2, j) : (k - powf(2, j + 1) > 0 ? powf(2, j) : 0);
    ret = ret * sign;
    //printf("%f after fixed point conversion: %f\n", ori, ret);
    return ret;
}
#endif