#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#define MAX_DECIMAL_BIT   10
#define MAX_FLOAT_BIT     10

#define MAX_LAYER 16


#define FIX(x) fix(x)
#define MUL(x, y) (FIX(FIX(x) * FIX(y)))
#define ADD(x, y) (FIX(FIX(x) + FIX(y)))



// 測試檔路徑
const char testFile[] = "../testCcode/test_images.txt";
const char ansFile[] = "../testCcode/test_label_list.txt";

typedef struct _nn
{
    int numlay;
    int maxdim;
    int laydim[MAX_LAYER];
    double *wei[MAX_LAYER];
    double *bia[MAX_LAYER];
    double *temp[2]; // for calculating
    /*
     * future add,
     * func ptr to act func
    */
}   _nn;

void init_nn(_nn *nn);
void test_nn(_nn *nn);
void free_nn(_nn *nn);
double fix(double x);
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

void test_nn(_nn *nn)
{
    FILE *fp = (FILE *) fopen(testFile, "r");
    FILE *fpans = (FILE *) fopen(ansFile, "r");
    assert(fp != NULL);
    assert(fpans != NULL);
    int i;
    int t;
    int m;
    int n;
    int k;
    int iptdim;
    int optdim;
    int test_cnt = 0, corr_cnt = 0;
    double temp;
    iptdim = nn->laydim[0];

    for(i = 0, t = 0; !feof(fp); i++)
    {
        fscanf(fp, "%lf", &temp);
        nn->temp[t & 1][i] = temp;
        if(i != 0 && i % iptdim == 0) // every single test;
        {
            i = 0;
            int max_idx = 0, ans;
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
                        //nn->temp[t & 1][k] += nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];
                        nn->temp[t & 1][k] = ADD(nn->temp[t & 1][k], MUL(nn->temp[!(t & 1)][n] , nn->wei[m][n * nn->laydim[m+1] + k]));
                    }
                }
                // activation functions here;
                for(k = 0; k < nn->laydim[m + 1]; k++)
                {
                    nn->temp[t & 1][k] = RELU(nn->temp[t & 1][k]);
                    nn->temp[t & 1][k] = FIX(nn->temp[t & 1][k]);
                }
            }
            optdim = nn->laydim[nn->numlay - 1];
            for(k = 0; k < optdim; k++)
            {
                //printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k]);
                if(nn->temp[t & 1][k] > nn->temp[t & 1][max_idx])
                    max_idx = k;
            }
            fscanf(fpans, "%d", &ans);
            printf("correct is %d, predicet is %d\n", ans, max_idx);
            if(ans == max_idx)
            {
                corr_cnt++;
            }
            test_cnt++;
        }//end of if(i != 0 && i % iptdim == 0) 
    }
    printf("The accuracy is %lf,  %d correct predict in %d test data!\n", corr_cnt / (double)test_cnt, corr_cnt, test_cnt);
    fclose(fpans);
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
}


double fix(double x)
{
    int _i, _j;
    double ret, _k;
    ret = 0;
    ret += (int) x > ((1 << MAX_DECIMAL_BIT) - 1) ? ((1 << MAX_DECIMAL_BIT) - 1) : (int) x;
    for(_i = 0, _j = -1, _k = 2 * (x - (int) x); _i < MAX_FLOAT_BIT - 1; _i++, _j--, _k+=_k)
    {
      ret += _k - 1 > 0 ? powf(2, _j) : 0;
      _k -= (int)_k;
    }
    ret += _k - 1 > 0 ? powf(2, _j) : (_k - powf(2, _j + 1) > 0 ? powf(2, _j) : 0);
    //printf("%lf after fixed point conversion: %lf\n", x, ret);
    return ret;
}