#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define NUMBER_OF_LAYER {} 
#define NUMBER_OF_GAP  {} 



typedef struct _nn
lbrace
    int numlay;
    int maxdim;
    int laydim[NUMBER_OF_LAYER];
    double *wei[NUMBER_OF_GAP];
    double *bia[NUMBER_OF_GAP];
    double *temp[2];
    // Future work, function pointers to activation functions
rbrace   _nn;

void init_nn(_nn *nn);
void test_nn(_nn *nn);
void free_nn(_nn *nn);
__attribute__((always_inline)) inline double RELU(const double x) lbrace return x > 0 ? x : 0; rbrace

int main()
lbrace
    _nn nn;
    init_nn(&nn);
    test_nn(&nn);
    free_nn(&nn);
    return 0;
rbrace


void init_nn(_nn *nn)
lbrace
    // dimension infos
    nn->numlay = NUMBER_OF_LAYER;
    nn->maxdim = {};
    memcpy(nn->laydim, {}, sizeof(int) * NUMBER_OF_LAYER);
    
 
    // extra space for storing results
    nn->temp[0] = (double *) malloc(sizeof(double) * nn->maxdim); assert(nn->temp[0] != NULL);
    nn->temp[1] = (double *) malloc(sizeof(double) * nn->maxdim); assert(nn->temp[1] != NULL);


    // construct nns
    int i, j, ofst = 0;
    double coef[] = {};
    for(i = 0; i < NUMBER_OF_GAP; i++)
    lbrace
        int wdim = nn->laydim[i] * nn->laydim[i + 1];
        int bdim = nn->laydim[i + 1];
        nn->wei[i] = (double *) malloc(sizeof(double) * wdim); assert(nn->wei[i] != NULL);
        nn->bia[i] = (double *) malloc(sizeof(double) * bdim); assert(nn->bia[i] != NULL);
        memcpy(nn->wei[i], &coef[ofst], sizeof(double) * wdim);
        ofst += wdim;
        memcpy(nn->bia[i], &coef[ofst], sizeof(double) * bdim);
        ofst += bdim;
    rbrace
rbrace

void test_nn(_nn *nn)
lbrace
    FILE *fp = (FILE *) fopen("td1.txt", "r");
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
    lbrace
        fscanf(fp, "%lf", &temp);
        nn->temp[t & 1][i] = temp;
        if(i != 0 && i % iptdim == 0) // every single test;
        lbrace
            i = 0;
            for(m = 0; m < nn->numlay - 1; m++)
            lbrace
                t++;
                memset(nn->temp[t & 1], 0, sizeof(double) * nn->maxdim);
                // set bias here
                memcpy(nn->temp[t & 1], nn->bia[m], sizeof(double) * nn->laydim[m + 1]); // should be m+1
                // weighting
                for(n = 0; n < nn->laydim[m]; n++)
                lbrace
                    for(k = 0; k < nn->laydim[m + 1]; k++)
                    lbrace
                        nn->temp[t & 1][k] += nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];
                    rbrace
                rbrace
                // activation functions here;
                for(k = 0; k < nn->laydim[m + 1]; k++)
                lbrace
                    nn->temp[t & 1][k] = RELU(nn->temp[t & 1][k]);
                rbrace
            rbrace
            optdim = nn->laydim[nn->numlay - 1];
            for(k = 0; k < optdim; k++)
            lbrace
                printf("[%d]:\t%lf\n", k, nn->temp[t & 1][k]);
            rbrace
        rbrace
    rbrace
    fclose(fp);
rbrace



void free_nn(_nn *nn)
lbrace
    int i;
    int numgap = nn->numlay - 1;
    for(i = 0; i < numgap; i++)
    lbrace
        free(nn->wei[i]);
        free(nn->bia[i]);
    rbrace
    free(nn->temp[0]);
    free(nn->temp[1]);
rbrace