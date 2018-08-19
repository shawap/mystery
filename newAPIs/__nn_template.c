#include "onnx2c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define NUMBER_OF_LAYER {}
#define NUMBER_OF_GAP  {}

/* Configuration */
#define LOOP_EXCHANGE    0
#define FIXPOINT         0
#define FACTORS_FILE     "./factors"


#if FIXPOINT
const int MAX_DECIMAL_BIT =    0;
int MAX_FLOAT_BIT   =   31;         
double fix(double x);  
#endif


//hard code const weigths and bias
{}

__attribute__((always_inline)) inline double RELU(const double x) lbrace return x > 0 ? x : 0; rbrace

int RunModel(double *input, double *output)
lbrace
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
rbrace



int Init_NN(_NN_ *nn)
lbrace
    // dimension infos
    nn->numlay = NUMBER_OF_LAYER;
    nn->maxdim = {};
    nn->factor = NULL;
    int err = 0;
    memcpy(nn->laydim, {}, sizeof(int) * NUMBER_OF_LAYER);


    // extra space for storing results
    nn->temp[0] = (double *) malloc(sizeof(double) * nn->maxdim); assert(nn->temp[0] != NULL);
    nn->temp[1] = (double *) malloc(sizeof(double) * nn->maxdim); assert(nn->temp[1] != NULL);


    // construct nns
    /*
    int i, j, ofst = 0;
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
    */

    //use python hard code the wieghts and bias
    {}

    return err;
rbrace


int GetNNInputSize(_NN_* nn)
lbrace
    assert(nn != NULL);
    return nn->laydim[0];
rbrace

int GetNNOutputSize(_NN_* nn)
lbrace
    assert(nn != NULL);
    return nn->laydim[nn->numlay - 1];
rbrace



int Norm_NN(_NN_ *nn)
lbrace
    int numOfLay = nn->numlay; // Note that it includes the input layer now
    int numOfGap = numOfLay - 1;
    int err = 0;

    // memory allocation for normalization factors
    nn->factor = (_norm *) malloc(sizeof(_norm) * numOfGap);
    assert(nn->factor != NULL);

    int idxi, idxj, idxm, idxn, idxk;
    int bdim, wdim, numOfNeuron;

    // get the nw, na, na2
    FILE * fptr = (FILE *) fopen(FACTORS_FILE, "r");
    assert(fptr != NULL);

    for(idxi = 0 ; idxi < numOfGap ; idxi++)
    lbrace
        fscanf(fptr, "%lf", &nn->factor[idxi].nw);
        fscanf(fptr, "%lf", &nn->factor[idxi].na);
        fscanf(fptr, "%lf", &nn->factor[idxi].na2);
        //printf("[%d] nw is %lf, na is %lf, na2 is %lf\n", idxi, nn->factor[idxi].nw, nn->factor[idxi].na, nn->factor[idxi].na2);
    rbrace
    fscanf(fptr, "%lf", &nn->compensate);

    //printf("%lf\n", nn->compensate);

    // normalize the weight and bias
    for(idxi = 0; idxi < numOfGap; idxi++)
    lbrace
        // get dimension infos at first
        bdim = nn->laydim[idxi + 1];
        wdim = nn->laydim[idxi] * nn->laydim[idxi + 1];

        // (1) normalize bias at the first palce (by previous layers' nas & na2s)
        // (1-1) from i=0 ~ current - 2, bias /= (na * nw)
        // (1-2) from i=current - 1 if i >= 0, bias /= (na2 * nw)
        for(idxj = 0; idxj < idxi - 1; idxj++)
        lbrace
            for(idxk = 0; idxk < bdim; idxk++)
            lbrace
                nn->bia[idxi][idxk] /= (nn->factor[idxj].na * nn->factor[idxj].nw);
            rbrace
        rbrace
        for(idxj = idxi - 1; idxj >= 0; idxj = -1)
        lbrace
            for(idxk = 0; idxk < bdim; idxk++)
            lbrace
                nn->bia[idxi][idxk] /= (nn->factor[idxj].na2 * nn->factor[idxj].nw);
            rbrace
        rbrace
        

        // change the original data directly for testing purposes
        // tuning weights
        for(idxk = 0; idxk < wdim; idxk++)
        lbrace
            nn->wei[idxi][idxk] /= nn->factor[idxi].nw;
        rbrace
        // tuning biases
        for(idxk = 0; idxk < bdim; idxk++)
        lbrace
            nn->bia[idxi][idxk] /= nn->factor[idxi].nw;
        rbrace
        
    rbrace


    // fix point
#if FIXPOINT
    for(idxi = 0 ; idxi < numOfGap ; idxi++)
    lbrace
        // get dimension infos at first
        bdim = nn->laydim[idxi + 1];
        wdim = nn->laydim[idxi] * nn->laydim[idxi + 1];

        for(idxj = 0 ; idxj < wdim ; idxj++)
            nn->wei[idxi][idxj] = fix(nn->wei[idxi][idxj]);
        for(idxj = 0 ; idxj < bdim ; idxj++)
            nn->bia[idxi][idxj] = fix(nn->bia[idxi][idxj]);
    rbrace
#endif

    fclose(fptr);


    return err;
rbrace




int Test_NN(_NN_ *nn, double *input, double *output)
lbrace
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
    lbrace
        //fscanf(fp, "%lf", &temp);
        nn->temp[t & 1][i] = input[i];
        if((i + 1) % iptdim == 0)  // every single test;
        lbrace
            // i = -1;
            for(m = 0; m < nn->numlay - 1; m++)
            lbrace
                t++;
                memset(nn->temp[t & 1], 0, sizeof(double) * nn->maxdim);
                // set bias here
                memcpy(nn->temp[t & 1], nn->bia[m], sizeof(double) * nn->laydim[m + 1]); // should be m+1

                /*** sheep move  ****/
                if(nn->factor != NULL)
                lbrace
                    if(m == 0)
                    lbrace
                        for(k = 0; k < nn->laydim[m + 1]; k++)
                        lbrace
                            nn->temp[t & 1][k] /= nn->factor[m].na2;
                            //nn->temp[t & 1][k] = fix(nn->temp[t & 1][k]);
                        rbrace
                    rbrace
                    else if(m > 0)
                    lbrace
                        for(k = 0; k < nn->laydim[m + 1]; k++)
                        lbrace
                            nn->temp[t & 1][k] = nn->temp[t & 1][k] * nn->factor[m - 1].na2 / nn->factor[m - 1].na / nn->factor[m].na2;
                            //nn->temp[t & 1][k] = fix(fix(fix(nn->temp[t & 1][k]) * fix(nn->factor[m - 1].na2)) / fix(fix(nn->factor[m - 1].na) / fix(nn->factor[m].na2)));
                            //nn->temp[t & 1][k] = fix(nn->temp[t & 1][k]);
                        rbrace
                    rbrace
                rbrace
                /********************/

                // weighting
#if LOOP_EXCHANGE
                for(k = 0; k < nn->laydim[m + 1]; k++)
                lbrace
                    for(n = 0; n < nn->laydim[m]; n++)
                    lbrace
#else
                for(n = 0; n < nn->laydim[m]; n++)
                lbrace
                    for(k = 0; k < nn->laydim[m + 1]; k++)
                    lbrace
#endif
                        /*** sheep add  ****/
                        if(nn->factor != NULL)
                        lbrace
                            double tmp = nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];
#if FIXPOINT
                            tmp = fix(tmp);
#endif                
                            assert(tmp < 1);
                            
                            if(m == 0)
                            lbrace
                                tmp /= nn->factor[m].na2;
                                nn->temp[t & 1][k] += tmp;
                            rbrace
                            else if(m > 0)
                            lbrace
                                tmp = tmp * nn->factor[m - 1].na2 / nn->factor[m - 1].na / nn->factor[m].na2;
                                nn->temp[t & 1][k] += tmp;
                            rbrace
                            assert(nn->temp[t & 1][k] < 1 );
                        rbrace
                        /********************/

                        /** original **/
                        else    nn->temp[t & 1][k] += nn->temp[!(t & 1)][n] * nn->wei[m][n * nn->laydim[m+1] + k];

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
                if(nn->factor != NULL)
                lbrace
                    nn->temp[t & 1][k] = nn->temp[t & 1][k] * nn->compensate;
                rbrace
                printf("[%d]:\t%lf\r\n", k, nn->temp[t & 1][k]);
                output[k] = nn->temp[t & 1][k];
            rbrace
        rbrace
    rbrace
    //fclose(fp);

    return err;
rbrace



void Free_NN(_NN_ *nn)
lbrace
    int i;
    int numgap = nn->numlay - 1;
    /*
    for(i = 0; i < numgap; i++)
    lbrace
        free(nn->wei[i]);
        free(nn->bia[i]);
    rbrace
    */
    free(nn->temp[0]);
    free(nn->temp[1]);

rbrace



#if FIXPOINT
double fix(double x)
lbrace
    int i, j, sign = 1;
    double ret, k, ori = x;
    if(x >= 0) sign = 1;
    else sign = -1;

    x = sign == 1 ? x : x * -1;
    if(MAX_DECIMAL_BIT == 0 && x >= 1)
    lbrace
        double tmp = (double)powf(2, - MAX_FLOAT_BIT - 1);
        x -= tmp;
        //printf("%lf\n", tmp);
    rbrace
    ret = 0;
    ret += (int) x > ((1 << MAX_DECIMAL_BIT) - 1) ? ((1 << MAX_DECIMAL_BIT) - 1) : (int) x;
    for(i = 0, j = -1, k = 2 * (x - (int) x); i < MAX_FLOAT_BIT - 1; i++, j--, k += k)
    lbrace
      ret += k - 1 >= 0 ? powf(2, j) : 0;
      k -= (int)k;
    rbrace
    ret += k - 1 > 0 ? powf(2, j) : (k - powf(2, j + 1) > 0 ? powf(2, j) : 0);
    ret = ret * sign;
    //printf("%f after fixed point conversion: %f\n", ori, ret);
    return ret;
rbrace
#endif