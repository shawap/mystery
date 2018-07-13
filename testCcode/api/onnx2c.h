#ifndef ONNXTOC_H
#define ONNXTOC_H



#define MAX_LAYER   16
#define NUM_LAYER   5


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

/**
 * input & outputshould be 1 dimension
 * 
 * 
 * */
int RunModel(double *input, double *output);


int init_nn(_nn *nn);

int GetNNInputSize(_nn* nn);

int GetNNOutputSize(_nn* nn);

int test_nn(_nn *nn, double *input, double *output);

void free_nn(_nn *nn);





#endif