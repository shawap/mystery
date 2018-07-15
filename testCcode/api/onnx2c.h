#ifndef ONNXTOC_H
#define ONNXTOC_H



#define MAX_LAYER   16


typedef struct _NN_
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
}   _NN_;

/**
 * input & outputshould be 1 dimension
 * 
 * 
 * */
int RunModel(double *input, double *output);


int Init_NN(_NN_ *nn);

int GetNNInputSize(_NN_* nn);

int GetNNOutputSize(_NN_* nn);

int Test_NN(_NN_ *nn, double *input, double *output);

void Free_NN(_NN_ *nn);





#endif