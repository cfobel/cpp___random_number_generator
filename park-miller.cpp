/* WBL Crest 21 March 2009 $Revision: 1.14 $
 * Based on cuda/sdk/projects/quasirandomGenerator/quasirandomGenerator.cpp
 * and pch.h W. Langdon cs.ucl.ac.uk 5 May 1994 (21 Feb 95 make it inline)
 */
/*********************************************************************

INTRODUCTION

This file contains (at your own risk) an implementation of the
Park-Miller minimum random number generator (Comm ACM Oct 1988
p1192-1201, Vol 31 Num 10).  It is suitable for use with GPQUICK-2.1

USAGE

All psuedo random number generators need a seed (ie initial starting
value from which to generate the sequence). In some cases this is
based on reading the current system time. This code allows you to
specify what it is. Any positive 32 bit number (NOT zero!) will do the
first time you call it. Use the 32 bit value it returns as the seed
for the next call, and so on for each sucessive call.

A positive psuedo random integer uniformly distributed between 1 and
2147483647 (ie 2**31-1) is returned.

Example

Taking the modulus 52 of the returned value will give you a value
between 0 and 51. This has a slight bias as 2**31-1 is not exactly
divisible by 52 but it may be good enough for your purposes.

NB use the whole 32 bit random value as the seed for the next call. Do
not use its modulus, as this will lead to a very short random number
sequence not the full 2147483647 which park-miller provides.

*********************************************************************/



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <vector>
using namespace std;


#ifdef WIN32
#define strcasecmp strcmpi
#endif



#include "ParkMillerRNG.hpp"


int main(int argc, char **argv){
    int u, seed;
	int cycles = 1000;
	unsigned int grid_size = 128;
	unsigned int block_size =  128*3 ;//QRNG_DIMENSIONS;
	unsigned int N=1048576*3; //QRNG_DIMENSIONS;
    seed = 1;

	if(argc<=5) printf("\nProgram to test Park-Miller Random number generator\n");
	if(argc>1) cycles    = atoi(argv[1]);
	if(argc>2) grid_size = atoi(argv[2]);
	if(argc>3) block_size= atoi(argv[3]);
	if(argc>4) N         = atoi(argv[4]);

	if(argc<=5) printf("seed=0 cycles=%d grid_size=%d block_size=%d data_size=%d\n",cycles,grid_size,block_size,N);

	if(argc<=5) printf("Genenerating 10,000 random numbers on CPU with an initial seed of 1\n");
    ParkMillerRNG rng(seed);
	for (int i=1; i<=10000; i++) u = seed = rng.get_value();

	if(argc<=5 || u != 1043618065 ) printf("10,000th random number is %d it should be 1043618065\n",u);

	if ( u != 1043618065 ) return 2;

    int *d_Output;

    int pos;

    double L1norm, gpuTime;
    int delta, sumDelta, sumRef, ref;
    int nerror=0;

    unsigned int hTimer;

    if(sizeof(INT64) != 8){
        printf("sizeof(INT64) != 8\n");
        return 0;
    }

    /*
    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
        cutilDeviceInit(argc, argv);
    else
    */
    cudaSetDevice( cutGetMaxGflopsDeviceId() );

    cutilCheckError(cutCreateTimer(&hTimer));

    int deviceIndex;
    cutilSafeCall(cudaGetDevice(&deviceIndex));
    cudaDeviceProp deviceProp;
    cutilSafeCall(cudaGetDeviceProperties(&deviceProp, deviceIndex));
    int version = deviceProp.major * 10 + deviceProp.minor;
    if(version < 13){
        printf("Double precision not supported.\n");
        cudaThreadExit();
        return 0;
    }

    CUDAParkMillerRNG cuda_rng(0);
	vector<int> h_OutputGPU = cuda_rng.get_values(cycles, grid_size, block_size, N);

    if(argc<=5) printf("Comparing to the CPU results...\n");
    sumDelta = 0;
    sumRef = 0;
    for(pos = 0; pos < N; pos++){
        const unsigned int p3 = pos % (N/3);
        ref = rng.get_value(pos);
        if(p3 <= 5 || p3 > (N / 3) - 5) {
            for (int i = 2; i <= cycles; i++) {
                ref = rng.get_value();
            }
            delta = h_OutputGPU[pos] - ref;
            if(delta != 0) {
                nerror++;
                if(nerror <= 10) printf("%6d %7d %2d %10d it should be %10d (%d)\n",pos/(N/3),N/3,p3, h_OutputGPU[pos],ref,delta);
            } else {
                if(argc <= 5) printf("%6d %7d %2d %10d\n",pos/(N/3),N/3,p3, h_OutputGPU[pos]); 
            }
            sumDelta += abs(delta);
            sumRef   += abs(ref);
        }
    }
    if(argc<=5 || nerror>0) printf("Error %d %d L1 norm: %E\n", nerror, sumDelta, L1norm = (double)sumDelta / (double) sumRef);
    if(argc<=5 || nerror>0) printf((nerror==0) ? "TEST PASSED\n" : "TEST FAILED\n");

    cudaThreadExit();

    return 0;
}
