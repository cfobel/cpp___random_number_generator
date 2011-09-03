/*WBL 21 March 2009 $Revision: 1.3 $
 * based on cuda/sdk/projects/quasirandomGenerator/quasirandomGenerator_SM13.cu
 */



#define DOUBLE_PRECISION
#include "park-miller_common.h"

//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Park-Miller quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////
static __global__ void parkmillerKernel(int *d_Output, unsigned int seed,
        int cycles, unsigned int N){
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);
    double const a    = 16807;      //ie 7**5
    double const m    = 2147483647; //ie 2**31-1
    double const reciprocal_m = 1.0/m;

    // W. Langdon cs.ucl.ac.uk 5 May 1994
    for(unsigned int pos = tid; pos < N; pos += threadN){
        unsigned int result = 0;
        unsigned int data = seed + pos;

        for(int i = 0; i < cycles; i++) {
            double temp = data * a;
            result = (int) (temp - m * floor ( temp * reciprocal_m ));
            data = result;
        } //endfor

        //d_Output[MUL(threadIdx.y, N) + pos] = (float)(result + 1) * INT_SCALE;
        //d_Output[MUL(threadIdx.y, N) + pos] = result;
        d_Output[pos] = result;
    }
}


//Host-side interface
extern "C"
void parkmillerGPU(int *d_Output, unsigned int seed, int cycles,
       unsigned int grid_size, unsigned int block_size, unsigned int N){
    parkmillerKernel<<<grid_size, block_size>>>(d_Output, seed, cycles, N);
    cutilCheckMsg("parkmillerKernel() execution failed.\n");
}
