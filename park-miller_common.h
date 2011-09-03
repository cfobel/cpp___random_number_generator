/* WBL Crest 21 March 2009 $Revision: 1.5 $
 * Based on cuda/sdk/projects/quasirandomGenerator/quasirandom_common.h
 */

#include <vector>
#include <iostream>
#include <cutil_inline.h>
using namespace std;


#ifndef PARKMILLER_COMMON_H
#define PARKMILLER_COMMON_H



////////////////////////////////////////////////////////////////////////////////
// Global types and constants
////////////////////////////////////////////////////////////////////////////////
typedef long long int INT64;

//#define QRNG_DIMENSIONS 3 now is default value.....
//#define QRNG_RESOLUTION 31
//#define INT_SCALE (1.0f / (float)0x80000001U)
//#define  PMc 1000
//#define  PMc 1


class ParkMillerRNG {
protected:
    unsigned int seed;
public:
    ParkMillerRNG(unsigned int seed=0) : seed(seed) {}
    void set_seed(unsigned int in_seed) {
        seed = in_seed;
    }
    int get_value(unsigned int in_seed) {
        set_seed(in_seed);
        return get_value();
    }
    int get_value() {
        //Generate single Park-Miller psuedo random number
        double const a = 16807;      //ie 7**5
        double const m = 2147483647; //ie 2**31-1

        double temp = seed * a;
        seed = (int) (temp - m * floor ( temp / m ));
        return seed;
    }
};


extern "C"
void parkmillerGPU(int *d_Output, unsigned int seed, int cycles,
       unsigned int grid_size, unsigned int block_size, unsigned int N);


class CUDAParkMillerRNG : public ParkMillerRNG {
protected:
    int *d_Output;
    int data_size;
public:
    CUDAParkMillerRNG(unsigned int seed=0) : ParkMillerRNG(seed), data_size(0), d_Output(NULL) {}

    vector<int> get_values(int cycles, unsigned int grid_size,
                            unsigned int block_size, unsigned int N){
        vector<int> output(N);
        int output_size = output.size() * sizeof(int);
        if(data_size < output_size) {
            if(NULL != d_Output) {
                cudaFree(d_Output);
            }
            cout << "Allocating " << output_size << " bytes..." << endl;
            cutilSafeCall(cudaMalloc((void **)&d_Output, output_size));
            cout << "  DONE" << endl;
            data_size = output_size;
        }
        cutilSafeCall(cudaMemset(d_Output, 0, output_size));
        cutilSafeCall(cudaThreadSynchronize());
        parkmillerGPU(d_Output, seed, cycles, grid_size, block_size, N);
        cutilSafeCall(cudaThreadSynchronize());
        cutilSafeCall(cudaMemcpy(&output[0], d_Output, output_size, cudaMemcpyDeviceToHost));

        return output;
    }

    ~CUDAParkMillerRNG() {
        if(NULL != d_Output) {
            cudaFree(d_Output);
        }
    }
};

#endif
