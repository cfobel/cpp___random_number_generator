#ifndef ___PARK_MILLER_RNG__HPP___
#define ___PARK_MILLER_RNG__HPP___


#include <vector>
#include <iostream>
#include "LehmerRNG.hpp"
using namespace std;


class ParkMillerRNG : public LehmerRNG {
protected:
    double get_modulus() {
        //ie 2**31-1
        return 2147483647;
    }
    double get_multiplier() {
        //ie 7**5
        return 16807;
    }
public:
    ParkMillerRNG(unsigned int seed=1) : LehmerRNG(seed) {}
};

#endif
