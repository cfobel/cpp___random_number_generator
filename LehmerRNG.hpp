#ifndef ___LEHMER_RNG__HPP___
#define ___LEHMER_RNG__HPP___

#include "RandomNumberGenerator.hpp"

class LehmerRNG : public RandomNumberGenerator {
/* Lehmer random number generator
*    X_{k+1} = (g * X_k) % n 
*
* where the modulus n is a prime number or a power of a prime number, the
* multiplier g is an element of high multiplicative order modulo n (e.g.,
* a primitive root modulo n), and the seed X0 is co-prime to n. */
protected:
    int get_max_value() {
        return get_modulus() - 1;
    }
    virtual double get_multiplier() = 0;
    virtual double get_modulus() = 0;
public:
    LehmerRNG(unsigned int seed=1) : RandomNumberGenerator(seed) {}
    using RandomNumberGenerator::get_value;
    virtual int get_value() {
        double const a = get_multiplier();
        double const m = get_modulus();

        double temp = seed * a;
        seed = (int) (temp - m * floor ( temp / m ));
        return seed;
    }
};

#endif
