#ifndef ___RANDOM_NUMBER_GENERATOR__HPP___
#define ___RANDOM_NUMBER_GENERATOR__HPP___

class RandomNumberGenerator {
protected:
    unsigned int seed;
    virtual int get_max_value() = 0;
public:
    RandomNumberGenerator(unsigned int seed=0) : seed(seed) {}
    void set_seed(unsigned int in_seed) {
        seed = in_seed;
    }
    unsigned int get_seed() const {
        return seed;
    }
    int get_value(unsigned int in_seed) {
        set_seed(in_seed);
        return get_value();
    }
    int rand_int(int max_value) {
        return get_value() % (max_value + 1);
    }
    double rand_double() {
        return get_value() / (double)get_max_value();
    }
    virtual int get_value() = 0;
};

#endif
