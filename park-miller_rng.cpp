#include <boost/format.hpp>
#include <iostream>
#include "ParkMillerRNG.hpp"
using namespace std;

#define _ boost::format

int main(int argc, char **argv) {
    ParkMillerRNG rng(1);

    cout << "data = [" << rng.get_value();
    for(int i = 1; i < 10000; i++) {
        cout << ", " << endl << rng.get_value();
    }
    cout << "]" << endl;

    return 0;
}
