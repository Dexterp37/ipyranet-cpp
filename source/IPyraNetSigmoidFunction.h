/*

*/

#ifndef _IPyraNetSigmoidFunction_h_
#define _IPyraNetSigmoidFunction_h_

#include "IPyraNetActivationFunction.h"
#include <math.h>

template<typename OutType>
class IPyraNetSigmoidFunction : public IPyraNetActivationFunction<OutType> {
public:
    IPyraNetSigmoidFunction() { };
    virtual ~IPyraNetSigmoidFunction() { };

    OutType compute(OutType val) {
        OutType exponential = exp((OutType)-val);
        
        return static_cast<OutType>(1 / (1 + exponential));
    };

    OutType derivative(OutType val) {
        // Sigmoid Function derivative considerations taken from here
        // http://www.learnartificialneuralnetworks.com/backpropagation.html
        // http://www.wkiri.com/cs461-w08/Lectures/Lec4/ANN-deriv.pdf
    
        OutType sigm = compute(val);
        
        return static_cast<OutType>(sigm * (1 - sigm));
    };
};

#endif // _IPyraNetSigmoidFunction_h_