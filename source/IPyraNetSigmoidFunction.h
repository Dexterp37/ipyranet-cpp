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

    virtual OutType compute(OutType val) {
        OutType exponential = exp((OutType)-val);
        
        return static_cast<OutType>(1 / (1 + exponential));
    };
};

#endif // _IPyraNetSigmoidFunction_h_