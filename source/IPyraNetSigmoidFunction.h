/**
 * ipyranet-cpp
 * 
 * Copyright (C) 2012 Alessio Placitelli
 *
 * Permission is hereby granted, free of charge, to any person 
 * obtaining a copy of this software and associated documentation 
 * files (the "Software"), to deal in the Software without 
 * restriction, including without limitation the rights to use, 
 * copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 * OTHER DEALINGS IN THE SOFTWARE.
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