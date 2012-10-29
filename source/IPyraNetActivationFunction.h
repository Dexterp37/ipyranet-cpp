/*

*/

#ifndef _IPyraNetActivationFunction_h_
#define _IPyraNetActivationFunction_h_

template<typename OutType>
class IPyraNetActivationFunction {
public:
    IPyraNetActivationFunction() { };
    virtual ~IPyraNetActivationFunction() { };

    virtual OutType compute(OutType val) = 0;
};

#endif // _IPyraNetActivationFunction_h_