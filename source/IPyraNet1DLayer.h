/* 
 *
 */

#ifndef _IPyraNet1DLayer_h_
#define _IPyraNet1DLayer_h_

#include "IPyraNetLayer.h"
#include <vector>

template<class OutType>
class IPyraNet1DLayer : public IPyraNetLayer<OutType> {
public:
    
    IPyraNet1DLayer();
    IPyraNet1DLayer(int receptive, int inhibitory, int overlap, IPyraNetActivationFunction<OutType>* activationFunc = NULL);
    virtual ~IPyraNet1DLayer();

    OutType getNeuronOutput(int dimensions, int* neuronLocation);    
    int getDimensions() const;
    void getSize(int* size);

    void setParentLayer(IPyraNetLayer<OutType>* parent);

private:
    int neurons;    // number of neurons (size of the layer in 1D)
};

#endif // _IPyraNet1DLayer_h_