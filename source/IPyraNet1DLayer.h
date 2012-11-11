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
    IPyraNet1DLayer(int numberNeurons, IPyraNetActivationFunction<OutType>* activationFunc = NULL);
    virtual ~IPyraNet1DLayer();
    
    LayerType getLayerType() const { return Layer1D; }
    OutType getErrorSensitivity(int dimensions, int* neuronLocation, OutType multiplier);
    OutType getNeuronOutput(int dimensions, int* neuronLocation);    
    int getDimensions() const;
    void getSize(int* size);
    OutType getNeuronWeight(int dimensions, int* neuronLocation);

    void setParentLayer(IPyraNetLayer<OutType>* parent, bool init);

    void saveToXML(pugi::xml_node& node);
    void loadFromXML(pugi::xml_node& node);

private:
    unsigned int neurons;    // number of neurons (size of the layer in 1D)

    // in 1D layers weights are per-connection, not per neuron.
    // so we have a inputs * neurons weights
    std::vector<std::vector<OutType> > weights;

    // one bias per neuron/output
    std::vector<OutType> biases;

    OutType getWeightedSumInput(int dimensions, int* neuronLocation);
    void initWeights();
    void initBiases();
};

#endif // _IPyraNet1DLayer_h_