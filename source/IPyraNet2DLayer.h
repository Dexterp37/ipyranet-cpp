/* 
 *
 */

#ifndef _IPyraNet2DLayer_h_
#define _IPyraNet2DLayer_h_

#include "IPyraNetLayer.h"
#include <vector>

template<class OutType>
class IPyraNet2DLayer : public IPyraNetLayer<OutType> {
public:
    
    IPyraNet2DLayer();
    IPyraNet2DLayer(int receptive, int inhibitory, int overlap, IPyraNetActivationFunction<OutType>* activationFunc = NULL);
    virtual ~IPyraNet2DLayer();

    LayerType getLayerType() const { return Layer2D; }

    void setReceptiveFieldSize(int r) { receptiveSize = r; }
    int getReceptiveFieldSize() const { return receptiveSize; }

    void setOverlap(int o) { overlap = o; }
    int getOverlap() const { return overlap; }

    void setInhibitoryFieldSize(int i) { inhibitorySize = i; }
    int getInhibitoryFieldSize() const { return inhibitorySize; }
    
    OutType getErrorSensitivity(int dimensions, int* neuronLocation, OutType multiplier);
    OutType getNeuronOutput(int dimensions, int* neuronLocation);    
    int getDimensions() const;
    void getSize(int* size);
    OutType getNeuronWeight(int dimensions, int* neuronLocation);

    void setParentLayer(IPyraNetLayer<OutType>* parent, bool init);

    void saveToXML(pugi::xml_node& node);
    void loadFromXML(pugi::xml_node& node);

private:
    int width;
    int height;

    int receptiveSize;
    int overlap;
    int inhibitorySize;

    // a weight for each neuron
    std::vector<std::vector<OutType> > weights;
    std::vector<std::vector<OutType> > biases;

    OutType getWeightedSumInput(int dimensions, int* neuronLocation);
    void initWeights();
    void initBiases();
};

#endif // _IPyraNet2DLayer_h_