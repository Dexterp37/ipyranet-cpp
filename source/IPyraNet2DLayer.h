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
    IPyraNet2DLayer(int width, int height);
    virtual ~IPyraNet2DLayer();

    void setLayerSize(int width, int height);

    void setReceptiveFieldSize(int r) { receptiveSize = r; }
    int getReceptiveFieldSize() const { return receptiveSize; }

    void setOverlap(int o) { overlap = o; }
    int getOverlap() const { return overlap; }

    void setInhibitoryFieldSize(int i) { inhibitorySize = i; }
    int getInhibitoryFieldSize() const { return inhibitorySize; }

    OutType getNeuronOutput(int dimensions, int* neuronLocation);    
    int getDimensions() const;
    void getSize(int* size);

private:
    int width;
    int height;

    int receptiveSize;
    int overlap;
    int inhibitorySize;

    // a weight for each neuron
    std::vector<std::vector<OutType> > weights;
    std::vector<std::vector<OutType> > biases;

    void initWeights();
    void initBiases();
};

#endif // _IPyraNet2DLayer_h_