/* 
 *
 */

#ifndef _IPyraNet2DLayer_h_
#define _IPyraNet2DLayer_h_

#include "IPyraNetLayer.h"
#include <vector>

class IPyraNet2DLayer {
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

    void setParentLayer(IPyraNet2DLayer* parent) { parentLayer = parent; }
    IPyraNet2DLayer* getParentLayer() { return parentLayer; }

private:
    int width;
    int height;

    int receptiveSize;
    int overlap;
    int inhibitorySize;

    // previous (adjacent bigger) layer in the pyramid
    IPyraNet2DLayer* parentLayer;

    // a weight for each neuron
    std::vector<std::vector<double> > weights;

    void initWeights();
};

#endif // _IPyraNet2DLayer_h_