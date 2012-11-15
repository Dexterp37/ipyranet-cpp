/* 
 *
 */

#ifndef _IPyraNet2DSourceLayer_h_
#define _IPyraNet2DSourceLayer_h_

#include "IPyraNetLayer.h"
#include <opencv2/core/core.hpp>
#include <string>

template<class OutType>
class IPyraNet2DSourceLayer : public IPyraNetLayer<OutType> {
public:
    IPyraNet2DSourceLayer();
    IPyraNet2DSourceLayer(const std::string& fileName);
    virtual ~IPyraNet2DSourceLayer();
    
    LayerType getLayerType() const { return Source; }

    bool load(const std::string& fileName);
    
    OutType getErrorSensitivity(int dimensions, int* neuronLocation, OutType multiplier) { return 0; };
    OutType getNeuronOutput(int dimensions, int* neuronLocation);    
    int getDimensions() const;
    void getSize(int* size);
    OutType getNeuronWeight(int dimensions, int* neuronLocation) { return 0; }
    void setNeuronWeight(int dimensions, int* neuronLocation, OutType value) { };
    OutType getNeuronBias(int dimensions, int* neuronLocation) { return 0; }
    void setNeuronBias(int dimensions, int* neuronLocation, OutType value) { };

    void saveToXML(pugi::xml_node& node);
    void loadFromXML(pugi::xml_node& node);

private:
    cv::Mat source;
};

#endif // _IPyraNet2DSourceLayer_h_