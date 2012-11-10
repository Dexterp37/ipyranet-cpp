/*

*/

#ifndef _IPyranetLayer_h_
#define _IPyranetLayer_h_

#include "IPyraNetActivationFunction.h"
#include "../3rdParties/pugixml-1.2/src/pugixml.hpp"

template<typename OutType>
class IPyraNetLayer {
public:

    enum LayerType {
        Unknown = -1,
        Source = 1,
        Layer2D,
        Layer1D
    };

    IPyraNetLayer() : parentLayer(0), activationFunction(0) { };
    virtual ~IPyraNetLayer() { 
        parentLayer = 0; 
        if (activationFunction) 
            delete activationFunction; 
        
        activationFunction = 0;
    };

    virtual LayerType getLayerType() const = 0;

    virtual OutType getNeuronOutput(int dimensions, int* neuronLocation) = 0;
    virtual int getDimensions() const = 0;
    virtual void getSize(int* size) = 0;

    virtual void setParentLayer(IPyraNetLayer<OutType>* parent, bool init = true) { parentLayer = parent; }
    IPyraNetLayer<OutType>* getParentLayer() { return parentLayer; }

    void setActivationFunction(IPyraNetActivationFunction<OutType>* func) { activationFunction = func; }
    IPyraNetActivationFunction<OutType>* getActivationFunction() { return activationFunction; }

    // serialization helper
    virtual void saveToXML(pugi::xml_node& node) = 0;
    virtual void loadFromXML(pugi::xml_node& node) = 0;

private:
    // previous (adjacent bigger) layer in the pyramid
    IPyraNetLayer<OutType>* parentLayer;
    IPyraNetActivationFunction<OutType>* activationFunction;
};

#endif // _IPyranetLayer_h_