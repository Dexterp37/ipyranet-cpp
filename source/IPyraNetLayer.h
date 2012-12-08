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

#ifndef _IPyranetLayer_h_
#define _IPyranetLayer_h_

#include "IPyraNetActivationFunction.h"
#include "../3rdParties/pugixml-1.2/src/pugixml.hpp"

//#define UNIFORM_PLUS_MINUS_ONE ( static_cast<OutType>((2.0 * rand())/RAND_MAX - 1.0) )
#define UNIFORM_PLUS_MINUS_ONE ( static_cast<OutType>(((4.0)*((float)rand()/RAND_MAX))-2.0) )

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

    virtual OutType getErrorSensitivity(int dimensions, int* neuronLocation, OutType multiplier) = 0;
    virtual OutType getNeuronOutput(int dimensions, int* neuronLocation) = 0;
    virtual int getDimensions() const = 0;
    virtual void getSize(int* size) = 0;

    virtual OutType getNeuronWeight(int dimensions, int* neuronLocation) = 0;
    virtual void setNeuronWeight(int dimensions, int* neuronLocation, OutType value) = 0;

    virtual OutType getNeuronBias(int dimensions, int* neuronLocation) = 0;
    virtual void setNeuronBias(int dimensions, int* neuronLocation, OutType value) = 0;

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