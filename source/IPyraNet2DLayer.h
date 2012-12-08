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
    void setNeuronWeight(int dimensions, int* neuronLocation, OutType value);
    OutType getNeuronBias(int dimensions, int* neuronLocation);
    void setNeuronBias(int dimensions, int* neuronLocation, OutType value);

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