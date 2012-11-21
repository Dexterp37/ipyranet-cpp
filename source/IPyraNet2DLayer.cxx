/* 
 *
 */

#include "IPyraNet2DLayer.h"
#include <assert.h>

template<class OutType>
IPyraNet2DLayer<OutType>::IPyraNet2DLayer() 
    : IPyraNetLayer<OutType>(),
    width(0),
    height(0),
    receptiveSize(0),
    overlap(0),
    inhibitorySize(0)
{

}

template<class OutType>
IPyraNet2DLayer<OutType>::IPyraNet2DLayer(int receptive, int inhibitory, int overlap, IPyraNetActivationFunction<OutType>* activationFunc) 
    : IPyraNetLayer<OutType>(),
    width(0),
    height(0),
    receptiveSize(receptive),
    overlap(overlap),
    inhibitorySize(inhibitory)
{
    setActivationFunction(activationFunc);
}

template<class OutType>
IPyraNet2DLayer<OutType>::~IPyraNet2DLayer() {

}

template<class OutType>
OutType IPyraNet2DLayer<OutType>::getWeightedSumInput(int dimensions, int* neuronLocation) {
    
    // sanity checks
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0 && neuronLocation[1] >= 0);
    assert (getParentLayer() != NULL);
    assert (getActivationFunction() != NULL);

    // parent layer pointer
    IPyraNetLayer<OutType>* parent = getParentLayer();

    // compute the gap
    const int gap = receptiveSize - overlap;

    // just for compliance with the article (which uses 1..n indices)
    const int u = neuronLocation[0] + 1;
    const int v = neuronLocation[1] + 1;

    OutType receptiveAccumulator = 0;
    OutType bias = biases[neuronLocation[0]][neuronLocation[1]];

    int parentLoc[2];

    int parentSize[2];
    parent->getSize(parentSize);

    // iterate through the neurons inside the receptive field of the previous layer
    //
    // ******
    // **uv**
    // ******
    //
    const int min_u = (u - 1) * gap + 1;
    const int max_u = (u - 1) * gap + receptiveSize;
    const int min_v = (v - 1) * gap + 1;
    const int max_v = (v - 1) * gap + receptiveSize;

    for (int i = min_u; i <= max_u; ++i) {

        // indices on the paper (2) go from 1 to n. Our indices go from 0 to n-1
        parentLoc[0] = i - 1;
        
        // ignore neuron indices which fall outside of the layer
        if (parentLoc[0] < 0)
            continue;

        if (parentLoc[0] > parentSize[0])
            continue;
        
        for (int j = min_v; j <= max_v; ++j) {
            
            parentLoc[1] = j - 1;

            // ignore neuron indices which fall outside of the layer
            if (parentLoc[1] < 0)
                continue;

            if (parentLoc[1] > parentSize[1])
                continue;
            
            OutType parentOutput = parent->getNeuronOutput(2, parentLoc);
            OutType weight = weights[parentLoc[0]][parentLoc[1]];

            receptiveAccumulator += parentOutput * weight;
        }
    }

    // iterate through the neurons inside the inhibitory field
    // 
    // xxxxxxxx
    // x******x
    // x**uv**x
    // x******x
    // xxxxxxxx
    //
    // the inhibitory field is 'x' and the receptive field is '*'
    OutType inhibitoryAccumulator = 0;
    const int inhibitory_min_u = min_u - inhibitorySize;
    const int inhibitory_max_u = max_u + inhibitorySize;
    const int inhibitory_min_v = min_v - inhibitorySize;
    const int inhibitory_max_v = max_v + inhibitorySize;
    
    for (int i = inhibitory_min_u; i <= inhibitory_max_u; ++i) {
        
        // indices on the paper (2) go from 1 to n. Our indices go from 0 to n-1
        parentLoc[0] = i - 1;
        
        // ignore neuron indices which fall outside of the layer
        if (parentLoc[0] < 0)
            continue;

        if (parentLoc[0] > parentSize[0])
            continue;

        for (int j = inhibitory_min_v; j <= inhibitory_max_v; ++j) {

            parentLoc[1] = j - 1;

            // ignore neuron indices which fall outside of the layer
            if (parentLoc[1] < 0)
                continue;

            if (parentLoc[1] > parentSize[1])
                continue;

            // ignore neurons in the receptive field!
            if (i >= min_u && i <= max_u)
                continue;

            if (j >= min_v && j <= max_v)
                continue;

            OutType parentOutput = parent->getNeuronOutput(2, parentLoc);
            OutType weight = weights[parentLoc[0]][parentLoc[1]];

            inhibitoryAccumulator += parentOutput * weight;
        }
    }

    return receptiveAccumulator - inhibitoryAccumulator + bias;
}

template<class OutType>
void IPyraNet2DLayer<OutType>::initWeights() {

    assert(getParentLayer() != NULL);

    IPyraNetLayer<OutType>* parent = getParentLayer();
    
    // get parent size
    int parentSize[2];
    parent->getSize(parentSize);

    // we need to have as many weights as the number of neurons in last
    // layer.
    weights.resize(parentSize[0]);

    for (int u = 0; u < parentSize[0]; ++u) {

        weights[u].resize(parentSize[1]);

        for (int v = 0; v < parentSize[1]; ++v) {
            weights[u][v] = UNIFORM_PLUS_MINUS_ONE;
        }
    }
}

template<class OutType>
void IPyraNet2DLayer<OutType>::initBiases() {

    biases.resize(width);

    for (int u = 0; u < width; ++u) {

        biases[u].resize(height);

        for (int v = 0; v < height; ++v) {
            biases[u][v] = UNIFORM_PLUS_MINUS_ONE;
        }
    }
}

template<class OutType>
OutType IPyraNet2DLayer<OutType>::getErrorSensitivity(int dimensions, int* neuronLocation, OutType multiplier) {
    
    OutType accumulator = getWeightedSumInput(dimensions, neuronLocation);

    // apply the activation function
    OutType result = getActivationFunction()->derivative(accumulator);

    return result * multiplier;
}

template<class OutType>
OutType IPyraNet2DLayer<OutType>::getNeuronOutput(int dimensions, int* neuronLocation) {

    OutType accumulator = getWeightedSumInput(dimensions, neuronLocation);    

    OutType result = getActivationFunction()->compute(accumulator);

    return result;
}

template<class OutType>
int IPyraNet2DLayer<OutType>::getDimensions() const {
    return 2;
}

template<class OutType>
void IPyraNet2DLayer<OutType>::getSize(int* size) {
    assert(size != NULL);

    size[0] = width;
    size[1] = height;
}

template<class OutType>
OutType IPyraNet2DLayer<OutType>::getNeuronWeight(int dimensions, int* neuronLocation) {
    // weights are "per neuron", so there is a weight for each parent's neuron
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0 && neuronLocation[1] >= 0);

    return weights[neuronLocation[0]][neuronLocation[1]];
}

template<class OutType>
void IPyraNet2DLayer<OutType>::setNeuronWeight(int dimensions, int* neuronLocation, OutType value) {
    // weights are "per neuron", so there is a weight for each parent's neuron
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0 && neuronLocation[1] >= 0);

    weights[neuronLocation[0]][neuronLocation[1]] = value;
}

template<class OutType>
OutType IPyraNet2DLayer<OutType>::getNeuronBias(int dimensions, int* neuronLocation) {
    // one bias for each neuron in this layer
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0 && neuronLocation[1] >= 0);

    return biases[neuronLocation[0]][neuronLocation[1]];
}

template<class OutType>
void IPyraNet2DLayer<OutType>::setNeuronBias(int dimensions, int* neuronLocation, OutType value) {
    // one bias for each neuron in this layer
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0 && neuronLocation[1] >= 0);

    biases[neuronLocation[0]][neuronLocation[1]] = value;
}

template<class OutType>
void IPyraNet2DLayer<OutType>::setParentLayer(IPyraNetLayer<OutType>* parent, bool init) { 
    
    assert(parent != NULL);
    assert(receptiveSize != 0);
    assert(overlap != 0);

    // calls base class
    IPyraNetLayer<OutType>::setParentLayer(parent);

    const int dims = parent->getDimensions();

    // we can just connect 2d layers to 2d layers
    assert(dims == 2);

    // get parent size
    int parentSize[2];
    parent->getSize(parentSize);

    // compute the gap
    const float gap = static_cast<float>(receptiveSize - overlap);

    int newWidth = static_cast<int>(floor(static_cast<float>(parentSize[0] - overlap) / gap));
    int newHeight = static_cast<int>(floor(static_cast<float>(parentSize[1] - overlap) / gap));

    // init weights and biases
    if (init) {
        width = newWidth;
        height = newHeight;

        initWeights();
        initBiases();
    } /*else {
        assert (width == newWidth);
        assert (height == newHeight);
    }*/
}

template<class OutType>
void IPyraNet2DLayer<OutType>::saveToXML(pugi::xml_node& node) {

    // save the size
    pugi::xml_attribute widthAttr = node.append_attribute("width");
    widthAttr.set_value(width);

    pugi::xml_attribute heightAttr = node.append_attribute("height");
    heightAttr.set_value(height);

    // receptive, overlap and inhibitory
    pugi::xml_attribute receptiveAttr = node.append_attribute("receptive");
    receptiveAttr.set_value(receptiveSize);
    
    pugi::xml_attribute overlapAttr = node.append_attribute("overlap");
    overlapAttr.set_value(overlap);
    
    pugi::xml_attribute inhibitoryAttr = node.append_attribute("inhibitory");
    inhibitoryAttr.set_value(inhibitorySize);

    // dump the weights
    pugi::xml_node weightsNode = node.append_child("weights");
    weightsNode.append_attribute("dim1").set_value(weights.size());
    weightsNode.append_attribute("dim2").set_value(weights[0].size());

    for (unsigned int u = 0; u < weights.size(); ++u) {
        for (unsigned int v = 0; v < weights[u].size(); ++v) {
            pugi::xml_node weightNode = weightsNode.append_child("weight");
            
            // weight indices as attributes
            pugi::xml_attribute index1Attr = weightNode.append_attribute("index1");
            index1Attr.set_value(u);

            pugi::xml_attribute index2Attr = weightNode.append_attribute("index2");
            index2Attr.set_value(v);

            pugi::xml_attribute weightAttr = weightNode.append_attribute("value");
            weightAttr.set_value(weights[u][v]);
        }
    }

    // dump the biases
    pugi::xml_node biasesNode = node.append_child("biases");
    biasesNode.append_attribute("dim1").set_value(biases.size());
    biasesNode.append_attribute("dim2").set_value(biases[0].size());
    for (int u = 0; u < width; ++u) {
        for (int v = 0; v < height; ++v) {
            pugi::xml_node biasNode = biasesNode.append_child("bias");
            
            // weight indices as attributes
            pugi::xml_attribute index1Attr = biasNode.append_attribute("index1");
            index1Attr.set_value(u);

            pugi::xml_attribute index2Attr = biasNode.append_attribute("index2");
            index2Attr.set_value(v);

            pugi::xml_attribute biasAttr = biasNode.append_attribute("value");
            biasAttr.set_value(biases[u][v]);
        }
    }
}

template<class OutType>
void IPyraNet2DLayer<OutType>::loadFromXML(pugi::xml_node& node) {

    // load the size
    width = node.attribute("width").as_int();
    height = node.attribute("height").as_int();

    // receptive, inhibitory and overlaps
    receptiveSize = node.attribute("receptive").as_int();
    overlap = node.attribute("overlap").as_int();
    inhibitorySize = node.attribute("inhibitory").as_int();

    // reshape weights buffer and load weights
    size_t dim1 = node.child("weights").attribute("dim1").as_uint();
    size_t dim2 = node.child("weights").attribute("dim2").as_uint();

    weights.resize(dim1);
    for (size_t k = 0; k < dim1; ++k) 
        weights[k].resize(dim2);

    // actual load from XML
    for (pugi::xml_node weight = node.child("weights").child("weight"); weight; weight = weight.next_sibling("weight")) {

        size_t weightIndex1 = weight.attribute("index1").as_uint();
        size_t weightIndex2 = weight.attribute("index2").as_uint();

        weights[weightIndex1][weightIndex2] = static_cast<OutType>(weight.attribute("value").as_double());
    }  

    // load biases
    dim1 = node.child("biases").attribute("dim1").as_uint();
    dim2 = node.child("biases").attribute("dim2").as_uint();

    biases.resize(dim1);
    for (size_t k = 0; k < dim1; ++k) 
        biases[k].resize(dim2);

    for (pugi::xml_node bias = node.child("biases").child("bias"); bias; bias = bias.next_sibling("bias")) {

        size_t biasIndex1 = bias.attribute("index1").as_uint();
        size_t biasIndex2 = bias.attribute("index2").as_uint();

        biases[biasIndex1][biasIndex2] = static_cast<OutType>(bias.attribute("value").as_double());
    }  
}

// explicit instantiations
template class IPyraNet2DLayer<float>;
template class IPyraNet2DLayer<double>;