/* 
 *
 */

#include "IPyraNet1DLayer.h"
#include <assert.h>

#define UNIFORM_PLUS_MINUS_ONE ( static_cast<OutType>((2.0 * rand())/RAND_MAX - 1.0) )

template<class OutType>
IPyraNet1DLayer<OutType>::IPyraNet1DLayer() 
    : IPyraNetLayer<OutType>(),
    neurons(0)
{

}

template<class OutType>
IPyraNet1DLayer<OutType>::IPyraNet1DLayer(int numberNeurons, IPyraNetActivationFunction<OutType>* activationFunc) 
    : IPyraNetLayer<OutType>(),
    neurons(numberNeurons)
{
    setActivationFunction(activationFunc);
}

template<class OutType>
IPyraNet1DLayer<OutType>::~IPyraNet1DLayer() {

}

template<class OutType>
OutType IPyraNet1DLayer<OutType>::getErrorSensitivity(int dimensions, int* neuronLocation, OutType multiplier) {
    
    OutType accumulator = getWeightedSumInput(dimensions, neuronLocation);

    // apply the activation function
    OutType result = getActivationFunction()->derivative(accumulator);

    return result * multiplier;
}

template<class OutType>
OutType IPyraNet1DLayer<OutType>::getNeuronOutput(int dimensions, int* neuronLocation) {
    
    OutType accumulator = getWeightedSumInput(dimensions, neuronLocation);

    // apply the activation function
    OutType result = getActivationFunction()->compute(accumulator);

    return result;
}

template<class OutType>
int IPyraNet1DLayer<OutType>::getDimensions() const {
    return 1;
}

template<class OutType>
void IPyraNet1DLayer<OutType>::getSize(int* size) {
    assert(size != NULL);

    size[0] = neurons;
}

template<class OutType>
OutType IPyraNet1DLayer<OutType>::getNeuronWeight(int dimensions, int* neuronLocation) {
    // weights are "per connection", so there is a weight for each neuron in the
    // previous layer connecting to a neuron in this layer.
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0 && neuronLocation[1] >= 0);

    return weights[neuronLocation[0]][neuronLocation[1]];
}

template<class OutType>
void IPyraNet1DLayer<OutType>::setParentLayer(IPyraNetLayer<OutType>* parent, bool init) { 
    
    assert(parent != NULL);
    
    // calls base class
    IPyraNetLayer<OutType>::setParentLayer(parent);
    
    // init weights and biases
    if (init) {
        initWeights();
        initBiases();
    }
}

template<class OutType>
OutType IPyraNet1DLayer<OutType>::getWeightedSumInput(int dimensions, int* neuronLocation) {

    // sanity checks
    assert (dimensions == 1);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0);
    assert (getParentLayer() != NULL);
    assert (getActivationFunction() != NULL);

    // parent layer pointer
    IPyraNetLayer<OutType>* parent = getParentLayer();

    const int parentDims = parent->getDimensions();
    
    // get parent size
    int parentSize[2];
    parent->getSize(parentSize);
    
    int n = neuronLocation[0];
    OutType accumulator = 0;

    // we can connect to both 1D and 2D layers so handle both
    // cases
    int inputNeurons = 0; 
    if (parentDims == 2) {
        // This 1-D layer is connected to a 2-D layer.

        // weighted sum of each parent neuron connecting to this level's 
        // output neuron
        inputNeurons = parentSize[0] * parentSize[1];
        int parentNeuronLoc[2]; 
        int parentNeuronIndex = 0;

        for (int m_u = 0; m_u < parentSize[0]; ++m_u) {

            parentNeuronLoc[0] = m_u;
            
            for (int m_v = 0; m_v < parentSize[1]; ++m_v) {
                parentNeuronLoc[1] = m_v;
                OutType parentNeuronOutput = parent->getNeuronOutput(2, parentNeuronLoc);

                parentNeuronIndex = (m_v * parentSize[1]) + m_u;
                OutType connectionWeight = weights[parentNeuronIndex][n];

                accumulator += parentNeuronOutput * connectionWeight;
            }
        }
    } else {
        // This 1-D layer is connected to a 1-D layer.
        inputNeurons = parentSize[0];

        for (int m = 0; m < inputNeurons; ++m) {
            OutType parentNeuronOutput = parent->getNeuronOutput(2, &m);

            OutType connectionWeight = weights[m][n];

            accumulator += parentNeuronOutput * connectionWeight;
        }
    }

    // apply the bias
    accumulator += biases[n];

    return accumulator;
}

template<class OutType>
void IPyraNet1DLayer<OutType>::initWeights() {

    assert(getParentLayer() != NULL);

    IPyraNetLayer<OutType>* parent = getParentLayer();

    const int parentDims = parent->getDimensions();
    
    // get parent size
    int parentSize[2];
    parent->getSize(parentSize);

    // we can connect to both 1D and 2D layers so handle both
    // cases
    int inputNeurons = 0; 
    if (parentDims == 2) {
        inputNeurons = parentSize[0] * parentSize[1];
    } else
        inputNeurons = parentSize[0];

    // we need to have a weight for each connection going from
    // each input to every neuron in this layer.
    weights.resize(inputNeurons);

    for (int u = 0; u < inputNeurons; ++u) {

        weights[u].resize(neurons);

        for (unsigned int v = 0; v < neurons; ++v) {
            weights[u][v] = UNIFORM_PLUS_MINUS_ONE;
        }
    }
}

template<class OutType>
void IPyraNet1DLayer<OutType>::initBiases() {
    
    biases.resize(neurons);

    for (unsigned int u = 0; u < neurons; ++u) {
        biases[u] = UNIFORM_PLUS_MINUS_ONE;
    }
}

template<class OutType>
void IPyraNet1DLayer<OutType>::saveToXML(pugi::xml_node& node) {

    // save the size
    pugi::xml_attribute attr = node.append_attribute("neurons");
    attr.set_value(neurons);

    // dump the weights
    size_t inputNeurons = weights.size();
    pugi::xml_node weightsNode = node.append_child("weights");
    weightsNode.append_attribute("inputs").set_value(inputNeurons);

    for (unsigned int u = 0; u < inputNeurons; ++u) {
        for (unsigned int v = 0; v < neurons; ++v) {
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
    for (unsigned int biasIndex = 0; biasIndex < neurons; ++biasIndex) {
        pugi::xml_node biasNode = biasesNode.append_child("bias");
            
        // weight indices as attributes
        pugi::xml_attribute indexAttr = biasNode.append_attribute("index");
        indexAttr.set_value(biasIndex);

        pugi::xml_attribute biasAttr = biasNode.append_attribute("value");
        biasAttr.set_value(biases[biasIndex]);
    }
}

template<class OutType>
void IPyraNet1DLayer<OutType>::loadFromXML(pugi::xml_node& node) {

    neurons = node.attribute("neurons").as_int();

    // reshape weights buffer and load weights
    size_t inputNeurons = node.child("weights").attribute("inputs").as_uint();
    weights.resize(inputNeurons);
    for (size_t k = 0; k < inputNeurons; ++k) 
        weights[k].resize(neurons);

    // actual load from XML
    for (pugi::xml_node weight = node.child("weights").child("weight"); weight; weight = weight.next_sibling("weight")) {

        size_t weightIndex1 = weight.attribute("index1").as_uint();
        size_t weightIndex2 = weight.attribute("index2").as_uint();

        weights[weightIndex1][weightIndex2] = static_cast<OutType>(weight.attribute("value").as_double());
    }  

    // load biases
    biases.resize(neurons);
    for (pugi::xml_node bias = node.child("biases").child("bias"); bias; bias = bias.next_sibling("bias")) {

        size_t biasIndex = bias.attribute("index").as_uint();
        biases[biasIndex] = static_cast<OutType>(bias.attribute("value").as_double());
    }  
}

// explicit instantiations
template class IPyraNet1DLayer<float>;
template class IPyraNet1DLayer<double>;