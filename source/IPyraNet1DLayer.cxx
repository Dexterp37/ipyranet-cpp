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
OutType IPyraNet1DLayer<OutType>::getNeuronOutput(int dimensions, int* neuronLocation) {
    
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
void IPyraNet1DLayer<OutType>::setParentLayer(IPyraNetLayer<OutType>* parent) { 
    
    assert(parent != NULL);
    
    // calls base class
    IPyraNetLayer<OutType>::setParentLayer(parent);
    
    // init weights and biases
    initWeights();
    initBiases();
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

        for (int v = 0; v < neurons; ++v) {
            weights[u][v] = UNIFORM_PLUS_MINUS_ONE;
        }
    }
}

template<class OutType>
void IPyraNet1DLayer<OutType>::initBiases() {
    
    biases.resize(neurons);

    for (int u = 0; u < neurons; ++u) {
        biases[u] = UNIFORM_PLUS_MINUS_ONE;
    }
}


// explicit instantiations
template class IPyraNet1DLayer<float>;
template class IPyraNet1DLayer<double>;