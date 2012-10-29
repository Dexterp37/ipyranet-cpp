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
IPyraNet1DLayer<OutType>::IPyraNet1DLayer(IPyraNetActivationFunction<OutType>* activationFunc) 
    : IPyraNetLayer<OutType>(),
    neurons(0)
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
    assert (neuronLocation[0] > 0);
    assert (getParentLayer() != NULL);
    assert (getActivationFunction() != NULL);

    return 0;
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
    
    const int parentDims = parent->getDimensions();
    
    // get parent size
    int parentSize[2];
    parent->getSize(parentSize);

    // we can connect to 2d layers and 1d layers
    if (parentDims == 2) {
        neurons = parentSize[0] * parentSize[1];
    } else
        neurons = parentSize[0];

    // TODO: is this the same amount of neurons of last layer!?

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