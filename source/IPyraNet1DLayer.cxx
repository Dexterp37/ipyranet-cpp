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
IPyraNet1DLayer<OutType>::IPyraNet1DLayer(int receptive, int inhibitory, int overlap, IPyraNetActivationFunction<OutType>* activationFunc) 
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
    /*
    const int dims = parent->getDimensions();

    // we can just connect 2d layers to 2d layers
    assert(dims == 2);

    // get parent size
    int parentSize[2];
    parent->getSize(parentSize);

    // compute the gap
    const float gap = static_cast<float>(receptiveSize - overlap);

    width = static_cast<int>(floor(static_cast<float>(parentSize[0] - overlap) / gap));
    height = static_cast<int>(floor(static_cast<float>(parentSize[1] - overlap) / gap));

    // init weights and biases
    initWeights();
    initBiases();*/
}

// explicit instantiations
template class IPyraNet1DLayer<float>;
template class IPyraNet1DLayer<double>;