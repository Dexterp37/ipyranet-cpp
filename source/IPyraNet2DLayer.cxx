/* 
 *
 */

#include "IPyraNet2DLayer.h"
#include <assert.h>

#define UNIFORM_PLUS_MINUS_ONE ( static_cast<OutType>((2.0 * rand())/RAND_MAX - 1.0) )

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
OutType IPyraNet2DLayer<OutType>::getNeuronOutput(int dimensions, int* neuronLocation) {
    
    // sanity checks
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] > 0 && neuronLocation[1] > 0);
    assert (getParentLayer() != NULL);
    assert (getActivationFunction() != NULL);

    // parent layer pointer
    IPyraNetLayer<OutType>* parent = getParentLayer();

    // compute the gap
    const int gap = receptiveSize - overlap;

    // just for compliance with the article
    const int u = neuronLocation[0];
    const int v = neuronLocation[1];

    OutType receptiveAccumulator = 0;
    OutType inhibitoryAccumulator = 0;
    OutType bias = biases[u][v];

    int parentLoc[2];

    // iterate through the neurons inside the receptive field of the previous layer
    const int max_u = (u - 1) * gap + receptiveSize;
    const int max_v = (v - 1) * gap + receptiveSize;

    for (int i = (u - 1) * gap + 1; i <= max_u; ++i) {

        parentLoc[0] = i;
        
        for (int j = (v - 1) * gap + 1; j <= max_v; ++j) {
            
            parentLoc[1] = j;

            OutType parentOutput = parent->getNeuronOutput(2, parentLoc);
            OutType weight = weights[i][j];

            receptiveAccumulator += parentOutput * weight;
        }
    }

    // iterate through the neurons inside the inhibitory field
    // TODO

    OutType result = getActivationFunction()->compute(receptiveAccumulator - inhibitoryAccumulator + bias);

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
void IPyraNet2DLayer<OutType>::setParentLayer(IPyraNetLayer<OutType>* parent) { 
    
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

    width = static_cast<int>(floor(static_cast<float>(parentSize[0] - overlap) / gap));
    height = static_cast<int>(floor(static_cast<float>(parentSize[1] - overlap) / gap));

    // init weights and biases
    initWeights();
    initBiases();
}

// explicit instantiations
template class IPyraNet2DLayer<float>;
template class IPyraNet2DLayer<double>;