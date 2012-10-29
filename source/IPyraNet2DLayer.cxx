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
IPyraNet2DLayer<OutType>::IPyraNet2DLayer(int width, int height) 
    : IPyraNetLayer<OutType>(),
    width(width),
    height(height),
    receptiveSize(0),
    overlap(0),
    inhibitorySize(0)
{
    initWeights();
}

template<class OutType>
IPyraNet2DLayer<OutType>::~IPyraNet2DLayer() {

}

template<class OutType>
void IPyraNet2DLayer<OutType>::setLayerSize(int width, int height) {
    
    this->width = width;
    this->height = height;

    initWeights();
}

template<class OutType>
void IPyraNet2DLayer<OutType>::initWeights() {

    weights.resize(width);

    for (int u = 0; u < width; ++u) {

        weights[u].resize(height);

        for (int v = 0; v < height; ++v) {
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

    // parent layer pointer
    IPyraNetLayer<OutType>* parent = getParentLayer();

    // compute the gap
    const int gap = receptiveSize - overlap;

    // just for compliance with the article
    const int u = neuronLocation[0];
    const int v = neuronLocation[1];

    OutType fieldAccumulator = 0;

    int parentLoc[2];

    // iterate through the neurons inside the receptive field of the previous layer
    // TODO: optimize (bring the condition outside the loop)
    for (int i = (u - 1) * gap + 1; i <= ((u - 1) * gap + receptiveSize); ++i) {

        parentLoc[0] = i;
        
        for (int j = (v - 1) * gap + 1; j <= ((v - 1) * gap + receptiveSize); ++j) {
            
            parentLoc[1] = j;

            OutType parentOutput = parent->getNeuronOutput(2, parentLoc);
            OutType weight = weights[i][j]; // TODO: should we take the weight of the last level? I think yes!
            OutType bias = biases[u][v];

            fieldAccumulator += parentOutput * weight + bias;
        }
    }

    // TODO: apply the sigmoid function to the fieldAccumulator
    OutType result = fieldAccumulator;

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

// explicit instantiations
template class IPyraNet2DLayer<float>;
template class IPyraNet2DLayer<double>;