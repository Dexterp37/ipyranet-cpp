/* 
 *
 */

#include "IPyraNet2DLayer.h"
#include <stdlib.h>

#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )

IPyraNet2DLayer::IPyraNet2DLayer() 
    : width(0),
    height(0),
    receptiveSize(0),
    overlap(0),
    inhibitorySize(0),
    parentLayer(NULL)
{

}

IPyraNet2DLayer::IPyraNet2DLayer(int width, int height) 
    : width(width),
    height(height),
    receptiveSize(0),
    overlap(0),
    inhibitorySize(0),
    parentLayer(NULL)
{
    initWeights();
}

IPyraNet2DLayer::~IPyraNet2DLayer() {

}

void IPyraNet2DLayer::setLayerSize(int width, int height) {
    
    this->width = width;
    this->height = height;

    initWeights();
}

void IPyraNet2DLayer::initWeights() {

    weights.resize(width);

    for (int u = 0; u < width; ++u) {

        weights[u].resize(height);

        for (int v = 0; v < height; ++v) {
            weights[u][v] = UNIFORM_PLUS_MINUS_ONE;
        }
    }
}