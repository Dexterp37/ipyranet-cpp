/* 
 *
 */

#include "IPyraNet.h"
#include "IPyraNet2DLayer.h"

IPyraNet::IPyraNet() {
    layers2D.clear();
}

IPyraNet::~IPyraNet() {
    destroy();
}

void IPyraNet::appendLayer(IPyraNet2DLayer* newLayer) {
    
    if (newLayer == NULL)
        return;

    layers2D.push_back(newLayer);
}

void IPyraNet::destroy() {

    size_t num2D = layers2D.size();

    for (size_t k = 0; k < num2D; ++k) {
        IPyraNet2DLayer* layer = layers2D[k];
        delete layer;
    }

    layers2D.clear();
}