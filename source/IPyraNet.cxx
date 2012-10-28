/* 
 *
 */

#include "IPyraNet.h"
#include "IPyraNetLayer.h"
#include <assert.h>

template <class NetType>
IPyraNet<NetType>::IPyraNet() {
    layers.clear();
}

template <class NetType>
IPyraNet<NetType>::~IPyraNet() {
    destroy();
}

template <class NetType>
void IPyraNet<NetType>::appendLayer(IPyraNetLayer<NetType>* newLayer) {
    
    if (newLayer == NULL)
        return;

    IPyraNetLayer<NetType>* lastLayer = NULL;

    // link this new layer to the last layer
    if (layers.size() > 0) 
        lastLayer = *(layers.end());

    newLayer->setParentLayer(lastLayer);
    layers.push_back(newLayer);
}

template <class NetType>
void IPyraNet<NetType>::destroy() {

    size_t num = layers.size();

    for (size_t k = 0; k < num; ++k) {
        IPyraNetLayer<NetType>* layer = layers[k];
        delete layer;
    }

    layers.clear();
}

// explicit instantiations
template class IPyraNet<float>;
template class IPyraNet<double>;