/* 
 *
 */

#include "IPyraNet.h"
#include "IPyraNetLayer.h"
#include "IPyraNet1DLayer.h"
#include <assert.h>

template <class NetType>
IPyraNet<NetType>::IPyraNet() 
    : trainingEpochs(0),
    trainingTechnique(Unknown) 
{
    layers.clear();
}

template <class NetType>
IPyraNet<NetType>::~IPyraNet() {
    destroy();
}


template <class NetType>
bool IPyraNet<NetType>::saveToXML(const std::string& fileName) {
        
    pugi::xml_document doc;

    size_t num = layers.size();

    for (size_t k = 0; k < num; ++k) {
        IPyraNetLayer<NetType>* layer = layers[k];

        pugi::xml_node node = doc.append_child("layer");
        pugi::xml_attribute attr = node.append_attribute("type");
        attr.set_value(layer->getLayerType());

        layer->saveToXML(node);
        /*
        // save layer specific information. TODO: This code is not OO
        // But hey. I don't have enough time atm.
        switch (layer->getLayerType()) {
        case IPyraNetLayer<NetType>::Layer1D: {
                
                IPyraNet1DLayer<NetType>* layer1D = static_cast<IPyraNet1DLayer<NetType>* >(layer);
                
                // size/number of neurons
                int size1D;
                layer1D->getSize(&size1D);
                attr = node.append_attribute("neurons");
                attr.set_value(size1D);

                // dump weights
            } break;

        case IPyraNetLayer<NetType>::Layer2D: {

            } break;

        case IPyraNetLayer<NetType>::Source: {

            } break;

        case IPyraNetLayer<NetType>::Unknown:
            // falls  through
        default:
            continue;
        }*/
    }

    layers.clear();

    return doc.save_file(fileName.c_str());
}
    
template <class NetType>
bool IPyraNet<NetType>::loadFromXML(const std::string& fileName) {
    return false;
}

template <class NetType>
void IPyraNet<NetType>::appendLayer(IPyraNetLayer<NetType>* newLayer) {
    
    if (newLayer == NULL)
        return;

    // link this new layer to the last layer
    if (layers.size() > 0) {
        IPyraNetLayer<NetType>* lastLayer = layers.back();
        newLayer->setParentLayer(lastLayer);
    }

    layers.push_back(newLayer);
}
    
template <class NetType>
void IPyraNet<NetType>::getOutput(std::vector<NetType>& outputs) {
    
    if (layers.size() < 1)
        return;

    // last layer must be an 1-D layer
    IPyraNetLayer<NetType>* lastLayer = layers.back();
    if (lastLayer->getDimensions() != 1)
        return;
    
    int size = 0;
    lastLayer->getSize(&size);

    for (int n = 0; n < size; ++n) {
        NetType out = lastLayer->getNeuronOutput(1, &n);
        outputs.push_back(out);
    }
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

template <class NetType>
void IPyraNet<NetType>::setTrainingEpochs(int epochs) {
    trainingEpochs = epochs;
}

template <class NetType>
int IPyraNet<NetType>::getTrainingEpochs() const {
    return trainingEpochs;
}

template <class NetType>
void IPyraNet<NetType>::setTrainingTechnique(TrainingTechnique technique) {
    trainingTechnique = technique;
}
/*
template <class NetType>
IPyraNet<NetType>::TrainingTechnique IPyraNet<NetType>::getTrainingTechnique() const {
    return trainingTechnique;
}*/

// explicit instantiations
template class IPyraNet<float>;
template class IPyraNet<double>;