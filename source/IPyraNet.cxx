/* 
 *
 */

#include "IPyraNet.h"
#include "IPyraNetLayer.h"
#include "IPyraNet2DSourceLayer.h"
#include "IPyraNet1DLayer.h"
#include "IPyraNet2DLayer.h"
#include "IPyraNetSigmoidFunction.h"
#include <assert.h>
#include "../3rdParties/dirent-1.12.1/dirent.h" // this is NOT cross platform!

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
    }

    return doc.save_file(fileName.c_str());
}
    
template <class NetType>
bool IPyraNet<NetType>::loadFromXML(const std::string& fileName) {

    // erase all data, if any
    destroy();

    // load the XML file
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(fileName.c_str());

    // traverse layers
    for (pugi::xml_node layer = doc.child("layer"); layer; layer = layer.next_sibling("layer")) {
        
        IPyraNetLayer<NetType>::LayerType layerType = (IPyraNetLayer<NetType>::LayerType)layer.attribute("type").as_int();

        IPyraNetLayer<NetType>* newLayer = NULL;

        switch (layerType) {
        case IPyraNetLayer<NetType>::Source:
            newLayer = new IPyraNet2DSourceLayer<NetType>();
            break;
        case IPyraNetLayer<NetType>::Layer1D:
            newLayer = new IPyraNet1DLayer<NetType>();
            newLayer->setActivationFunction(new IPyraNetSigmoidFunction<NetType>());
            break;
        case IPyraNetLayer<NetType>::Layer2D:
            newLayer = new IPyraNet2DLayer<NetType>();
            newLayer->setActivationFunction(new IPyraNetSigmoidFunction<NetType>());
            break;
        }

        newLayer->loadFromXML(layer);
        appendLayerNoInit(newLayer);
    }

    return result;
}

#include <iostream> // removeme

template <class NetType>
void IPyraNet<NetType>::train(const std::string& path) {
    
    std::string facePath(path);
    facePath.append("/face");
    NetType faceDesired[2] = { 1.0, 0.0 };

    std::string nonFacePath(path);
    nonFacePath.append("/non-face");
    NetType nonFaceDesired[2] = { 0.0, 1.0 };

    DIR *faceDir = opendir (facePath.c_str());
    if (faceDir == NULL) 
        return;

    struct dirent *ent;

    // iterate through files
    IPyraNet2DSourceLayer<NetType>* sourceLayer = ((IPyraNet2DSourceLayer<NetType>*)layers[0]);
    while ((ent = readdir (faceDir)) != NULL) {

        // skip "." and ".."
        if (ent->d_name[0] == '.')
            continue;

        //printf ("%s\n", ent->d_name);
        std::cout << "Processing " << ent->d_name;

        // extract the full filename
        std::string fullPath(facePath);
        fullPath.append("/");
        fullPath.append(ent->d_name);

        // process this image and compute the output
        if (!sourceLayer->load(fullPath.c_str())) {
            std::cout << "ERROR!" << std::endl;
            continue;
        }

        // compute network output
        std::vector<NetType> outputs;
        getOutput(outputs);

        // compute the error signal
        std::vector<NetType> errorSignal(outputs.size());
        for (int k = 0; k < errorSignal.size(); ++k)
            errorSignal[k] = faceDesired[k] - outputs[k];

        backpropagation_run(errorSignal);

        std::cout << " OUT [" << outputs[0] << " | " << outputs[1] << "] ";
        std::cout << " Err [" << errorSignal[0] << " | " << errorSignal[1] << "]" << std::endl;
    }

    closedir (faceDir);
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

template <class NetType>
void IPyraNet<NetType>::appendLayerNoInit(IPyraNetLayer<NetType>* newLayer) {
    
    if (newLayer == NULL)
        return;

    // link this new layer to the last layer
    if (layers.size() > 0) {
        IPyraNetLayer<NetType>* lastLayer = layers.back();
        newLayer->setParentLayer(lastLayer, false);
    }

    layers.push_back(newLayer);
}

template <class NetType>
void IPyraNet<NetType>::backpropagation_run(const std::vector<NetType>& errorSignal) {

    // compute deltas for the output layer
    IPyraNet1DLayer<NetType>* outputLayer = ((IPyraNet1DLayer<NetType>*)layers.back());
    
    int location[2] = {0, 0};
    int outputNeurons = 0;
    outputLayer->getSize(&outputNeurons);
    std::vector<NetType> outputDeltas;
    
    for (int n = 0; n < outputNeurons; ++n) {
        outputDeltas.push_back(outputLayer->getErrorSensitivity(1, location, errorSignal[n]));
    }

    // TODO: compute other 1D layers deltas
    // TODO: compute other 2D layers deltas
    // TODO: compute the error gradient
}

// explicit instantiations
template class IPyraNet<float>;
template class IPyraNet<double>;