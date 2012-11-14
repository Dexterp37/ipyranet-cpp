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

    initDeltaStorage();
    initGradientStorage();

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
        for (size_t k = 0; k < errorSignal.size(); ++k)
            errorSignal[k] = faceDesired[k] - outputs[k];

        backpropagation(errorSignal);

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
void IPyraNet<NetType>::initDeltaStorage() {

    // initialize delta storage for each layer
    layersDeltas.resize(layers.size());

    int layerSize[2];
    for (unsigned int l = 0; l < layers.size(); ++l) {
        layers[l]->getSize(layerSize);

        layersDeltas[l].deltas.resize(layerSize[0]);

        // this is a 2D layer
        if (layers[l]->getDimensions() == 2) {

            for (int dx = 0; dx < layerSize[0]; ++dx) {
                layersDeltas[l].deltas[dx].resize(layerSize[1]);
            }

        } else { // this is a 1D layer

            for (int dx = 0; dx < layerSize[0]; ++dx) {
                layersDeltas[l].deltas[dx].resize(1);
            }
        }
    }
}

template <class NetType>
void IPyraNet<NetType>::initGradientStorage() {
    
    // initialize gradient storage for each layer
    layersGradient.resize(layers.size());

    // get the size of source layer
    int parentSize[2];
    int parentDims = 2;

    layers[0]->getSize(parentSize);

    for (unsigned int l = 1; l < layers.size(); ++l) {

        // 2D layer
        if (layers[l]->getDimensions() == 2) {

            // we need to have as many weights as the number of neurons in last
            // layer.
            layersGradient[l].weightsGrad.resize(parentSize[0]);

            for (int u = 0; u < parentSize[0]; ++u) {

                layersGradient[l].weightsGrad[u].resize(parentSize[1]);

                for (int v = 0; v < parentSize[1]; ++v) {
                    layersGradient[l].weightsGrad[u][v] = 0.0;
                }
            }

            // now the weights gradient. A bias for each neuron of current layer.
            // save it in "parentSize", so that it is already updated.
            layers[l]->getSize(parentSize);

            layersGradient[l].biasesGrad.resize(parentSize[0]);

            for (int u = 0; u < parentSize[0]; ++u) {

                layersGradient[l].biasesGrad[u].resize(parentSize[1]);

                for (int v = 0; v < parentSize[1]; ++v) {
                    layersGradient[l].biasesGrad[u][v] = 0.0;
                }
            }

        } else {    // 1D layer
                
            // we can connect to both 1D and 2D layers so handle both
            // cases
            int inputNeurons = 0; 
            if (parentDims == 2) {
                inputNeurons = parentSize[0] * parentSize[1];
            } else
                inputNeurons = parentSize[0];

            
            int neurons = 0;
            layers[l]->getSize(&neurons);

            // we need to have a weight for each connection going from
            // each input to every neuron in this layer.
            layersGradient[l].weightsGrad.resize(inputNeurons);

            for (int u = 0; u < inputNeurons; ++u) {

                layersGradient[l].weightsGrad[u].resize(neurons);

                for (unsigned int v = 0; v < neurons; ++v) {
                    layersGradient[l].weightsGrad[u][v] = 0.0;
                }
            }

            // now enough room for the biases    
            layersGradient[l].biasesGrad.resize(neurons);

            for (unsigned int u = 0; u < neurons; ++u) {
                layersGradient[l].biasesGrad[u].resize(1);
                layersGradient[l].biasesGrad[u][0] = 0.0;
            }

        }

        // update parent dims
        layers[l]->getSize(parentSize);
        parentDims = layers[l]->getDimensions();
    }
}

template <class NetType>
void IPyraNet<NetType>::backpropagation(const std::vector<NetType>& errorSignal) {

    // compute sensitivities (deltas)
    computeErrorSensitivities(errorSignal);

    // compute the gradient
    computeGradient();
}

template <class NetType>
void IPyraNet<NetType>::computeErrorSensitivities(const std::vector<NetType>& errorSignal) {
    
    // compute deltas (error sensitivities) for the output layer
    int currentLayer = layers.size() - 1;
    int location[2] = {0, 0};
    int outputNeurons = 0;

    layers[currentLayer]->getSize(&outputNeurons);
    
    for (int n = 0; n < outputNeurons; ++n) {
        
        // get error sensitivity for neuron n
        location[0] = n;

        layersDeltas[currentLayer].deltas[n][0] = layers[currentLayer]->getErrorSensitivity(1, location, errorSignal[n]);
    }

    // compute other 1D layers deltas
    int lastLayerNeurons = outputNeurons;
    currentLayer = layers.size() - 2;
    for (currentLayer; currentLayer > 0; --currentLayer) {

        if (layers[currentLayer]->getLayerType() != IPyraNetLayer<NetType>::Layer1D)
            break;
        
        // compute a delta for each neuron on this layer
        layers[currentLayer]->getSize(&outputNeurons);

        for (int n = 0; n < outputNeurons; ++n) {

            // compute the inner summation of (16) in the paper
            NetType summation = 0;
            int weightLocation[2] = {n, 0};

            for (int m = 0; m < lastLayerNeurons; ++m) {
                weightLocation[1] = m;

                summation += layersDeltas[currentLayer + 1].deltas[m][0] * layers[currentLayer + 1]->getNeuronWeight(2, weightLocation);
            }
            
            // get error sensitivity for neuron n
            location[0] = n;

            layersDeltas[currentLayer].deltas[n][0] = layers[currentLayer]->getErrorSensitivity(1, location, summation);
        }

        lastLayerNeurons = outputNeurons;
    }

    // compute other 2D layers deltas
    bool lastPyramidalLayer = true;
    int lastLayerSize[2];
    int outputSize[2];
    for (currentLayer; currentLayer > 0; --currentLayer) {

        if (layers[currentLayer]->getLayerType() != IPyraNetLayer<NetType>::Layer2D)
            break;

        if (lastPyramidalLayer) {
            
            // compute deltas as if it were a 1D layer, but rearrange deltas as a 2D gris
            // because this is the last pyramida layer (2D) before 1D layers
            layers[currentLayer]->getSize(outputSize);

            for (int u = 0; u < outputSize[0]; ++u) {

                location[0] = u;

                for (int v = 0; v < outputSize[1]; ++v) {

                    // compute the inner summation of (16) in the paper
                    NetType summation = 0;

                    //parentNeuronIndex = (m_v * parentSize[1]) + m_u;
                    int n = (v * outputSize[1]) + u;//u * outputSize[0] + v;  // TODO: check? (17)
                    int weightLocation[2] = {n, 0};

                    for (int m = 0; m < lastLayerNeurons; ++m) {
                        weightLocation[1] = m;

                        summation += layersDeltas[currentLayer + 1].deltas[m][0] * layers[currentLayer + 1]->getNeuronWeight(2, weightLocation);
                    }
            
                    // get error sensitivity for neuron u,v
                    location[1] = v;

                    layersDeltas[currentLayer].deltas[u][v] = layers[currentLayer]->getErrorSensitivity(2, location, summation);
                }
            }

            lastLayerSize[0] = outputSize[0];
            lastLayerSize[1] = outputSize[1];

            lastPyramidalLayer = false;
            continue;
        }

        // other 2D pyramidal layer
        layers[currentLayer]->getSize(outputSize);

        IPyraNet2DLayer<NetType>* nextLayer2D = (IPyraNet2DLayer<NetType>*)layers[currentLayer + 1];

        int receptive = nextLayer2D->getReceptiveFieldSize();
        int overlap = nextLayer2D->getOverlap();
        NetType gap = receptive - overlap;

        // iLow, iHight, jLow, jHigh
        int iLow = 0, iHigh = 0, jLow = 0, jHigh = 0;

        for (int u = 0; u < outputSize[0]; ++u) {
            location[0] = u;

            for (int v = 0; v < outputSize[1]; ++v) {

                // compute the inner summation of (18) in the paper
                NetType summation = 0;

                // compute bounds as in (19) and (20)
                iLow = ceil((u - receptive) / gap) + 1;
                iHigh = floor((u /*- 1*/) / gap);
                jLow = ceil((v - receptive) / gap) + 1;
                jHigh = floor((v /* - 1*/) / gap);

                for (int i = iLow; i < iHigh; ++i) {
                    for (int j = jLow; j < jHigh; ++j) {
                        // double summation in (18), delta_i,j
                        summation += layersDeltas[currentLayer + 1].deltas[i][j];
                    }
                }

                // multiply the summation by the weight u,v
                summation *= layers[currentLayer + 1]->getNeuronWeight(2, location);

                // get error sensitivity for neuron u,v
                location[1] = v;

                layersDeltas[currentLayer].deltas[u][v] = layers[currentLayer]->getErrorSensitivity(2, location, summation);
            }
        }

        lastLayerSize[0] = outputSize[0];
        lastLayerSize[1] = outputSize[1];
    }
}

template <class NetType>
void IPyraNet<NetType>::computeGradient() {
    
    // this gets called for each sample (image), but we need to cumulate it!

    // compute the error gradient for 1D layers
    int currentLayer = layers.size() - 1;
    int location[2] = {0, 0};

    for (currentLayer; currentLayer > 0; --currentLayer) {

        if (layers[currentLayer]->getLayerType() != IPyraNetLayer<NetType>::Layer1D)
            break;

        //layersDeltas[currentLayer].deltas[n][0] 
    }

    // TODO: compute the error gradient for 2D layers
}

// explicit instantiations
template class IPyraNet<float>;
template class IPyraNet<double>;