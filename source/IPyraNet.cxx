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
#include <iostream>
#include "../3rdParties/dirent-1.12.1/dirent.h" // this is NOT cross platform!
#include <opencv2/highgui/highgui.hpp>

template <class NetType>
IPyraNet<NetType>::IPyraNet() 
    : trainingEpochs(0),
    learningRate(0.0),
    wantedError(0.0),
    trainingTechnique(Unknown),
    batchMode(false),
    verboseOutput(false)
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

    // dump some useful info, just for reference

    pugi::xml_node infoNode = doc.append_child("trainInfo");
    infoNode.append_attribute("learningRate").set_value(getLearningRate());
    infoNode.append_attribute("epochs").set_value(getTrainingEpochs());

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
bool IPyraNet<NetType>::saveOutputToXML(const std::string& fileName) {
    
    pugi::xml_document doc;

    // dump some useful info, just for reference
    size_t num = layers.size();
    int layerSize[2];
    int dims = 0;

    for (size_t k = 0; k < num; ++k) {

        IPyraNetLayer<NetType>* layer = layers[k];
        layer->getSize(layerSize);
        dims = layer->getDimensions();
        
        pugi::xml_node node = doc.append_child("layer");
        node.append_attribute("type").set_value(layer->getLayerType());
        node.append_attribute("dims").set_value(dims);

        // save, raster style (row by row)
        for (size_t v = 0; v < layerSize[1]; ++v) {
            for (size_t u = 0; u < layerSize[0]; ++u) {
                int neuronLoc[2] = {u, v};
                NetType output = layer->getNeuronOutput(dims, neuronLoc);

                pugi::xml_node nodeOut = node.append_child("output");
                nodeOut.append_attribute("u").set_value(u);
                nodeOut.append_attribute("v").set_value(v);
                nodeOut.append_attribute("out").set_value(output);
            }
        }
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

template <class NetType>
void IPyraNet<NetType>::train(const std::string& path) {
    
    // build an array with data and the desired output vector
    struct Sample {
        std::string filePath;
        NetType desired[2];
        cv::Mat image;
        bool preprocessed;
    };

    // load all samples in the heap, just once.
    std::vector<Sample>* samples = new std::vector<Sample>();

    std::string facePath(path);
    facePath.append("/face");

    std::string nonFacePath(path);
    nonFacePath.append("/non-face");

    struct dirent *ent;

    // fill the array with the "faces"
    std::cout << "Populating samples database...";

    DIR* faceDir = opendir (facePath.c_str());
    if (faceDir == NULL) {
        delete samples;
        return;
    }

    while ((ent = readdir (faceDir)) != NULL) {

        // skip "." and ".."
        if (ent->d_name[0] == '.')
            continue;

        // extract the full filename and fill the data structure
        Sample sample;
        sample.desired[0] = 1.0;
        sample.desired[1] = 0.0;
        sample.filePath = facePath;
        sample.filePath.append("/");
        sample.filePath.append(ent->d_name);

        // this should not be here, I know, but we need it to make the
        sample.image = cv::imread(sample.filePath, CV_LOAD_IMAGE_GRAYSCALE);
        sample.preprocessed = false;

        // push into the array
        samples->push_back(sample);
    }

    closedir (faceDir);    
    
    // fill the array with the "NON faces"
    DIR* nonFaceDir = opendir (nonFacePath.c_str());
    if (nonFaceDir == NULL) {
        delete samples;
        return;
    }

    while ((ent = readdir (nonFaceDir)) != NULL) {

        // skip "." and ".."
        if (ent->d_name[0] == '.')
            continue;

        // extract the full filename and fill the data structure
        Sample sample;
        sample.desired[0] = 0.0;
        sample.desired[1] = 1.0;
        sample.filePath = nonFacePath;
        sample.filePath.append("/");
        sample.filePath.append(ent->d_name);

        // this should not be here, I know, but we need it to make the
        sample.image = cv::imread(sample.filePath, CV_LOAD_IMAGE_GRAYSCALE);
        sample.preprocessed = false;

        // push into the array
        samples->push_back(sample);
    }

    closedir (nonFaceDir);

    std::cout << "\t DONE" << std::endl;

    // initialize storage memory
    initDeltaStorage();
    initGradientStorage();

    size_t numSamples = samples->size();

    // now train the neural network for multiple epochs
    IPyraNet2DSourceLayer<NetType>* sourceLayer = ((IPyraNet2DSourceLayer<NetType>*)layers[0]);
    NetType errorCE = 0.0;

    for (int epoch = 0; epoch < trainingEpochs; ++epoch) {

        // reset the error counter for this epoch
        errorCE = 0.0;

        // shuffle samples
        std::cout << "Shuffling samples...";
        std::random_shuffle(samples->begin(), samples->end());
        std::cout << "\tDONE" << std::endl;

        // train
        for (size_t index = 0; index < numSamples; ++index) {

            // if we didn't preprocess the image, do it now
            sourceLayer->setPreprocessingEnabled(samples->at(index).preprocessed == false);

            // process this image and compute the output
            if (!sourceLayer->load(samples->at(index).image)) {
                std::cout << "ERROR!" << std::endl;
                continue;
            }

            samples->at(index).preprocessed = true;

            // compute network output
            std::vector<NetType> outputs;
            getOutput(outputs);

            // compute the error signal
            std::vector<NetType> errorSignal(outputs.size());
            computeErrorSignal(outputs, samples->at(index).desired, errorSignal);

            // compute the error function (10) and accumulate it for each image
            errorCE += computeCrossEntropyError(outputs, samples->at(index).desired);

            // run the backpropagation algorithm
            backpropagation(errorSignal);

            // ok, we are in online mode, so update
            if (!batchMode) {
                updateWeightsAndBiases();
                resetGradient();
            }

            if (verboseOutput) {
                std::cout << " D [" << samples->at(index).desired[0] << " | " << samples->at(index).desired[1] << "]";
                std::cout << " OUT [" << outputs[0] << " | " << outputs[1] << "] ";
                std::cout << " Err [" << errorSignal[0] << " | " << errorSignal[1] << "]" << std::endl;
            }
        }

        if (batchMode) {
            updateWeightsAndBiases();
            resetGradient();
        }

        // (10) requires the minus sign, so swap signs :)
        errorCE = -errorCE;
        std::cout << "Value of the error function for epoch " << epoch << " is " << errorCE << std::endl;

        if (errorCE < wantedError) {
            // error is under our wanted threshold. Exit from the loop
            std::cout << "Training stopped. Error is " << errorCE << " and the wanted error is " << wantedError << std::endl;
            delete samples;
            return;
        }
    }

    // free samples memory
    delete samples;
}

template <class NetType>
void IPyraNet<NetType>::test(const std::string& path) {
    
    // build an array with data and the desired output vector
    struct Sample {
        std::string filePath;
        NetType desired[2];
    };

    std::vector<Sample> samples;

    std::string facePath(path);
    facePath.append("/face");

    std::string nonFacePath(path);
    nonFacePath.append("/non-face");

    struct dirent *ent;

    // fill the array with the "faces"
    std::cout << "Populating test database...";

    DIR* faceDir = opendir (facePath.c_str());
    if (faceDir == NULL) 
        return;

    int faces = 0;
    int nonFaces = 0;

    while ((ent = readdir (faceDir)) != NULL) {

        // skip "." and ".."
        if (ent->d_name[0] == '.')
            continue;

        // extract the full filename and fill the data structure
        Sample sample;
        sample.desired[0] = 1.0;
        sample.desired[1] = 0.0;
        sample.filePath = facePath;
        sample.filePath.append("/");
        sample.filePath.append(ent->d_name);

        // push into the array
        samples.push_back(sample);

        ++faces;
    }

    closedir (faceDir);    
    
    // fill the array with the "NON faces"
    DIR* nonFaceDir = opendir (nonFacePath.c_str());
    if (nonFaceDir == NULL) 
        return;

    while ((ent = readdir (nonFaceDir)) != NULL) {

        // skip "." and ".."
        if (ent->d_name[0] == '.')
            continue;

        // extract the full filename and fill the data structure
        Sample sample;
        sample.desired[0] = 0.0;
        sample.desired[1] = 1.0;
        sample.filePath = nonFacePath;
        sample.filePath.append("/");
        sample.filePath.append(ent->d_name);

        // push into the array
        samples.push_back(sample);

        ++nonFaces;
    }

    closedir (nonFaceDir);

    std::cout << "\t DONE" << std::endl;

    size_t numSamples = samples.size();
    int faceConfusionMatrix[2][2] = {0, 0, 0, 0};
    NetType errorCE = 0.0;

    IPyraNet2DSourceLayer<NetType>* sourceLayer = ((IPyraNet2DSourceLayer<NetType>*)layers[0]);

    // test
    for (size_t index = 0; index < numSamples; ++index) {

        // get the sample
        Sample sample = samples[index];

        // process this image and compute the output
        if (!sourceLayer->load(sample.filePath.c_str())) {
            std::cout << "ERROR!" << std::endl;
            continue;
        }

        // compute network output
        std::vector<NetType> outputs;
        getOutput(outputs);

        // compute the error function (10) and accumulate it for each image
        errorCE += computeCrossEntropyError(outputs, sample.desired);

        bool classifiedAsFace = outputs[0] > outputs[1];
        bool desiredFace = sample.desired[0] > sample.desired[1]; 
        bool correctlyClassified = (classifiedAsFace && desiredFace) || (!classifiedAsFace && !desiredFace);

        // save some stats
        int inputIndex = desiredFace ? 0 : 1;
        int outputIndex = classifiedAsFace ? 0 : 1;
        faceConfusionMatrix[inputIndex][outputIndex] += 1;
    }

    // requires the minus sign
    errorCE = -errorCE;

    // print the stats
    std::cout << "Test database has " << faces << " faces and " << nonFaces << " non faces" << std::endl;
    std::cout << std::endl << "Confusion matrix" << std::endl;
    std::cout << "\tF\t\tNF" << std::endl;
    std::cout << "F\t" << faceConfusionMatrix[0][0] << "\t\t" << faceConfusionMatrix[1][0] << std::endl;
    std::cout << "NF\t" << faceConfusionMatrix[0][1] << "\t\t" << faceConfusionMatrix[1][1] << std::endl;

    double percCorrectFaces = (faceConfusionMatrix[0][0] / (double)faces);
    double percCorrectNonFaces = (faceConfusionMatrix[1][1] / (double)nonFaces);
    std::cout << std::endl << "Classification stats" << std::endl;
    std::cout << "Correctly classified faces:\t" << percCorrectFaces << "%" << std::endl;
    std::cout << "Correctly classified non faces:\t" << percCorrectNonFaces << "%" << std::endl;
    std::cout << "Averge:\t" << ((percCorrectFaces + percCorrectNonFaces) * 0.5) << "%" << std::endl;
    std::cout << std::endl << "Cross Entropy Error:\t" << errorCE << std::endl;
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
void IPyraNet<NetType>::setLearningRate(NetType rate) {
    learningRate = rate;
}

template <class NetType>
NetType IPyraNet<NetType>::getLearningRate() const {
    return learningRate;
}

template <class NetType>
void IPyraNet<NetType>::setBatchMode(bool batch) {
    batchMode = batch;
}

template <class NetType>
bool IPyraNet<NetType>::getBatchMode() const {
    return batchMode;
}

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

                for (int v = 0; v < neurons; ++v) {
                    layersGradient[l].weightsGrad[u][v] = 0.0;
                }
            }

            // now enough room for the biases    
            layersGradient[l].biasesGrad.resize(neurons);

            for (int u = 0; u < neurons; ++u) {
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
                    int n = (v * outputSize[0]) + u;//u * outputSize[0] + v;  // TODO: check? (17)
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

        // other 2D pyramidal layer (18)
        layers[currentLayer]->getSize(outputSize);

        IPyraNet2DLayer<NetType>* nextLayer2D = (IPyraNet2DLayer<NetType>*)layers[currentLayer + 1];

        int receptive = nextLayer2D->getReceptiveFieldSize();
        int overlap = nextLayer2D->getOverlap();
        NetType gap = receptive - overlap;

        // iLow, iHight, jLow, jHigh
        int iLow = 0, iHigh = 0, jLow = 0, jHigh = 0;

        // indices in (18) go from 1 to n
        for (int u = 1; u <= outputSize[0]; ++u) {
            location[0] = u - 1;

            for (int v = 1; v <= outputSize[1]; ++v) {
                location[1] = v - 1;

                // compute the inner summation of (18) in the paper
                NetType summation = 0;

                // compute bounds as in (19) and (20)
                iLow = ceil((u - receptive) / gap);// + 1; (maybe the +1 offset is wrong here?)
                iHigh = floor((u - 1) / gap) + 1;
                jLow = ceil((v - receptive) / gap);// + 1;
                jHigh = floor((v - 1) / gap) + 1;

                int ijMinusOne[2];

                for (int i = iLow; i <= iHigh; ++i) {
                    
                    ijMinusOne[0] = i - 1;

                    for (int j = jLow; j <= jHigh; ++j) {
                    
                        ijMinusOne[1] = j - 1;

                        // double summation in (18), delta_i,j
                        
                        if (ijMinusOne[0] < 0 || ijMinusOne[1] < 0)
                            continue;

                        if (ijMinusOne[0] >= layersDeltas[currentLayer + 1].deltas.size() ||
                            ijMinusOne[1] >= layersDeltas[currentLayer + 1].deltas[ijMinusOne[0]].size())
                            continue;

                        summation += layersDeltas[currentLayer + 1].deltas[ijMinusOne[0]][ijMinusOne[1]];
                    }
                }

                // multiply the summation by the weight u,v
                summation *= layers[currentLayer + 1]->getNeuronWeight(2, location);

                // get error sensitivity for neuron u,v (consider indices offset!)
                layersDeltas[currentLayer].deltas[location[0]][location[1]] = layers[currentLayer]->getErrorSensitivity(2, location, summation);
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
    
    // get parent size
    int parentDims = 0;
    int parentSize[2];

    for (currentLayer; currentLayer > 0; --currentLayer) {

        if (layers[currentLayer]->getLayerType() != IPyraNetLayer<NetType>::Layer1D)
            break;
        
        // parent layer pointer
        IPyraNetLayer<NetType>* parent = layers[currentLayer]->getParentLayer();
        parentDims = parent->getDimensions();
        parent->getSize(parentSize);

        // current layer size
        int neurons = 0;
        layers[currentLayer]->getSize(&neurons);

        // compute the biases gradient
        for (int n = 0; n < neurons; ++n)
            layersGradient[currentLayer].biasesGrad[n][0] += layersDeltas[currentLayer].deltas[n][0];

        // compute the weights gradient
        if (parentDims == 2) {

            // 2D connecting to 1D
            int parentNeuronLoc[2]; 
            int m = 0; // as defined in (21)

            for (int m_u = 0; m_u < parentSize[0]; ++m_u) {

                parentNeuronLoc[0] = m_u;

                for (int m_v = 0; m_v < parentSize[1]; ++m_v) {
                    parentNeuronLoc[1] = m_v;

                    NetType parentNeuronOutput = parent->getNeuronOutput(2, parentNeuronLoc);

                    m = (m_v * parentSize[0]) + m_u;

                    // update the gradient
                    for (int n = 0; n < neurons; ++n)
                        layersGradient[currentLayer].weightsGrad[m][n] += layersDeltas[currentLayer].deltas[n][0] * parentNeuronOutput;

                }
            }

        } else if (parentDims == 1) {

            // 1D connecting to 1D
            for (int m = 0; m < parentSize[0]; ++m) {
                NetType parentNeuronOutput = parent->getNeuronOutput(2, &m);

                // update the gradient
                for (int n = 0; n < neurons; ++n)
                    layersGradient[currentLayer].weightsGrad[m][n] += layersDeltas[currentLayer].deltas[n][0] * parentNeuronOutput;

            }
        }
    }

    // compute the error gradient for 2D layers
    int receptive = 0;
    int overlap = 0;
    NetType gap = 0;

    // uLow, uHight, vLow, vHigh
    int uLow = 0, uHigh = 0, vLow = 0, vHigh = 0;

    for (currentLayer; currentLayer > 0; --currentLayer) {

        if (layers[currentLayer]->getLayerType() != IPyraNetLayer<NetType>::Layer2D)
            break;

        // current layer size
        int currentSize[2];
        layers[currentLayer]->getSize(currentSize);

        // compute the biases gradient
        for (int u = 0; u < currentSize[0]; ++u)
            for (int v = 0; v < currentSize[1]; ++v)
                layersGradient[currentLayer].biasesGrad[u][v] += layersDeltas[currentLayer].deltas[u][v];

        // parent layer pointer
        IPyraNetLayer<NetType>* parent = layers[currentLayer]->getParentLayer();
        parentDims = parent->getDimensions();
        parent->getSize(parentSize);

        // cast to a layer 2d
        IPyraNet2DLayer<NetType>* layer2D = (IPyraNet2DLayer<NetType>*)layers[currentLayer];

        // get layer informations
        receptive = layer2D->getReceptiveFieldSize();
        overlap = layer2D->getOverlap();
        gap = receptive - overlap;
            
        int parentNeuronLoc[2]; 

        // compute weights gradient
        for (int i = 1; i <= parentSize[0]; ++i) {
            parentNeuronLoc[0] = i - 1;

            for (int j = 1; j <= parentSize[1]; ++j) {
                parentNeuronLoc[1] = j - 1;    
                
                NetType parentNeuronOutput = parent->getNeuronOutput(2, parentNeuronLoc);
                
                // compute uLow-uHigh and vLow-vHigh (23)
                uLow = ceil((i - receptive) / gap);// + 1; // maybe the +1 is wrong?
                uHigh = floor((i - 1) / gap) + 1;
                vLow = ceil((j - receptive) / gap);// + 1;
                vHigh = floor((j - 1) / gap) + 1;
                
                NetType summation = 0;
                
                int uvMinusOne[2];

                for (int u = uLow; u <= uHigh; ++u) {
                    uvMinusOne[0] = u - 1;

                    for (int v = vLow; v <= vHigh; ++v) {
                        uvMinusOne[1] = v - 1;

                        if (uvMinusOne[0] < 0 || uvMinusOne[1] < 0)
                            continue;

                        if (uvMinusOne[0] >= layersDeltas[currentLayer].deltas.size() ||
                            uvMinusOne[1] >= layersDeltas[currentLayer].deltas[uvMinusOne[0]].size())
                            continue;

                        summation += layersDeltas[currentLayer].deltas[uvMinusOne[0]][uvMinusOne[1]];
                    }
                }

                layersGradient[currentLayer].weightsGrad[parentNeuronLoc[0]][parentNeuronLoc[1]] += summation * parentNeuronOutput;
            }
        }
    }
}

template <class NetType>
void IPyraNet<NetType>::resetGradient() {

    size_t gradientsNum = layersGradient.size();

    // zero out each gradient
    for (size_t g = 0; g < gradientsNum; ++g) {

        // zero weights
        size_t u_max = layersGradient[g].weightsGrad.size();
        size_t v_max = 0;

        for (size_t u = 0; u < u_max; ++u) {
            v_max = layersGradient[g].weightsGrad[u].size();

            for (size_t v = 0; v < v_max; ++v) {
                layersGradient[g].weightsGrad[u][v] = 0.0;
            }
        }
    
        // zero biases
        u_max = layersGradient[g].biasesGrad.size();
        v_max = 0;

        for (size_t u = 0; u < u_max; ++u) {
            v_max = layersGradient[g].biasesGrad[u].size();

            for (size_t v = 0; v < v_max; ++v) {
                layersGradient[g].biasesGrad[u][v] = 0.0;
            }
        }
    }

}

template <class NetType>
void IPyraNet<NetType>::computeErrorSignal(const std::vector<NetType>& output, const NetType* desired, std::vector<NetType>& error) {
    
    assert(error.size() == output.size());

    // compute the quote
    NetType expSum = 0.0;
    for (size_t i = 0; i < output.size(); ++i)
        expSum += exp(output[i]);

    // finally compute the error signal
    for (size_t e = 0; e < output.size(); ++e) {
        NetType pn = exp(output[e]) / expSum;
        error[e] = pn - desired[e];
    }
}

template <class NetType>
NetType IPyraNet<NetType>::computeCrossEntropyError(const std::vector<NetType>& output, const NetType* desired) {

    // compute the quote to estimate each pn
    NetType expSum = 0.0;
    for (size_t i = 0; i < output.size(); ++i)
        expSum += exp(output[i]);

    // now compute each pn and finally the error
    NetType errorCE = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        NetType pn = exp(output[i]) / expSum;
        errorCE += desired[i] * log(pn);
    }

    return errorCE;
}

template <class NetType>
void IPyraNet<NetType>::updateWeightsAndBiases() {
    
    // get the size of source layer
    int parentSize[2];
    int parentDims = 2;
    int location[2];

    layers[0]->getSize(parentSize);

    for (unsigned int l = 1; l < layers.size(); ++l) {

        // 2D layer
        if (layers[l]->getDimensions() == 2) {

            // we have as many weights as the number of neurons in last layer.
            for (int u = 0; u < parentSize[0]; ++u) {
                location[0] = u;

                for (int v = 0; v < parentSize[1]; ++v) {
                    location[1] = v;

                    NetType oldWeight = layers[l]->getNeuronWeight(2, location);

                    // compute the new weight as OLD - LearningRate * Gradient
                    NetType newWeight = oldWeight - learningRate * layersGradient[l].weightsGrad[u][v];

                    layers[l]->setNeuronWeight(2, location, newWeight);
                }
            }

            // now update the biases. A bias for each neuron of current layer.
            // save it in "parentSize", so that it is already updated.
            layers[l]->getSize(parentSize);

            for (int u = 0; u < parentSize[0]; ++u) {
                location[0] = u;

                for (int v = 0; v < parentSize[1]; ++v) {
                    location[1] = v;

                    NetType oldBias = layers[l]->getNeuronBias(2, location);

                    // compute the new weight as OLD - LearningRate * Gradient
                    NetType newBias = oldBias - learningRate * layersGradient[l].biasesGrad[u][v];

                    layers[l]->setNeuronBias(2, location, newBias);
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

            // we have a weight for each connection going from
            // each input to every neuron in this layer.
            for (int u = 0; u < inputNeurons; ++u) {
                location[0] = u;

                for (int v = 0; v < neurons; ++v) {
                    location[1] = v;

                    NetType oldWeight = layers[l]->getNeuronWeight(2, location);

                    // compute the new bias as OLD - LearningRate * Gradient
                    NetType newWeight = oldWeight - learningRate * layersGradient[l].weightsGrad[u][v];

                    layers[l]->setNeuronWeight(2, location, newWeight);
                }
            }

            // now update the biases
            for (int u = 0; u < neurons; ++u) {

                location[0] = u;

                NetType oldBias = layers[l]->getNeuronBias(1, location);

                // compute the new bias as OLD - LearningRate * Gradient
                NetType newBias = oldBias - learningRate * layersGradient[l].biasesGrad[u][0];

                layers[l]->setNeuronBias(1, location, newBias);
            }
        }

        // update parent dims
        layers[l]->getSize(parentSize);
        parentDims = layers[l]->getDimensions();
    }
}

// explicit instantiations
template class IPyraNet<float>;
template class IPyraNet<double>;