/**
 * ipyranet-cpp
 * 
 * Copyright (C) 2012 Alessio Placitelli
 *
 * Permission is hereby granted, free of charge, to any person 
 * obtaining a copy of this software and associated documentation 
 * files (the "Software"), to deal in the Software without 
 * restriction, including without limitation the rights to use, 
 * copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _IPyraNet_h_
#define _IPyraNet_h_

#include <vector>

// forward declarations
template<class OutType> class IPyraNetLayer;

template <class NetType>
class IPyraNet {
public:
    enum TrainingTechnique {
        Unknown = -1,
        GradientDescend = 1
    };

    IPyraNet();
    ~IPyraNet();

    // XML serialization
    bool saveToXML(const std::string& fileName);
    bool loadFromXML(const std::string& fileName);
    bool saveOutputToXML(const std::string& fileName);

    // layer management
    void appendLayer(IPyraNetLayer<NetType>* newLayer);
    void getOutput(std::vector<NetType>& outputs);

    void destroy();

    // training methods
    void setTrainingEpochs(int epochs);
    int getTrainingEpochs() const;
    void setTrainingTechnique(TrainingTechnique technique);
    //TrainingTechnique getTrainingTechnique() const;
    void setLearningRate(NetType rate);
    NetType getLearningRate() const;
    void setBatchMode(bool batch);
    bool getBatchMode() const;

    void train(const std::string&  path);
    void test(const std::string&  path);

private:    
    
    struct LayerDeltas {
        std::vector<std::vector<NetType> > deltas;
    };

    struct LayerGradient {
        std::vector<std::vector<NetType> > weightsGrad;
        std::vector<std::vector<NetType> > biasesGrad;
    };

    std::vector<LayerDeltas> layersDeltas;  // deltas storage
    std::vector<LayerGradient> layersGradient;  // gradient storage

	int trainingEpochs;
    NetType learningRate;
    NetType wantedError;
    TrainingTechnique trainingTechnique;
    bool batchMode;

    bool verboseOutput;

    std::vector<IPyraNetLayer<NetType>*> layers;

    void appendLayerNoInit(IPyraNetLayer<NetType>* newLayer);
    void initDeltaStorage();
    void initGradientStorage();
    void backpropagation(const std::vector<NetType>& errorSignal);
    void computeErrorSensitivities(const std::vector<NetType>& errorSignal);
    void computeGradient();
    void resetGradient();
    void computeErrorSignal(const std::vector<NetType>& output, const NetType* desired, std::vector<NetType>& error);
    NetType computeCrossEntropyError(const std::vector<NetType>& output, const NetType* desired);
    void updateWeightsAndBiases();
};

#endif // _IPyraNet_h_