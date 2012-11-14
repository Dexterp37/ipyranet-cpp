/* 
 *
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

    // layer management
    void appendLayer(IPyraNetLayer<NetType>* newLayer);
    void getOutput(std::vector<NetType>& outputs);

    void destroy();

    // training methods
    void setTrainingEpochs(int epochs);
    int getTrainingEpochs() const;
    void setTrainingTechnique(TrainingTechnique technique);
    //TrainingTechnique getTrainingTechnique() const;

    void train(const std::string&  path);

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
    TrainingTechnique trainingTechnique;

    std::vector<IPyraNetLayer<NetType>*> layers;

    void appendLayerNoInit(IPyraNetLayer<NetType>* newLayer);
    void initDeltaStorage();
    void initGradientStorage();
    void backpropagation(const std::vector<NetType>& errorSignal);
    void computeErrorSensitivities(const std::vector<NetType>& errorSignal);
    void computeGradient();
};

#endif // _IPyraNet_h_