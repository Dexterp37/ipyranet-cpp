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

    void appendLayer(IPyraNetLayer<NetType>* newLayer);
    void getOutput(std::vector<NetType>& outputs);

    void destroy();

    // training methods
    void setTrainingEpochs(int epochs);
    int getTrainingEpochs() const;
    void setTrainingTechnique(TrainingTechnique technique);
    //TrainingTechnique getTrainingTechnique() const;

private:
	int trainingEpochs;
    NetType learningRate;
    TrainingTechnique trainingTechnique;

    std::vector<IPyraNetLayer<NetType>*> layers;
};

#endif // _IPyraNet_h_