/* 
 *
 */

#include "IPyraNet2DSourceLayer.h"
#include <opencv2/highgui/highgui.hpp>
#include <assert.h>

template<class OutType>
IPyraNet2DSourceLayer<OutType>::IPyraNet2DSourceLayer() 
    : IPyraNetLayer<OutType>()
{

}

template<class OutType>
IPyraNet2DSourceLayer<OutType>::IPyraNet2DSourceLayer(const std::string& fileName) {
    load(fileName);
}

template<class OutType>
IPyraNet2DSourceLayer<OutType>::~IPyraNet2DSourceLayer() {

}

template<class OutType>
bool IPyraNet2DSourceLayer<OutType>::load(const std::string& fileName) {

    source = cv::imread(fileName);

    if (!source.data)
        return false;

    return true;
}
   
template<class OutType>
OutType IPyraNet2DSourceLayer<OutType>::getNeuronOutput(int dimensions, int* neuronLocation) {
    
    // sanity checks
    assert (dimensions == 2);
    assert (neuronLocation != NULL);
    assert (neuronLocation[0] >= 0 && neuronLocation[1] >= 0);

    return static_cast<OutType>(source.at<unsigned char>(neuronLocation[1], neuronLocation[0]));
}

template<class OutType>
int IPyraNet2DSourceLayer<OutType>::getDimensions() const {
    return 2;
}

template<class OutType>
void IPyraNet2DSourceLayer<OutType>::getSize(int* size) {
    assert(size != NULL);

    size[0] = source.cols;
    size[1] = source.rows;
}


template<class OutType>
void IPyraNet2DSourceLayer<OutType>::saveToXML(pugi::xml_node& node) {

    // save the size
    pugi::xml_attribute widthAttr = node.append_attribute("width");
    widthAttr.set_value(source.cols);

    pugi::xml_attribute heightAttr = node.append_attribute("height");
    heightAttr.set_value(source.rows);
}

// explicit instantiations
template class IPyraNet2DSourceLayer<float>;
template class IPyraNet2DSourceLayer<double>;