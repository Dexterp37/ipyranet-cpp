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

    cv::Mat original = cv::imread(fileName);
    original.convertTo(source, CV_64F, 1.0 / 255.0);

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

    return static_cast<OutType>(source.at<double>(neuronLocation[1], neuronLocation[0]));
//    return static_cast<OutType>(source.at<unsigned char>(neuronLocation[1], neuronLocation[0]));
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

template<class OutType>
void IPyraNet2DSourceLayer<OutType>::loadFromXML(pugi::xml_node& node) {
    
    int initialWidth = node.attribute("width").as_int();
    int initialHeight = node.attribute("height").as_int();

    source.create(initialHeight, initialWidth, CV_8U);
}

// explicit instantiations
template class IPyraNet2DSourceLayer<float>;
template class IPyraNet2DSourceLayer<double>;