/* 
 *
 */

#include "IPyraNet2DSourceLayer.h"
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>

IPyraNet2DSourceLayer::IPyraNet2DSourceLayer() 
    : IPyraNet2DLayer()
{

}

IPyraNet2DSourceLayer::IPyraNet2DSourceLayer(const std::string& fileName) {
    load(fileName);
}

IPyraNet2DSourceLayer::~IPyraNet2DSourceLayer() {

}

bool IPyraNet2DSourceLayer::load(const std::string& fileName) {

    source = cv::imread(fileName);

    if (!source.data)
        return false;

    setLayerSize(source.cols, source.rows);

    return true;
}