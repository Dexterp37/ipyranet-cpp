/* 
 *
 */

#ifndef _IPyraNet2DSourceLayer_h_
#define _IPyraNet2DSourceLayer_h_

#include "IPyraNet2DLayer.h"
#include <opencv2/core/core.hpp>
#include <string>

class IPyraNet2DSourceLayer : public IPyraNet2DLayer {
public:
    IPyraNet2DSourceLayer();
    IPyraNet2DSourceLayer(const std::string& fileName);
    virtual ~IPyraNet2DSourceLayer();

    bool load(const std::string& fileName);

private:
    cv::Mat source;
};

#endif // _IPyraNet2DSourceLayer_h_