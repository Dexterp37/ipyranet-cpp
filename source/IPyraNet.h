/* 
 *
 */

#ifndef _IPyraNet_h_
#define _IPyraNet_h_

#include <vector>

// forward declarations
class IPyraNet2DLayer;

class IPyraNet {
public:
    IPyraNet();
    ~IPyraNet();

    void appendLayer(IPyraNet2DLayer* newLayer);

    void destroy();

private:
    std::vector<IPyraNet2DLayer*> layers2D;
    //std::vector<IPyraNet1DLayer*> layers1D;
};

#endif // _IPyraNet_h_