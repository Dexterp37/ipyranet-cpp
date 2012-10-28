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
    IPyraNet();
    ~IPyraNet();

    void appendLayer(IPyraNetLayer<NetType>* newLayer);

    void destroy();

private:
    std::vector<IPyraNetLayer<NetType>*> layers;
};

#endif // _IPyraNet_h_