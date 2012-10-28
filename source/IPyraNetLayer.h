/*

*/

#ifndef _IPyranetLayer_h_
#define _IPyranetLayer_h_

template<typename OutType>
class IPyraNetLayer {
public:
    IPyraNetLayer() { };
    virtual ~IPyraNetLayer() { };

    virtual OutType getNeuronOutput(int dimensions, int* neuronLocation) = 0;
};

#endif // _IPyranetLayer_h_