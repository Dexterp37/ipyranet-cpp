/*

*/

#ifndef _IPyranetLayer_h_
#define _IPyranetLayer_h_

template<typename OutType>
class IPyraNetLayer {
public:
    IPyraNetLayer() : parentLayer(0) { };
    virtual ~IPyraNetLayer() { parentLayer = 0; };

    virtual OutType getNeuronOutput(int dimensions, int* neuronLocation) = 0;

    void setParentLayer(IPyraNetLayer<OutType>* parent) { parentLayer = parent; }
    IPyraNetLayer<OutType>* getParentLayer() { return parentLayer; }

private:
    // previous (adjacent bigger) layer in the pyramid
    IPyraNetLayer<OutType>* parentLayer;
};

#endif // _IPyranetLayer_h_