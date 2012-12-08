/**
 * ipyranet-cpp
 * 
 * Copyright (C) 2012 Alessio Placitelli
 *
 * Permission is hereby granted, free of charge, to any person 
 * obtaining a copy of this software and associated documentation 
 * files (the "Software"), to deal in the Software without 
 * restriction, including without limitation the rights to use, 
 * copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be 
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef _IPyraNet2DSourceLayer_h_
#define _IPyraNet2DSourceLayer_h_

#include "IPyraNetLayer.h"
#include <opencv2/core/core.hpp>
#include <string>

template<class OutType>
class IPyraNet2DSourceLayer : public IPyraNetLayer<OutType> {
public:
    IPyraNet2DSourceLayer();
    IPyraNet2DSourceLayer(const std::string& fileName);
    IPyraNet2DSourceLayer(int initialWidth, int initialHeight);
    virtual ~IPyraNet2DSourceLayer();
    
    LayerType getLayerType() const { return Source; }

    bool load(const std::string& fileName);
    bool load(cv::Mat& sourceImage);
    
    OutType getErrorSensitivity(int dimensions, int* neuronLocation, OutType multiplier) { return 0; };
    OutType getNeuronOutput(int dimensions, int* neuronLocation);    
    int getDimensions() const;
    void getSize(int* size);
    OutType getNeuronWeight(int dimensions, int* neuronLocation) { return 0; }
    void setNeuronWeight(int dimensions, int* neuronLocation, OutType value) { };
    OutType getNeuronBias(int dimensions, int* neuronLocation) { return 0; }
    void setNeuronBias(int dimensions, int* neuronLocation, OutType value) { };

    void saveToXML(pugi::xml_node& node);
    void loadFromXML(pugi::xml_node& node);

    void setPreprocessingEnabled(bool b);
    bool getPreprocessingEnabled() const;

	void setGaborEnabled(bool b);
	bool getGaborEnabled() const;

private:
    cv::Mat source;
    bool preprocessingEnabled;
	bool gaborEnabled;
	
	// gabor kernel settings
	double gaborSigma;
	double gaborTheta;
	double gaborLambda;
	double gaborGamma;
	int gaborKernelSize;

    static cv::Mat gaborKernel;

    void preprocessImage(const cv::Mat& source, cv::Mat& dest);
};

#endif // _IPyraNet2DSourceLayer_h_