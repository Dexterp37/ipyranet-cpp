/* 
 *
 */

#include "IPyraNet2DSourceLayer.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <assert.h>

template<class OutType>
IPyraNet2DSourceLayer<OutType>::IPyraNet2DSourceLayer() 
    : IPyraNetLayer<OutType>(),
    preprocessingEnabled(true),
	gaborEnabled(false),

    // taken from I-Pyranet paper
    gaborSigma(4.0),		// gaussian standard deviation
    gaborTheta(CV_PI/3),	// orientation, ~60°
    gaborLambda(1.0/8.0),	// wavelength (central frequency f=8)
    gaborGamma(1.0),		// aspect ratio
    gaborKernelSize(15)		// this was not in the paper
{

}

template<class OutType>
IPyraNet2DSourceLayer<OutType>::IPyraNet2DSourceLayer(const std::string& fileName)  
    : IPyraNetLayer<OutType>(),
    preprocessingEnabled(true),
	gaborEnabled(false),

    // taken from I-Pyranet paper
    gaborSigma(4.0),		// gaussian standard deviation
    gaborTheta(CV_PI/3),	// orientation, ~60°
    gaborLambda(1.0/8.0),	// wavelength (central frequency f=8)
    gaborGamma(1.0),		// aspect ratio
    gaborKernelSize(15)		// this was not in the paper
{
    load(fileName);
}

template<class OutType>
IPyraNet2DSourceLayer<OutType>::IPyraNet2DSourceLayer(int initialWidth, int initialHeight)  
    : IPyraNetLayer<OutType>(),
    preprocessingEnabled(true),
	gaborEnabled(false),

    // taken from I-Pyranet paper
    gaborSigma(4.0),		// gaussian standard deviation
    gaborTheta(CV_PI/3),	// orientation, ~60°
    gaborLambda(1.0/8.0),	// wavelength (central frequency f=8)
    gaborGamma(1.0),		// aspect ratio
    gaborKernelSize(15)		// this was not in the paper
{
    source.create(initialHeight, initialWidth, CV_8U);
}

template<class OutType>
IPyraNet2DSourceLayer<OutType>::~IPyraNet2DSourceLayer() {

}

template<class OutType>
bool IPyraNet2DSourceLayer<OutType>::load(const std::string& fileName) {

    cv::Mat original = cv::imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

    if (preprocessingEnabled)
        preprocessImage(original, source);
    else 
        original.convertTo(source, CV_64F, 2.0 / 255.0, -1.0); // just convert to the -1 +1 range

    if (!source.data)
        return false;

    return true;
}

template<class OutType>
bool IPyraNet2DSourceLayer<OutType>::load(cv::Mat& sourceImage) {

    if (preprocessingEnabled)
        preprocessImage(sourceImage, sourceImage);

    source = sourceImage;

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
    node.append_attribute("width").set_value(source.cols);
    node.append_attribute("height").set_value(source.rows);

	node.append_attribute("preprocess").set_value(preprocessingEnabled);
	node.append_attribute("gabor").set_value(gaborEnabled);

	node.append_attribute("sigma").set_value(gaborSigma);
	node.append_attribute("theta").set_value(gaborTheta);
	node.append_attribute("lambda").set_value(gaborLambda);
	node.append_attribute("gamma").set_value(gaborGamma);
	node.append_attribute("ksize").set_value(gaborKernelSize);
}

template<class OutType>
void IPyraNet2DSourceLayer<OutType>::loadFromXML(pugi::xml_node& node) {
    
    int initialWidth = node.attribute("width").as_int();
    int initialHeight = node.attribute("height").as_int();

    source.create(initialHeight, initialWidth, CV_8U);

	preprocessingEnabled = node.attribute("preprocess").as_bool();
	gaborEnabled = node.attribute("gabor").as_bool();

	gaborSigma = node.attribute("sigma").as_double();
	gaborTheta = node.attribute("theta").as_double();
	gaborLambda = node.attribute("lambda").as_double();
	gaborGamma = node.attribute("gamma").as_double();
	gaborKernelSize = node.attribute("ksize").as_int();
}

template<class OutType>
void IPyraNet2DSourceLayer<OutType>::setPreprocessingEnabled(bool b) {
    preprocessingEnabled = b;
}

template<class OutType>
bool IPyraNet2DSourceLayer<OutType>::getPreprocessingEnabled() const {
    return preprocessingEnabled;
}

template<class OutType>
void IPyraNet2DSourceLayer<OutType>::setGaborEnabled(bool b) {
    gaborEnabled = b;
}

template<class OutType>
bool IPyraNet2DSourceLayer<OutType>::getGaborEnabled() const {
    return gaborEnabled;
}

template<class OutType>
void IPyraNet2DSourceLayer<OutType>::preprocessImage(const cv::Mat& src, cv::Mat& dest) {
   
    // initialize the gabor filter (just once)
    if (gaborKernel.cols == 0 || gaborKernel.rows == 0) {
        gaborKernel = cv::getGaborKernel(cv::Size(gaborKernelSize, gaborKernelSize) , gaborSigma, 
			gaborTheta, gaborLambda, gaborGamma);
    }
     
    // apply Histogram Equalization
    cv::equalizeHist(src, dest);

    // convert the image from 0-255 to [-1.0 +1.0] and apply the gabor filter
    cv::Mat scaledTo1;
    dest.convertTo(scaledTo1, CV_64F, 2.0 / 255.0, -1.0); // (maxVal - minVal) / 255.0, minVal);

	if (gaborEnabled) {
		cv::Mat gaboredData;
		cv::filter2D(scaledTo1, dest, CV_64F, gaborKernel);
	} else {
		dest = scaledTo1;
	}
}

// explicit instantiations
template class IPyraNet2DSourceLayer<float>;
template class IPyraNet2DSourceLayer<double>;

cv::Mat IPyraNet2DSourceLayer<float>::gaborKernel;
cv::Mat IPyraNet2DSourceLayer<double>::gaborKernel;
