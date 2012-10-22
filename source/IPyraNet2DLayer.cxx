/* 
 *
 */

#include "IPyraNet2DLayer.h"
#include <stdio.h>

IPyraNet2DLayer::IPyraNet2DLayer(int width, int height) 
    : width(width),
    height(height),
    receptiveSize(0),
    overlap(0),
    inhibitorySize(0),
    parentLayer(NULL)
{

}

IPyraNet2DLayer::~IPyraNet2DLayer() {

}