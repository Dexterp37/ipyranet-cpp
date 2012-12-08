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

#include <iostream>
#include <ctime>
#include "IPyraNet.h"
#include "IPyraNet2DSourceLayer.h"
#include "IPyraNet1DLayer.h"
#include "IPyraNet2DLayer.h"
#include "IPyraNetSigmoidFunction.h"

struct CommandLineOptions {
    bool train;
    double learningRate;
    int trainingEpochs;
    std::string xmlInput;
    bool test;
};

void printHelp() {
    std::cout << "IPyraNet.exe <XML File> [help]" << std::endl << std::endl;
    std::cout << "IPyraNet.exe <XML File> [train <learning rate value>] [epochs <number of epochs>] [test]" << std::endl << std::endl;
    std::cout << "IPyraNet.exe <XML File> train 0.015 test" << std::endl << std::endl;
    std::cout << "IPyraNet.exe <XML File> train 0.015" << std::endl << std::endl;
    std::cout << "IPyraNet.exe <XML File>" << std::endl << std::endl;
    std::cout << "IPyraNet.exe train 0.015" << std::endl << std::endl;
}

void parseCommandLine(CommandLineOptions& cmd, int argc, char** argv) {

    // this is just a quick and crappy way to deal with command line options.
	// something better could be written, that's for sure. But I don't have
	// enough time :)
	
    // reset the cmd structure
    cmd.train = false;
    cmd.learningRate = -1.0;
    cmd.trainingEpochs = -1;
    cmd.test = false;

    if (argc == 1) {
        printHelp();
        return;
    }


    bool nextIsLearningRate = false;
    bool nextIsEpochs = false;

    // parse command line arguments (very naive)
    for (int arg = 1; arg < argc; ++arg) {

        char* currentArg = argv[arg];

        if (!strcmp(currentArg, "help")) {
            printHelp();
        } else if (!strcmp(currentArg, "train")) {
            cmd.train = true;
            nextIsLearningRate = true;
            nextIsEpochs = false;
        } else if (nextIsLearningRate) {
            cmd.learningRate = atof(currentArg);
            nextIsLearningRate = false;
            nextIsEpochs = false;
        } else if (!strcmp(currentArg, "epochs")) {
            nextIsEpochs = true;
            nextIsLearningRate = false;
        } else if (nextIsEpochs) {
            cmd.trainingEpochs = atoi(currentArg);
            nextIsEpochs = false;
            nextIsLearningRate = false;
        } else if (!strcmp(currentArg, "test")) {
            cmd.test = true;
            nextIsLearningRate = false;
            nextIsEpochs = false;
        } else
            cmd.xmlInput = currentArg;
    }
}

int main(int argc, char** argv) {

    std::cout << "I-PyraNet Face Detection" << std::endl;
    std::cout << "Author: Alessio Placitelli - Computer Vision" << std::endl << std::endl;

    // initialize random seed
    srand(time(NULL));

    CommandLineOptions cmd;
    parseCommandLine(cmd, argc, argv);


    // IPyraNet initialized as in "A Receptive Field Based Approach For Face Detection" by Fernandes et al.
    // Section V(B)
    IPyraNet<double> ipnn;
    ipnn.setLearningRate(0.015);
    ipnn.setTrainingEpochs(1100);
    ipnn.setBatchMode(true);

    if (cmd.xmlInput.length() > 0) {
        // load from file, if an argument was provided
        std::cout << "Loading the I-PyraNet network from " << cmd.xmlInput << std::endl;
        ipnn.loadFromXML(cmd.xmlInput);
    } else {
        std::cout << "Building the I-PyraNet network" << std::endl;
        ipnn.appendLayer(new IPyraNet2DSourceLayer<double>(19, 19));
        ipnn.appendLayer(new IPyraNet2DLayer<double>(4, 0, 1, new IPyraNetSigmoidFunction<double>()));
        ipnn.appendLayer(new IPyraNet2DLayer<double>(3, 1, 1, new IPyraNetSigmoidFunction<double>()));
        ipnn.appendLayer(new IPyraNet1DLayer<double>(2, new IPyraNetSigmoidFunction<double>()));
        
        // Serialize to XML the initial NN
        ipnn.saveToXML("initial.xml");
    }

    // Train IPyraNet if requested
    if (cmd.train) {

        if (cmd.learningRate > -1.0) {
            ipnn.setLearningRate(cmd.learningRate);
            std::cout << "Setting learning rate to " << cmd.learningRate << std::endl;
        }

        if (cmd.trainingEpochs > -1) {
            ipnn.setTrainingEpochs(cmd.trainingEpochs);
            std::cout << "Setting training epochs to " << cmd.trainingEpochs << std::endl;
        }

        std::cout << "Training the I-PyraNet network" << std::endl;
        ipnn.train("faces/train");

        // Serialize to XML the initial NN
        ipnn.saveToXML("trained.xml");
    }

    // Test the NN
    if (cmd.test) {
        std::cout << "Testing the I-PyraNet network" << std::endl;
        ipnn.test("faces/test");
    }

    return -1;
}