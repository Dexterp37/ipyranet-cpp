ipyranet-cpp
============

This is a C++ implementation of pyramidal neural network for visual pattern recognition or PyraNet. This software also implements the concept of inhibitory field introduced in the relative scientific literature (I-PyraNet). 
For a more comprehensve overview of the aforementioned concepts, please have a look at the papers from the respective authors.

 * "A Pyramidal Neural Network For Visual Pattern Recognition" by SL Phung (http://dx.doi.org/10.1109/TNN.2006.884677)
 * "A Pyramidal Neural Network Based on Nonclassical Receptive Field Inhibition" by B. Fernandes (http://dx.doi.org/10.1109/ICTAI.2008.111)

Libraries
------------
In order to compile the provided code, you must install the following libraries:

 * [OpenCV 2.4.2] (http://opencv.willowgarage.com/wiki/)
 * [dirent 1.12.1] (http://www.softagalleria.net/dirent.php)
 * [pugixml 1.2] (http://code.google.com/p/pugixml/)

Tools
-----
The project was developed using Microsoft Visual Studio 2010. However, thanks to the provided CMakeLists file, it should be possible to generate project files for other IDE as well. In order to do so, you should install [CMake 2.8.9] (http://www.cmake.org/).

Dataset
-------
The provided example software makes use of the [MIT CBCL Face Database #1] (http://www.ai.mit.edu/projects/cbcl.old/) to train the neural network. The experiment tries to distinguish between human faces and non human faces.

Report a bug
------------
If you believe you've found a bug, please report it using the [issue submission form](https://github.com/Dexterp37/ipyranet-cpp/issues). Be sure to include the relevant information so that the bug can be reproduced: the version of the software, compiler version and target architecture, a little example of code that reproduces the bug, etc.

Documentation and contacts
--------------------------
The code is heavily commented so please have a look at the code for further insights on its inner workings. In case you need to contact me, please feel free to do so at a.placitelli AT a2p.it or through GitHub.

Acknowledgments
---------------
This software is based on pugixml library (http://pugixml.org).
pugixml is Copyright (C) 2006-2012 Arseny Kapoulkine.

This software is based on dirent library (http://www.softagalleria.net/dirent.php).
dirent is Copyright (C) 2006-2012 Toni Ronkko