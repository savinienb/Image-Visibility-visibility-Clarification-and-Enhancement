---------------------------------------------------------------------------------------------------------------------------------
README (Version: 1.0)
---------------------------------------------------------------------------------------------------------------------------------
PROJECT NAME  : Variational Image Enhancement and Clarification
PROJECT GROUP : Savinien Bonheur, Thomas Drevet, Omair Khalid ( {sjb4,td11,ok19}@hw.ac.uk )
DESCRIPTION   : This folder contains all the code required to run the DeMisty_RI CNN and DeMisty_FI CNN Models developed
	        as part of the project. Both the CNN Models can be tested for both images and video.
		
		Below is the description of the STRUCTURE of the folder, PREREQUISITES for running the code, and the 
	        HOW TO TEST section, detailing how to perform tests.
---------------------------------------------------------------------------------------------------------------------------------
STRUCTURE: 
---------------------------------------------------------------------------------------------------------------------------------

This folder contains three sub-folders, two python files, and two Caffe models(DeMisty_RI CNN and DeMisty_FI):
	
	1. codes : contains two python files
		
		1. dehaze.py  : employs CNN models to dehaze the image
		2. haze.py    : applies haze to the image

	2. data : contains the images

		1. img        : contains images to be tested
		2. result     : stores the dehazed images

	3. template  : contains models of DeMistify_Fl and DeMistify_Rl CNN model

	4. DeMistify_Fl.caffemodel : Caffe model of DeMistify_Fl

	5. DeMistify_Rl.caffemodel : Caffe model of DeMistify_Rl

	6. demo_im.py  : runs both dehazing procedures on the images present in the img folder

	7. demo_vid.py :  runs the dehazing code using the computer webcam

---------------------------------------------------------------------------------------------------------------------------------
PREREQUISITES (with recommended versions): 
---------------------------------------------------------------------------------------------------------------------------------
The user is requested to install the following libraries to be able to run the code. 
Our implementation has been tested with:

	1. Ubuntu 16.04
		
	2. Python 2.7 (https://www.python.org/downloads/)
		
		Libraries: numpy, scikit, copy, time

	3. OpenCV 2.4.9.1 (https://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/)

	4. Cuda 8.0.61 (https://developer.nvidia.com/cuda-80-ga2-download-archive)

	5. Caffe 1.0.0 (https://github.com/BVLC/caffe/releases)

---------------------------------------------------------------------------------------------------------------------------------
HOW TO TEST:
---------------------------------------------------------------------------------------------------------------------------------

	1. To run the image code :

		1. Place images into "./data/images" (a set is already provided) 

		2. Run demo_im.py

	2. To run the video code :

		1. Run demo_vid.py

	3.  Commands :
						
		    Shift = Switch CNN model
			Up arrow = Generate Random Haze
			N = Generate Perlin-Noise Haze
			C = Clean N
			alt-Gt = Switch between CPU and GPU