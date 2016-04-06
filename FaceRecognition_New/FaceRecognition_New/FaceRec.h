#pragma once
#include <iostream>
#include <string>
#include "opencv2/core.hpp"
//#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//include opencv core
#include "opencv2\core\core.hpp"
//#include "opencv2\contrib\contrib.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"
//#include "opencv2\contrib\contrib.hpp"
//file handling
#include <fstream>
#include <sstream>
using namespace std;
using namespace cv;

void dbread(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file) {
		string error = "no valid input file";
		CV_Error(CV_StsBadArg, error);
	}

	string line, path, label;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, label);
		if (!path.empty() && !label.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(label.c_str()));
		}
	}
}


void LBPHFaceTrainer() {

	vector<Mat> images;
	vector<int> labels;

	try {
		string filename = "E:/at.txt";
		dbread(filename, images, labels);

		cout << "size of the images is " << images.size() << endl;
		cout << "size of the labes is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	//lbph face recognier model
	Ptr<cv::FaceRecognizer> model = createLBPHFaceRecognizer();

	//training images with relevant labels 
	model->train(images, labels);

	//save the data in yaml file
	model->save("E:/LBPHface.yml");

	cout << "training finished...." << endl;

	waitKey(10000);
}