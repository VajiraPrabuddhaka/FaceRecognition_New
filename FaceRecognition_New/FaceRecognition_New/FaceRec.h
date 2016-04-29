
#include <iostream>
#include "opencv2\opencv.hpp"

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

int  FaceRecognition() {

	cout << "start recognizing..." << endl;

	//load pre-trained data sets
	Ptr<FaceRecognizer>  model = createLBPHFaceRecognizer();
	model->load("E:/LBPHface.yml");
	/*
	Mat testSample = imread("E:/db/s41/5.pgm", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;
	*/

	//lbpcascades/lbpcascade_frontalface.xml
	string classifier = "C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(classifier)) {
		cout << " Error loading file" << endl;
		return -1;
	}

	VideoCapture cap(0);
	//VideoCapture cap("C:/Users/lsf-admin/Pictures/Camera Roll/video000.mp4");

	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return -1;
	}

	//double fps = cap.get(CV_CAP_PROP_FPS);
	//cout << " Frames per seconds " << fps << endl;
	namedWindow(window, 1);
	long count = 0;

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;
		//cap.read(frame);
		count = count + 1;//count frames;

		if (!frame.empty()) {

			//clone from original frame
			original = frame.clone();

			//convert image to gray scale and equalize
			cvtColor(original, graySacleFrame, CV_BGR2GRAY);
			//equalizeHist(graySacleFrame, graySacleFrame);

			//detect face in gray image
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			//number of faces detected
			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			//region of interest
			//cv::Rect roi;

			//person name
			string Pname = "";

			for (int i = 0; i < faces.size(); i++)
			{
				//region of interest
				Rect face_i = faces[i];

				//crop the roi from grya image
				Mat face = graySacleFrame(face_i);

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, Size(50,50), 1.0, 1.0, INTER_CUBIC);

				//recognizing what faces detected
				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout << " confidencde " << confidence << endl;

				//drawing green rectagle in recognize face
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				string text = "Detected";
				if (label == 40) {
					//string text = format("Person is  = %d", label);
					Pname = "gihan";
				}
				else {
					Pname = "unknown";
				}


				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				//name the person who is in the image
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
				//cv::imwrite("E:/FDB/"+frameset+".jpg", cropImg);

			}


			putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(original, "Person: " + Pname, Point(30, 90), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			//display to the winodw
			cv::imshow(window, original);

			//cout << "model infor " << model->getDouble("threshold") << endl;

		}
		if (waitKey(30) >= 0) break;
	}
}