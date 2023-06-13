/* Applied Video Sequence Analysis - Escuela Politecnica Superior - Universidad Autonoma de Madrid
 *
 *	Starter code for Task 3.1b of the assignment "Lab3 - Kalman Filtering for object tracking"
 *
 *	This code has been tested using Ubuntu 18.04, OpenCV 3.4.4 & Eclipse 2019-12
 *
 * Author: Juan C. SanMiguel (juancarlos.sanmiguel@uam.es)
 * Date: March 2022
 */

#include <opencv2/opencv.hpp>
#include "Kalman.hpp"
#include "blobs.hpp" //from lab2
#include "ShowManyImages.hpp"

using namespace cv;
using namespace std;

int main(int argc, char ** argv)
{
	Mat frame;
	//path for the input video
	std::string inputvideo = "/home/avsa/Documents/dataset_lab3/lab3.3/abandonedBox_600_1000_clip.mp4";
	//std::string inputvideo = "/home/avsa/Documents/dataset_lab3/lab3.3/boats_6950_7900_clip.mp4";
	//std::string inputvideo = "/home/avsa/Documents/dataset_lab3/lab3.3/pedestrians_800_1025_clip.mp4";
	//std::string inputvideo = "/home/avsa/Documents/dataset_lab3/lab3.3/streetCornerAtNight_0_100_clip.mp4";
	//switch between the 2 implemented models
	bool isConstantVelocity = true;

	//select if we want to write out our results into folders
	bool isWriteFrames = true;

	//if there are no measurementsS
	const int noMeasurement = -100;
	const Size not_used = Size(1 ,1);

	//parameters for the EOM
	const double learning_rate = 0.0001; //0.001 for all other sequences
	const Size opening_size = Size(3, 3);
	const Size closing_size = not_used; //set to not_used if not needed
	const MorphShapes type = MORPH_RECT;
	int history = 100;
	double varThreshold = 16;
	int min_width = 10;
	int min_height = 20;
	int connectivity = 8;
	bool detect_shadows = true;
	int frame_count = 0;
	int startFrameMeasurement = 3;

	string project_root_path = "/home/avsa/Documents/dataset_lab3/lab3.1/results";
	string results_tracking_path = project_root_path + "/tracking";
	string results_tracking_route_path = project_root_path + "/path";

	if (isWriteFrames) {
		string makedir_cmd = "mkdir " + project_root_path;
		system(makedir_cmd.c_str());
		makedir_cmd = "mkdir "+ results_tracking_path;
		system(makedir_cmd.c_str());

		makedir_cmd = "mkdir " + project_root_path;
		system(makedir_cmd.c_str());
		makedir_cmd = "mkdir "+ results_tracking_route_path;
		system(makedir_cmd.c_str());
	}

	if (argc == 4) {
		inputvideo = argv[1];
		isConstantVelocity = argv[2];
		isWriteFrames = argv[3];
	}
	VideoCapture cap(inputvideo);

	if (!cap.isOpened())
		throw std::runtime_error("Could not open video file " + inputvideo);


	//Balazses part
	cv::Mat gray, mask, opened, blobs_frame, first_frame;
	cv::Mat StructElement = cv::getStructuringElement(type, opening_size);
	cv::Mat StructElement_cl = cv::getStructuringElement(type, closing_size);

	Ptr<BackgroundSubtractor> BgS = cv::createBackgroundSubtractorMOG2(history, varThreshold, detect_shadows);

	cvBlob bigBlob;
	cv::Point coordinate_pair;
	std::vector<cvBlob> blobs;

	//hand-crafted Kalman
	Kalman *kalman = new Kalman(isConstantVelocity, noMeasurement);

	for (int i = 0; true; i++) {
		cap >> frame;
		frame_count++;
		if (!frame.data)
			break;

		//Grayscale conversion for foreground detection
		cvtColor(frame, gray, COLOR_RGB2GRAY);

		//Do background subtraction:
		BgS->apply(gray, mask, learning_rate);

		//Apply morbological opening:
		cv::morphologyEx(mask,opened, MORPH_OPEN, StructElement);
		cv::morphologyEx(opened,opened, MORPH_CLOSE, StructElement_cl);

		//initilaise
		if (frame_count < startFrameMeasurement)
		{
			opened = Mat::zeros(mask.cols, mask.rows, CV_32F);
		}

		//extract blobs (grassfire): I am using the blobs.cpp we created for Lab 2.
		extractBlobs(opened, blobs, connectivity); //int connectivity

		//chose the biggest blob and check if it is big enough
		if (biggestBlob(blobs, bigBlob, min_width, min_height)) //int for min width and height
		{
			//save coordinates of the blob if big enough
			coordinate_pair.x = bigBlob.x + int(bigBlob.w / 2);
			coordinate_pair.y = bigBlob.y + int(bigBlob.h / 2);
		} else {
			coordinate_pair.x = noMeasurement;
			coordinate_pair.y = noMeasurement;
		}
		kalman->predict();
		ActualState state = kalman->update(coordinate_pair);

		Mat detections = kalman->paintBlobImage(frame, state, coordinate_pair);
		std::string label;
		if (isConstantVelocity) {
			label = "KF using Constant Velocity";
		} else {
			label = "KF using Constant Acceleration";
		}

		cv::Mat path = kalman->paintTrackPath(frame);

		//showing the results on 2 frames
    	ShowManyImages(label, 4, detections, path, mask, opened);

        string outFileDetection = results_tracking_path + "/" + std::to_string(i) +".png";
        string outFilePath = results_tracking_route_path + "/" + std::to_string(i) +".png";

        //writing out the results into 2 different folders
        if (isWriteFrames) {
			imwrite(outFileDetection, detections);
			imwrite(outFilePath, path);
        }

		if( (char)waitKey(100) == 27)
			break;
	}

	//release our resources
	delete kalman;
	cap.release();
	destroyAllWindows();
	return 0;
}
