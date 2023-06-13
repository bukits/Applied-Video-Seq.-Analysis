#include <opencv2/opencv.hpp>
#include "objectTracking.hpp"
using namespace cv;
using namespace std;

/**
 * It creates the image of the histogram with normalization, which means dividing each values
 * of the histogram by the sum of pixels according to the bin.
 *
 * @param frame: selected feature frame
 * @return generated histogram
 */
cv::Mat Feature::initHistogram(cv::Mat frame) {
	float range[] = { 0, upperRange };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	cv::Mat hist;

	//calculate the histogram
	calcHist(&frame, 1, 0, Mat(), hist, 1, &num_bins, &histRange, uniform, accumulate);

	//normalizing according to the lecture
	hist = hist / cv::sum(hist)[0];

	return hist;
}

void Feature::setNumBins(int nBins) {
	num_bins = nBins;
}

Mat Feature::getConvertedImage() {
	return converted_img;
}

cv::Mat Gray::generateHistogram(cv::Mat frame_box) {
	upperRange = 255;
	cvtColor(frame_box, frame_box, cv::COLOR_BGR2GRAY);
	frame_box.copyTo(converted_img);
	Mat histImage = initHistogram(frame_box);
	return histImage;
}

cv::Mat H_color::generateHistogram(cv::Mat frame_box) {
	Mat frame;
	upperRange = 179;
	frame_box.copyTo(frame);
    cvtColor(frame, frame, cv::COLOR_BGR2HSV);
    frame.copyTo(converted_img);
    split(frame, color_planes);
	selected_plane = color_planes[0];
	Mat histImage = initHistogram(selected_plane);
	return histImage;
}

cv::Mat S_color::generateHistogram(cv::Mat frame_box) {
	Mat frame;
	upperRange = 255;
	frame_box.copyTo(frame);
    cvtColor(frame, frame, cv::COLOR_BGR2HSV);
    frame.copyTo(converted_img);
    split(frame, color_planes);
	selected_plane = color_planes[1];
	Mat histImage = initHistogram(selected_plane);
	return histImage;
}

cv::Mat V_color::generateHistogram(cv::Mat frame_box) {
	Mat frame;
	upperRange = 255;
	frame_box.copyTo(frame);
    cvtColor(frame, frame, cv::COLOR_BGR2HSV);
    frame.copyTo(converted_img);
    split(frame, color_planes);
	selected_plane = color_planes[2];
	Mat histImage = initHistogram(selected_plane);
	return histImage;
}

cv::Mat R_color::generateHistogram(cv::Mat frame_box) {
	upperRange = 255;
    split(frame_box, color_planes);
    //2 according to BGR
	selected_plane = color_planes[2];
	frame_box.copyTo(converted_img);
	Mat histImage = initHistogram(selected_plane);
	return histImage;
}

cv::Mat G_color::generateHistogram(cv::Mat frame_box) {
	upperRange = 255;
    split(frame_box, color_planes);
    //1 according to BGR
	selected_plane = color_planes[1];
	frame_box.copyTo(converted_img);
	Mat histImage = initHistogram(selected_plane);
	return histImage;
}

cv::Mat B_color::generateHistogram(cv::Mat frame_box) {
	upperRange = 255;
    split(frame_box, color_planes);
    //0 according to BGR
	selected_plane = color_planes[0];
	frame_box.copyTo(converted_img);
	Mat histImage = initHistogram(selected_plane);
	return histImage;
}

/**
 * It computes the HOG descriptors by the built in HOG in OpenCV.
 *
 * @param frame: frame of the bounding box, based on the ground truth
 * @return gradients calculated by HOG.
 *
 */
cv::Mat Gradient::computeHOGs(cv::Mat frame) {
    cv::HOGDescriptor hog;
    std::vector<float> descriptors;
    cv::Mat gradients;

	cvtColor(frame, frame, COLOR_BGR2GRAY);
	frame.copyTo(converted_img);
	resize(frame, frame, cv::Size(64, 128));
    hog.nbins = num_bins;

    hog.compute(frame, descriptors, cv::Size(8, 8), cv::Size(0, 0));
    gradients = Mat(descriptors).clone();
	return gradients;
}

/**
 * Override function which computes the HOG descriptors, calling computeHOGs and creates the histogram differently.
 * It does not calculate the histogram bins as the others do.
 *
 * @param frame_box: frame of the bounding box, based on the ground truth
 * @return generated histogram
 *
 */
cv::Mat Gradient::generateHistogram(cv::Mat frame_box) {
	Mat gradients = computeHOGs(frame_box);
	return gradients;
}

/**
	 * It calculates the histogram Bhattacharyya distances between the candidates and the target model. It is implemented in the class since
	 * for the candidates it is needed to calculate the histogram according to the selected feature. It is know by the class itself.
	 *
	 * @param cand_pred: a list of candidates
	 * @param frame: the actual frame
	 * @param hist_object: the target model's histogram
	 * @return list of distances between the candidates and the target model
	 */
std::vector<double> Feature::distance(const cv::Mat &hist_object, const std::vector<cv::Rect> &cand_pred, const cv::Mat &frame)
{
	std::vector<double> dist;

	for (size_t i = 0; i < cand_pred.size(); i++) //iterate candidates
	{
		//get the feature histogram patch of a single candidate:
		Mat hist_candidate = generateHistogram(frame(cand_pred[i]));

		double distance = cv::compareHist(hist_object, hist_candidate, cv::HISTCMP_BHATTACHARYYA);

		dist.push_back(distance);
	}

	return dist;
}
