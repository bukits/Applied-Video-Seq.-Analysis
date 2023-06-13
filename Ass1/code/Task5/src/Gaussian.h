#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class Gaussian {
public:
	cv::Mat _weight;
	cv::Mat _mean;
	cv::Mat _deviation;
	Gaussian();
	virtual ~Gaussian();
};
