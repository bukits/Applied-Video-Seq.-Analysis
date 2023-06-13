#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//helper struct for storing the point and the name of the actual KF step
struct ActualState {
	cv::Point point;
	std::string name;
};

//KalmanFilter class which has uses the cv::KalmanFilter as a reference and adds functionality and the matrixes to handle them in one class
class Kalman {
public:
	//constructor which initializes the matrixes according to the model type
	Kalman(bool isConstantVelocity, int noMeasurement);

	//calculates the positions if there are no measurements
	void predict();

	//corrects the predictions from the measurements
	ActualState update(const cv::Point &measurement);

	//drawing functions
	cv::Mat paintBlobImage(cv::Mat frame, const ActualState &state, const cv::Point &meas);
	Mat paintTrackPath(const cv::Mat &frame);
private:
	//property for handling the model selection
	bool _isConstantVelocity;

	bool isInitialised;

	//built in reference
	cv::KalmanFilter kf;

	int state_size;
	int measurement_size;
	int control_size;

	//variable for no measurements handling
	int _noMeasurement;

	//matrices
	cv::Mat current_state;
	cv::Mat measurement;
	cv::Mat transition;
	cv::Mat observation;
	cv::Mat uncertainty;
	cv::Mat measurement_uncertainty;
	cv::Mat noise;

	//measurement points from the txt file
	std::vector<cv::Point> ground_truths;

	//helper vector for drawing
	std::vector<ActualState> states;
};
