//Setup parameters in lines:
//line 16 for uncertainty_elements
//line 47 and 91 for uncertainty_scaling
	//for constant velocity and constant acceleration model respectively

#include <opencv2/opencv.hpp>
#include "Kalman.hpp"



Kalman::Kalman(bool isConstantVelocity, int noMeasurement) {
	_isConstantVelocity = isConstantVelocity;
	_noMeasurement = noMeasurement;
	measurement_size = 2;
	control_size = 0;
	isInitialised = false;

	std::vector<float> observation_elements {1.0f, 1.0f};
	//CHOOSE PARAMETERS----------------------------------------------------
	//Video 2 3, and 5:
	std::vector<float> measurement_uncertainty_elements {25.0f, 25.0f};
	//Video 6:
	//std::vector<float> measurement_uncertainty_elements {125.0f, 125.0f};
	//---------------------------------------------------------------------
	if (_isConstantVelocity) {
		state_size = 4;

		std::vector<float> transition_elements {1.0f, 1.0f};
		std::vector<float> noise_elements {25.0f, 10.0f, 25.0f, 10.0f};

		transition = Mat::zeros(cv::Size(state_size, state_size), CV_32F);
		cv::setIdentity(transition);
		transition.at<float>(0, 1) = transition_elements[0];
		transition.at<float>(2, 3) = transition_elements[1];

		observation = Mat::zeros(cv::Size(state_size, measurement_size), CV_32F);
		observation.at<float>(0, 0) = observation_elements[0];
		observation.at<float>(1, 2) = observation_elements[1];

		noise = Mat::zeros(cv::Size(state_size, state_size), CV_32F);
		noise.at<float>(0, 0) = noise_elements[0];
		noise.at<float>(1, 1) = noise_elements[1];
		noise.at<float>(2, 2) = noise_elements[2];
		noise.at<float>(3, 3) = noise_elements[4];


		//CHOOSE PARAMETERS----------------------------------------------------

		std::vector<float> uncertainty_scaling = {10e1, 10e1, 10e1, 10e1};

		//std::vector<float> uncertainty_scaling = {10e3, 10e3, 10e3, 10e3};

		//std::vector<float> uncertainty_scaling = {10e5, 10e5, 10e5, 10e5};

		//---------------------------------------------------------------------


		uncertainty = Mat::zeros(cv::Size(state_size, state_size), CV_32F);
		cv::setIdentity(uncertainty);

		for (int i = 0; i < state_size; i++) {
			uncertainty.at<float>(i, i) *= uncertainty_scaling[i];
		}
	} else {
		state_size = 6;

		std::vector<float> transition_elements {1.0f, 0.5f, 1.0f, 1.0f, 0.5f, 1.0f};
		std::vector<float> noise_elements {25.0f, 10.0f, 1.0f, 25.0f, 10.0f, 1.0f};

		transition = Mat::zeros(cv::Size(state_size, state_size), CV_32F);
		cv::setIdentity(transition);
		transition.at<float>(0, 1) = transition_elements[0];
		transition.at<float>(0, 2) = transition_elements[1];
		transition.at<float>(1, 2) = transition_elements[2];
		transition.at<float>(3, 4) = transition_elements[3];
		transition.at<float>(3, 5) = transition_elements[4];
		transition.at<float>(4, 5) = transition_elements[5];

		observation = Mat::zeros(cv::Size(state_size, measurement_size), CV_32F);
		observation.at<float>(0, 0) = observation_elements[0];
		observation.at<float>(1, 3) = observation_elements[1];

		noise = Mat::zeros(cv::Size(state_size, state_size), CV_32F);
		noise.at<float>(0, 0) = noise_elements[0];
		noise.at<float>(1, 1) = noise_elements[1];
		noise.at<float>(2, 2) = noise_elements[2];
		noise.at<float>(3, 3) = noise_elements[3];
		noise.at<float>(4, 4) = noise_elements[4];
		noise.at<float>(5, 5) = noise_elements[5];

		//CHOOSE PARAMETERS----------------------------------------------------

		std::vector<float> uncertainty_scaling = {10e1, 10e1, 10e1, 10e1, 10e1, 10e1};

		//std::vector<float> uncertainty_scaling = {10e3, 10e3, 10e3, 10e3, 10e3, 10e3};

		//std::vector<float> uncertainty_scaling = {10e5, 10e5, 10e5, 10e5, 10e5, 10e5};

		//---------------------------------------------------------------------

		uncertainty = Mat::zeros(cv::Size(state_size, state_size), CV_32F);
		cv::setIdentity(uncertainty);

		for (int i = 0; i < state_size; i++) {
			uncertainty.at<float>(i, i) *= uncertainty_scaling[i];
		}
	}


	//here I have changed the values of the diagonal from 25 to 100 to get more accurate predictions
	measurement_uncertainty = Mat::zeros(cv::Size(measurement_size, measurement_size), CV_32F);
	measurement_uncertainty.at<float>(0, 0) = measurement_uncertainty_elements[0];
	measurement_uncertainty.at<float>(1, 1) = measurement_uncertainty_elements[1];

	current_state = Mat::zeros(cv::Size(1, state_size), CV_32F);
	measurement = Mat::zeros(measurement_size, 1, CV_32F);
	kf = KalmanFilter(state_size, measurement_size, control_size, CV_32F);
	transition.copyTo(kf.transitionMatrix);
	observation.copyTo(kf.measurementMatrix);
	uncertainty.copyTo(kf.errorCovPre);
	noise.copyTo(kf.processNoiseCov);
	measurement_uncertainty.copyTo(kf.measurementNoiseCov);
}

void Kalman::predict() {
	if (isInitialised) {
		kf.predict();
	}
}

ActualState Kalman::update(const cv::Point &measurement_point) {
	ActualState state;

	if (measurement_point.x != _noMeasurement || measurement_point.y != _noMeasurement) {
		if (!isInitialised) {
			isInitialised = true;
		}
		//initialize KF to get better coordinates from the beginning as well
		measurement.at<float>(0) = measurement_point.x;
		measurement.at<float>(1) = measurement_point.y;
		kf.correct(measurement);
		state.name = "Corrected";
		kf.statePost.copyTo(current_state);
	} else {
		state.name = "Predicted";
		kf.statePre.copyTo(current_state);
	}

	if (_isConstantVelocity) {
		state.point.x = current_state.at<float>(0);
		state.point.y = current_state.at<float>(2);
	} else {
		state.point.x = current_state.at<float>(0);
		state.point.y = current_state.at<float>(3);
	}

	ground_truths.push_back(measurement_point);
	states.push_back(state);
	return state;
}

//here I am giving the frame as a reference to get cleaner code, I do not have to return with changed frame
void Kalman::paintBlobImage(cv::Mat &frame, const ActualState &state, const cv::Point &meas) {
	Scalar color;
	if (state.name == "Corrected") {
		color = Scalar(255,0,0);
	} else if (state.name == "Predicted") {
		color = Scalar(0,255,0);
	}

	Point p1 = Point(state.point.x - 40, state.point.y - 40);
	Point p2 = Point(state.point.x + 40, state.point.y + 40);

	Point p1_meas = Point(meas.x - 40, meas.y - 40);
	Point p2_meas = Point(meas.x + 40, meas.y + 40);

	rectangle(frame, p1_meas, p2_meas, Scalar(255, 255, 255), 4, 8, 0);
	rectangle(frame, p1, p2, color, 4, 8, 0);
	putText(frame, state.name, p1, FONT_HERSHEY_SIMPLEX, 2, color, 2);

	putText(frame, "Measurement: " + to_string(meas.x) + ";" + to_string(meas.y), Point(10, 15 + 20), FONT_HERSHEY_SIMPLEX, 2, Scalar(255, 255, 255), 2);
	if (state.name == "Corrected") {
		putText(frame, "Corrected: (" + to_string(state.point.x) + ";" + to_string(state.point.y) + ")", Point(10, 65 + 20), FONT_HERSHEY_SIMPLEX, 2, Scalar(255,0,0), 2);
	} else if (state.name == "Predicted") {
		putText(frame, "Prediction: (" + to_string(state.point.x) + ";" + to_string(state.point.y) + ")", Point(10, 65 + 20), FONT_HERSHEY_SIMPLEX, 2, Scalar(0,255,0), 2);
	}
}

Mat Kalman::paintTrackPath(const cv::Mat &frame) {
	cv::Mat track;
	frame.copyTo(track);
	for (unsigned int i = 0; i < ground_truths.size(); ++i) {
		cv::Point gt = ground_truths[i];
		cv::Point actual = states[i].point;
		string type = states[i].name;
		Scalar color;
		if (type == "Predicted") {
			color = Scalar(0,255,0);
		} else if (type == "Corrected") {
			color = Scalar(0,0,255);
		}
		drawMarker(track, gt, Scalar(255, 255, 255), MARKER_CROSS, 80, 4);
		drawMarker(track, actual, color, MARKER_SQUARE, 80, 4);
	}
	return track;
}
