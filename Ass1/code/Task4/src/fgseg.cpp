/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>
#include "fgseg.hpp"

using namespace fgseg;

//default constructor
bgs::bgs(double threshold, bool rgb, bool selective_update, double alpha, int ghosts_threshold, double alpha_sh, double beta_sh, double saturation_th, double hue_th)
{
	_rgb=rgb;
	_threshold=threshold;
	_alpha_sh = alpha_sh;
	_beta_sh = beta_sh;
	_saturation_th = saturation_th;
	_hue_th = hue_th;
	_selective_update = selective_update;
	_alpha = alpha;
	_ghosts_threshold = ghosts_threshold;
}

//default destructor
bgs::~bgs(void)
{
}

//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame)
{

	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
	}
	Frame.copyTo(_bkg);
	_bgsmask = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1);
	_diff = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1);
}

//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{
	if (!_rgb){
		cvtColor(Frame, Frame, COLOR_BGR2GRAY); // to work with gray even if input is color
		Frame.copyTo(_frame);
		_diff = abs(_frame - _bkg);
		_bgsmask = _diff > _threshold;
	} else {
		Frame.copyTo(_frame);
		_diff = abs(_frame - _bkg);
		Mat diff_planes[3];
		split(_diff, diff_planes);
		_bgsmask = diff_planes[0] + diff_planes[1] + diff_planes[2] > _threshold;
	}

}

void bgs::removeShadows()
{
	if (_rgb) {
		_bgsmask.copyTo(_fgmask);
		for (int j = 0; j < _bgsmask.rows; j++) {
			for (int i = 0; i < _bgsmask.cols; i++) {
				//in the background mask the foreground part is 255
				if (_bgsmask.at<uchar>(j, i) == 255 && _shadowmask.at<uchar>(j, i) == 255) {
					_fgmask.at<uchar>(j, i) = 0;
					_bgsmask.at<uchar>(j, i) = 0;
				}
			}
		}
	} else {
		_bgsmask.copyTo(_fgmask);
		_bgsmask.copyTo(_shadowmask);
	}
}

//ADD ADDITIONAL FUNCTIONS HERE
void bgs::initCounter(cv::Mat Frame) {
	_counter = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1);
}

void bgs::updateBackground() {
	if (_selective_update) {
		for (int j = 0; j < _bgsmask.rows; j++) {
			for (int i = 0; i < _bgsmask.cols; i++) {
				if ((int)_bgsmask.at<uchar>(j, i) == 0) {
					if (!_rgb) {
						_bkg.at<uchar>(j, i) = _alpha * _frame.at<uchar>(j, i) + (1 - _alpha) * _bkg.at<uchar>(j, i);
					} else {
						_bkg.at<Vec3b>(j, i) = _alpha * _frame.at<Vec3b>(j, i) + (1 - _alpha) * _bkg.at<Vec3b>(j, i);
					}
				}
			}
		}
	} else {
		_bkg = _alpha * _frame + (1 - _alpha) * _bkg;
	}
}

void bgs::removeGhosts() {
	for (int j = 0; j < _bgsmask.rows; j++) {
		for (int i = 0; i < _bgsmask.cols; i++) {
			//detected as foreground
			if ((int)_bgsmask.at<uchar>(j, i) == 255) {
				_counter.at<uchar>(j, i) = _counter.at<uchar>(j, i) + 1;
			//detected as background
			} else if ((int)_bgsmask.at<uchar>(j, i) == 0){
				_counter.at<uchar>(j, i) = 0;
			}
			if ((int)_counter.at<uchar>(j, i) >= _ghosts_threshold) {
				_bgsmask.at<uchar>(j, i) = 0;
				if (_rgb) {
					_bkg.at<Vec3b>(j, i) = _frame.at<Vec3b>(j, i);
				} else {
					_bkg.at<uchar>(j, i) = _frame.at<uchar>(j, i);
				}
				_counter.at<uchar>(j, i) = 0;
			}
		}
	}
}

void bgs::detectShadow()
{
	if (_rgb) {
		cvtColor(_frame, _frame, COLOR_BGR2HSV);

		_shadowmask = Mat::zeros(Size(_frame.cols,_frame.rows), CV_8UC1);

		Mat frame_HSV_planes_shadow[3];
		Mat bkg_HSV_planes_shadow[3];

		split(_frame, frame_HSV_planes_shadow);
		split(_bkg, bkg_HSV_planes_shadow);

		for (int j = 0; j < _frame.rows; j++) {
			for (int i = 0; i < _frame.cols; i++) {
				double pixel_frame_H = frame_HSV_planes_shadow[0].at<uchar>(j, i);
				double pixel_frame_S = frame_HSV_planes_shadow[1].at<uchar>(j, i);
				double pixel_frame_V = frame_HSV_planes_shadow[2].at<uchar>(j, i);

				double pixel_bkg_H = bkg_HSV_planes_shadow[0].at<uchar>(j, i);
				double pixel_bkg_S = bkg_HSV_planes_shadow[1].at<uchar>(j, i);
				double pixel_bkg_V = bkg_HSV_planes_shadow[2].at<uchar>(j, i);

				double D_h = min(abs(2*(pixel_frame_H - pixel_bkg_H)), 360 - abs(2*(pixel_frame_H - pixel_bkg_H)));
				bool exp_1 = (pixel_frame_V / pixel_bkg_V) >= _alpha_sh && (pixel_frame_V / pixel_bkg_V) <= _beta_sh;
				bool exp_2 = abs(pixel_frame_S - pixel_bkg_S) <= _saturation_th;
				bool exp_3 = D_h <= _hue_th;
				if(exp_1 && exp_2 && exp_3) {
					_shadowmask.at<uchar>(j, i) = 255;
				 }
			}
		}
	}
}

void bgs::gaussian(cv::Mat Frame) {
	if (!_rgb) {
		cvtColor(Frame, Frame, COLOR_BGR2GRAY);
		Frame.copyTo(_frame);
		for (int j = 0; j < _frame.rows; j++) {
			for (int i = 0; i < _frame.cols; i++) {
				int val = _frame.at<uchar>(j, i);
				float mean = _mean.at<float>(j, i);
				float deviation = _deviation.at<float>(j, i);
				//background
				if (_selective_update) {
					if (abs(val - mean) <= 2 * deviation) {
						_bgsmask.at<uchar>(j, i) = 0;
						//update mean
						_mean.at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
						//update deviation
						_deviation.at<float>(j, i) = sqrt(_alpha * pow(val - mean, 2) + (1 - _alpha) * pow(deviation, 2));
					}
					//foreground
					else {
						_bgsmask.at<uchar>(j, i) = 255;
					}
				} else {
					if (abs(val - mean) <= 2 * deviation) {
						_bgsmask.at<uchar>(j, i) = 0;
					} else {
						_bgsmask.at<uchar>(j, i) = 255;
					}

					_mean.at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
					//update deviation
					_deviation.at<float>(j, i) = sqrt(_alpha * pow(val - mean, 2) + (1 - _alpha) * pow(deviation, 2));
				}
			}
		}
	} else {
		Frame.copyTo(_frame);
		Mat frame_planes[3];
		split(_frame, frame_planes);

		for (int j = 0; j < _frame.rows; j++) {
			for (int i = 0; i < _frame.cols; i++) {
				bool allIn = true;
				for (int z = 0; z < 3; ++z) {
					int val = frame_planes[z].at<uchar>(j, i);
					float mean = _mean_planes[z].at<float>(j, i);
					float deviation = _deviation_planes[z].at<float>(j, i);
					if (abs(val - mean) > 2 * deviation) {
						allIn = false;
					}
				}
				if (_selective_update) {
					if (allIn) {
						for (int z = 0; z < 3; ++z) {
							int val = frame_planes[z].at<uchar>(j, i);
							float mean = _mean_planes[z].at<float>(j, i);
							float deviation = _deviation_planes[z].at<float>(j, i);
							_bgsmask.at<uchar>(j, i) = 0;
							//update mean
							_mean_planes[z].at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
							//update deviation
							_deviation_planes[z].at<float>(j, i) = sqrt(_alpha * pow(val - mean, 2) + (1 - _alpha) * pow(deviation, 2));
						}
					} else {
						_bgsmask.at<uchar>(j, i) = 255;
					}
				} else {
					for (int z = 0; z < 3; ++z) {
						int val = frame_planes[z].at<uchar>(j, i);
						float mean = _mean_planes[z].at<float>(j, i);
						float deviation = _deviation_planes[z].at<float>(j, i);
						if (abs(val - mean) <= 2 * deviation) {
							_bgsmask.at<uchar>(j, i) = 0;
						} else {
							_bgsmask.at<uchar>(j, i) = 255;
						}

						_mean_planes[z].at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
						//update deviation
						_deviation_planes[z].at<float>(j, i) = sqrt(_alpha * pow(val - mean, 2) + (1 - _alpha) * pow(deviation, 2));
					}
				}
			}
		}
	}
}

void bgs::initGaussianParams(cv::Mat Frame) {
	if(!_rgb) {
		cvtColor(Frame, Frame, COLOR_BGR2GRAY);
		Frame.convertTo(_mean, CV_32FC1);
		_deviation = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * 100;
	} else {
		Frame.convertTo(Frame, CV_32FC1);
		split(Frame, _mean_planes);

		Mat deviation = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * 100;
		for (int z = 0; z < 3; z++) {
			_deviation_planes[z] = deviation;
		}
	}
}
