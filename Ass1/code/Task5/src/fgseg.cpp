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
bgs::bgs(double threshold, bool rgb, bool selective_update, double alpha, int ghosts_threshold, double alpha_sh, double beta_sh, double saturation_th, double hue_th, int K, int start_deviation)
{
	_threshold=threshold;
	_alpha_sh = alpha_sh;
	_beta_sh = beta_sh;
	_saturation_th = saturation_th;
	_hue_th = hue_th;
	_ghosts_threshold = ghosts_threshold;
	//task 100:

	_rgb=rgb;
	_selective_update = selective_update;
	_alpha = alpha;
	_number_of_gausians = K;
	_start_deviation = start_deviation;
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
	_shadowmask = Mat::zeros(Size(_frame.cols,_frame.rows), CV_8UC1);
	_fgmask = Mat::zeros(Size(_frame.cols,_frame.rows), CV_8UC1);
}

void bgs::mixturedGaussian(cv::Mat Frame) {
	if (!_rgb) {
		cvtColor(Frame, Frame, COLOR_BGR2GRAY);
		Frame.copyTo(_frame);
		for (int j = 0; j < _frame.rows; j++) {
			for (int i = 0; i < _frame.cols; i++) {
				int val = _frame.at<uchar>(j, i);
				//sort weight (beginning is the highest)
			    std::sort(_gaussians.begin(), _gaussians.end(), [j, i](Gaussian* a, Gaussian *b) {
			        if (a->_weight.at<uchar>(j, i) > b->_weight.at<uchar>(j, i)) {
			            return true;
			        }
			    });

			    int selected_index = -1;
			    int k = 0;
			    for (Gaussian *g : _gaussians) {
			    	if (abs(val - g->_mean.at<uchar>(j, i)) <= 2.5 * g->_deviation.at<uchar>(j, i)) {
			    		selected_index = k;
			    		k++;
			    		break;
			    	}
			    }

			    _bgsmask.at<uchar>(j, i) = 0;
			    if (selected_index == -1) {
			    	_bgsmask.at<uchar>(j, i) = 255;
			    	selected_index = _number_of_gausians - 1;
			    	//replace last and selected becomes K.
			    	Gaussian *last = _gaussians[selected_index];
			    	last->_mean.at<float>(j, i) = val;
			    	last->_deviation.at<float>(j, i) = _start_deviation;
			    	//update weight maybe
			    }

			    float sum_of_weights = 0.0;
			    for (Gaussian *g : _gaussians) {
					if (k == selected_index) { //update selected
						g->_mean.at<float>(j, i) = _alpha * val + (1 - _alpha) * g->_mean.at<float>(j, i);
						g->_deviation.at<float>(j, i) = sqrt(_alpha * pow(val - g->_mean.at<float>(j, i), 2) + (1 - _alpha) * pow(g->_deviation.at<float>(j, i), 2));
						g->_weight.at<float>(j, i) = (1 - _alpha) * g->_weight.at<float>(j, i) + _alpha;
					} else {
						//decrease weight of others
						g->_weight.at<float>(j, i) = (1 - _alpha) * g->_weight.at<float>(j, i);
					}
					sum_of_weights += g->_weight.at<float>(j, i);
				}

				//NORMALIZE WEIGHTS
			    for (Gaussian *g : _gaussians) {
			    	g->_weight.at<float>(j, i) /= sum_of_weights;
				}
			}
		}
	} else {

	}
}

void bgs::realaseGaussians() {
	for (auto g : _gaussians) {
	     delete g;
	}
	_gaussians.clear();
}

void bgs::initMixturedGaussianParams(cv::Mat Frame) {
	if (!_rgb) {
		Gaussian *g_fix = new Gaussian();
		cv::Mat mean, deviation, weight;
		Frame.convertTo(mean, CV_32FC1);
		g_fix->_mean = mean;
		deviation = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * _start_deviation;
		g_fix->_deviation = deviation;
		weight = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * 0.5;
		g_fix->_weight = weight;
		_gaussians.push_back(g_fix);

		for (int i = 1; i < _number_of_gausians; ++i) {
			Gaussian *g_rand = new Gaussian();
			cv::Mat mean(Frame.cols, Frame.rows, CV_8UC1);
			randu(mean, Scalar(1), Scalar(255));
			mean.convertTo(mean, CV_32FC1);
			g_rand->_mean = mean;
			deviation = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * _start_deviation;
			g_rand->_deviation = deviation;
			weight = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * (1 - 0.5) / (_number_of_gausians - 1);
			g_rand->_weight = weight;
			_gaussians.push_back(g_rand);
		}
	} else {}
}




//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)//NOT FOR TASK 100
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
				//in the background mask the foreground part is 2100100
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
				if (abs(val - mean) <= 2 * deviation) {
					_bgsmask.at<uchar>(j, i) = 0;
					//update mean
					_mean.at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
					//update deviation
					_deviation.at<float>(j, i) = sqrt(_alpha * pow(val - mean, 2) + (1 - _alpha) * pow(deviation, 2));////
				}
				//foreground
				else {
					_bgsmask.at<uchar>(j, i) = 255;
				}
				if (_selective_update) {
					_mean.at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
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
				for (int z = 0; z < 3; ++z) {
					int val = frame_planes[z].at<uchar>(j, i);
					float mean = _mean_planes[z].at<float>(j, i);
					float deviation = _deviation_planes[z].at<float>(j, i);
					//background
					if (abs(val - mean) <= 2 * deviation) {
						_bgsmask.at<uchar>(j, i) = 255;
						//update mean
						_mean_planes[z].at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
						//update deviation
						_deviation_planes[z].at<float>(j, i) = sqrt(_alpha * pow(val - mean, 2) + (1 - _alpha) * pow(deviation, 2));
					}
					//foreground
					else {
						_bgsmask.at<uchar>(j, i) = 0;
					}
					if (_selective_update) {
						_mean_planes[z].at<float>(j, i) = _alpha * val + (1 - _alpha) * mean;
						_deviation_planes[z].at<float>(j, i) = sqrt(_alpha * pow(val - mean, 2) + (1 - _alpha) * pow(deviation, 2));
					}
				}
			}
		}
	}
}

void bgs::initGaussianParams(cv::Mat Frame) {
	if(!_rgb) {
		Frame.convertTo(_mean, CV_32FC1);
		_deviation = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * _start_deviation;
	} else {
		for (int z = 0; z < 3; z++) {
			Mat mean;
			Mat deviation = Mat::ones(Size(Frame.cols,Frame.rows), CV_32FC1) * _start_deviation;
			Frame.convertTo(mean, CV_32FC1);
			_mean_planes[z] = mean;
			_deviation_planes[z] = deviation;
		}
	}
}
