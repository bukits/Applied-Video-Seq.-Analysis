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
bgs::bgs(double threshold, double alpha, bool selective_bkg_update, int threshold_ghosts2, bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
	_alpha = alpha;
	_selective_update = selective_bkg_update;
	_ghosts_threshold = threshold_ghosts2;
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
		Frame.copyTo(_bkg);
	}
	Frame.copyTo(_bkg);

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
//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows()
{
	_bgsmask.copyTo(_shadowmask); // creates the mask (currently with bgs)

	absdiff(_bgsmask, _bgsmask, _shadowmask);// currently void function mask=0 (should create shadow mask)

	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
}

//ADD ADDITIONAL FUNCTIONS HERE
void bgs::initCounter(cv::Mat Frame) {
	_counter = Mat::zeros(Size(Frame.cols,Frame.rows), CV_8UC1);
}

void bgs::updateBackground() {
	if (_selective_update) {
		for (int j = 0; j < _bgsmask.rows; j++) {
			for (int i = 0; i < _bgsmask.cols; i++) {
				//detected as a background
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



