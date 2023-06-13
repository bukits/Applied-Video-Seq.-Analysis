/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.hpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */
//FINAL

#include <opencv2/opencv.hpp>
#include "Gaussian.h"

#ifndef FGSEG_H_INCLUDE
#define FGSEG_H_INCLUDE

using namespace cv;
using namespace std;

namespace fgseg {


	//Declaration of FGSeg class based on BackGround Subtraction (bgs)
	class bgs{
	public:

		//constructor with parameter "threshold"
		bgs(double threshold, bool rgb, bool selective_update, double alpha, int ghosts_threshol, double alpha_sh, double beta_sh, double saturation_th, double hue_th, int K, int start_deviation);

		//destructor
		~bgs(void);

		//method to initialize bkg (first frame - hot start)
		void init_bkg(cv::Mat Frame);

		//method to perform BackGroundSubtraction
		void bkgSubtraction(cv::Mat Frame);

		//method to detect and remove shadows in the binary BGS mask
		void removeShadows();

		//returns the BG image
		cv::Mat getBG(){return _bkg;};

		//returns the DIFF image
		cv::Mat getDiff(){return _diff;};

		//returns the BGS mask
		cv::Mat getBGSmask(){return _bgsmask;};

		//returns the binary mask with detected shadows
		cv::Mat getShadowMask(){return _shadowmask;};

		//returns the binary FG mask
		cv::Mat getFGmask(){return _fgmask;};


		//ADD ADITIONAL METHODS HERE
		//...
		void detectShadow();
		void updateBackground();
		void removeGhosts();
		void initCounter(cv::Mat Frame);
		//task 5:
		//XXXXXXXXXXXXXXXXXXXXXXXXXXXX
		void mixturedGaussian(cv::Mat Frame);
		void initMixturedGaussianParams(cv::Mat Frame);
		void realaseGaussians();
		//XXXXXXXXXXXXXXXXXXXXXXXXXXXX
		void gaussian(cv::Mat Frame);
		void initGaussianParams(cv::Mat Frame);
	private:
		cv::Mat _bkg; //Background model
		cv::Mat	_frame; //current frame
		cv::Mat _diff; //abs diff frame
		cv::Mat _bgsmask; //binary image for bgssub (FG)
		cv::Mat _shadowmask; //binary mask for detected shadows
		cv::Mat _fgmask; //binary image for foreground (FG)

		bool _rgb;

		double _threshold;
		//ADD ADITIONAL VARIABLES HERE
		//...
		double _alpha;
		double _alpha_sh;
		double _beta_sh;
		double _saturation_th;
		double _hue_th;
		bool _selective_update;

		Mat _counter;
		int _ghosts_threshold;

		Mat _mean, _deviation;
		Mat _mean_planes[3];
		Mat _deviation_planes[3];

		//task 5:
		int _number_of_gausians;
		int _start_deviation;
		std::vector<Gaussian*> _gaussians;

	};//end of class bgs

}//end of namespace

#endif




