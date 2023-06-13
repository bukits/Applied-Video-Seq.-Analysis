#ifndef OBJECTTRACKING_HPP_
#define  OBJECTTRACKING_HPP_

#include <opencv2/opencv.hpp>

/**
 * Base class of the selected feature
 */
class Feature {
public:
	/**
	 * Virtual function which is overwritten in every inherited classes.
	 * It generates the histogram of the selected feature by switching to the color space
	 * and calculate the histogram on this.
	 *
	 * @param frame_box: frame of the bounding box, based on the ground truth
	 * @return generated histogram
	 *
	 */
	virtual cv::Mat generateHistogram(cv::Mat frame_box) = 0;
	/**
	 * It calculates the histogram Bhattacharyya distances between the candidates and the target model. It is implemented in the class since
	 * for the candidates it is needed to calculate the histogram according to the selected feature. It is know by the class itself.
	 *
	 * @param cand_pred: a list of candidates
	 * @param frame: the actual frame
	 * @param hist_object: the target model's histogram
	 * @return list of distances between the candidates and the target model
	 */
	std::vector<double> distance(const cv::Mat &hist_object, const std::vector<cv::Rect> &cand_pred, const cv::Mat &frame);

	/**
	 * Getter function for the frame in the selected color feature
	 */
	cv::Mat getConvertedImage();

	/**
	 * Setter function for the number of bins
	 */
	void setNumBins(int num_bins);

	/**
	 * Virtual destruktor
	 */
	virtual ~Feature() {}

protected:
	/**
	 * It creates the image of the histogram with normalization, which means dividing each values
	 * of the histogram by the sum of pixels according to the bin.
	 *
	 * @param frame: selected feature frame
	 * @return generated normalized histogram
	 */
	cv::Mat initHistogram(cv::Mat frame);

	/**
	 * Protected variables
	 */
	//vector for the dimensions
	std::vector<cv::Mat> color_planes;

	//plane of the feature eg.: (RGB) -> R
	cv::Mat selected_plane;

	//number of bins
	int num_bins;

	//frame converted to the selected feature
	cv::Mat converted_img;

	//upper range of the histogram
	float upperRange;
};

/**
 * Inherited class of the GRAY feature
 */
class Gray : public Feature {
public:
    cv::Mat generateHistogram(cv::Mat frame_box) override;
};

/**
 * Inherited class of the Hue feature (HSV)
 */
class H_color : public Feature {
public:
    cv::Mat generateHistogram(cv::Mat frame_box) override;
};

/**
 * Inherited class of the Saturate feature (HSV)
 */
class S_color : public Feature {
public:
    cv::Mat generateHistogram(cv::Mat frame_box) override;
};

/**
 * Inherited class of the Value feature (HSV)
 */
class V_color : public Feature {
public:
    cv::Mat generateHistogram(cv::Mat frame_box) override;
};

/**
 * Inherited class of the Red feature (RGB)
 */
class R_color : public Feature {
public:
    cv::Mat generateHistogram(cv::Mat frame_box) override;
};

/**
 * Inherited class of the Green feature (RGB)
 */
class G_color : public Feature {
public:
    cv::Mat generateHistogram(cv::Mat frame_box) override;
};

/**
 * Inherited class of the Blue feature (RGB)
 */
class B_color : public Feature {
public:
    cv::Mat generateHistogram(cv::Mat frame_box) override;
};

/**
 * Inherited class of the HOG feature
 */
class Gradient : public Feature {
public:
	/**
	 * Override function which computes the HOG descriptors, calling computeHOGs and creates the histogram differently.
	 * It does not calculate the histogram bins as the others do.
	 *
	 * @param frame_box: frame of the bounding box, based on the ground truth
	 * @return generated histogram
	 *
	 */
    cv::Mat generateHistogram(cv::Mat frame_box) override;

private:
	/**
	 * It computes the HOG descriptors by the built in HOG in OpenCV.
	 *
	 * @param frame: frame of the bounding box, based on the ground truth
	 * @return gradients calculated by HOG.
	 *
	 */
    cv::Mat computeHOGs(cv::Mat frame);
};

#endif
