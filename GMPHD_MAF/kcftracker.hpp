/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: horizontal area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */
#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_

#include "VOT.h"


class KCFTracker : public VOT
{
public:
    // Constructor
	// ���� �ɼ� (dense):	true, false, true, true, false
	/// bool hog = true, bool fixed_window = false, bool multiscale = true, bool lab = true, bool roi_only = false
	// ���� �ɼ� (medium):	true, true, false, true, false
	/// bool hog = true, bool fixed_window = true, bool multiscale = false, bool lab = true, bool roi_only = false
	// ����ȭ �ɼ� (simple):	false, true, false, false, false (1.3~1.4 �� �� ����)
	/// bool hog = false, bool fixed_window = true, bool multiscale = false, bool lab = false, bool roi_only = false
	KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = false, bool lab = true, bool roi_only = false);

    // Initialize tracker 
    virtual void init(const cv::Mat& image, const cv::Rect &roi, const cv::Mat &mask = cv::Mat(), const bool& USE_MASK = false);
    // Update position based on the new frame
    virtual cv::Rect update(cv::Mat& image);
	virtual cv::Rect update(cv::Mat& image, float& confProb);
	// Update position based on the new frame with ROI (sym)
	virtual cv::Rect update(const cv::Mat& image, float& confProb,  
		cv::Mat &confMapVis, const cv::Rect &roi = cv::Rect(0, 0, 0, 0), const bool& GET_VIS = false,
		const cv::Mat &mask = cv::Mat(), const bool& USE_MASK = false);

    float interp_factor; // linear interpolation factor for adaptation
    float sigma; // gaussian kernel bandwidth
    float lambda; // regularization
    int cell_size; // HOG cell size
    int cell_sizeQ; // cell size^2, to avoid repeated operations
    float padding; // extra area surrounding the target
    float output_sigma_factor; // bandwidth of gaussian target
    int template_size; // template size
    float scale_step; // scale step for multi-scale estimation
    float scale_weight;  // to downweight detection scores of other scales for added stability

protected:
    // Detect object in the current frame.
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);
	// Detect object in the current frame. (sym)
	cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value, cv::Mat& res);

    // train tracker with a single image
    void train(cv::Mat x, float train_interp_factor);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

    // Create Gaussian Peak. Function called only in the first frame.
    cv::Mat createGaussianPeak(int sizey, int sizex);

public:
    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);
	// Obtain sub-window from image, with a specified ROI and extract features
	cv::Mat getFeaturesROI(const cv::Mat & image, const cv::Rect &roi_specified, float scale_adjust = 1.0f);
	cv::Mat clone_tmpl() {
		return this->_tmpl.clone();
	}
	cv::Mat clone_alphaf() {
		return this->_alphaf.clone();
	}
	// Operator =
	KCFTracker& operator=(const KCFTracker& copy);
	// Deep Copy
	void copyTo(KCFTracker& dst);
protected:
    // Initialize Hanning window. Function called only in the first frame.
    void createHanningMats();

    // Calculate sub-pixel peak for one dimension
    float subPixelPeak(float left, float center, float right);

public:
    cv::Mat _alphaf;
    cv::Mat _prob;
    cv::Mat _tmpl;
    cv::Mat _num;
    cv::Mat _den;
    cv::Mat _labCentroids;

//private:
    int size_patch[3];
    cv::Mat hann;
    cv::Size _tmpl_sz;
    float _scale;
	float _scale_MOT;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;

	int fWidth;
	int fHeight;
private:
	cv::Mat color_map;
	void InitializeColorMap() {
		cv::Mat img_gray(32, 256, CV_8UC1);

		for (int r = 0; r < img_gray.rows; r++)
			for (int c = 0; c < img_gray.cols; c++)
				img_gray.at<uchar>(r, c) = (uchar)(c);
		//
		//for (int r = 0; r < img_gray.rows; r++) {
		//	for (int c = 0; c < 26; c++)
		//	img_gray.at<uchar>(r, c) = (uchar)(0);
		//}

		if (img_gray.empty()) {
			CV_Error(cv::Error::StsBadArg, "Sample image is empty. Please adjust your path, so it points to a valid input image!");
		}
		color_map = cv::Mat(32, 256, CV_8UC3);

		// Apply the colormap:
		applyColorMap(img_gray(cv::Rect(0, 0, 256, 32)), this->color_map(cv::Rect(0, 0, 256, 32)), cv::COLORMAP_JET);

		for (int r = 0; r < img_gray.rows; r++) {
			for (int c = 0; c < 1; c++)
				this->color_map.at<cv::Vec3b>(r, c) = cv::Vec3b(0,0,0);
		}
	}
	// Rect Region Correction for preventing out of frame
	cv::Rect RectExceptionHandling(int fWidth, int fHeight, cv::Rect rect) {

		if (rect.x < 0) {
			//rect.width += rect.x;
			rect.x = 0;
		}
		if (rect.width < 0) rect.width = 0;
		if (rect.x >= fWidth) rect.x = fWidth - 1;
		if (rect.width > fWidth) rect.width = fWidth;
		if (rect.x + rect.width >= fWidth) rect.width = fWidth - rect.x -1;

		if (rect.y < 0) {
			//rect.height += rect.y;
			rect.y = 0;
		}
		if (rect.height < 0) rect.height = 0;
		if (rect.y >= fHeight) rect.y = fHeight - 1;
		if (rect.height > fHeight) rect.height = fHeight;
		if (rect.y + rect.height >= fHeight) rect.height = fHeight - rect.y -1;

		return rect;
	}
	void cvPrintMat(cv::Mat matrix, std::string name)
	{
		/*
		<Mat::type()>
		depth�� channels���� �����ϴ� ���� ex. CV_64FC1
		<Mat::depth()>
		CV_8U - 8-bit unsigned integers ( 0..255 )
		CV_8S - 8-bit signed integers ( -128..127 )
		CV_16U - 16-bit unsigned integers ( 0..65535 )
		CV_16S - 16-bit signed integers ( -32768..32767 )
		CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
		CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
		CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
		*/
		printf("Matrix %s=\n", name);
		if (matrix.depth() == CV_64F) {
			//int channels = matrix.channels();
			for (int r = 0; r < matrix.rows; r++) {
				for (int c = 0; c < matrix.cols; c++) {
					//printf("(");
					//for( int cn=0 ; cn<channels ; cn++){
					printf("%6.2lf ", matrix.at<double>(r, c)/*[cn]*/);
					//} printf(")");
				}
				printf("\n");
			}
		}
		else if (matrix.depth() == CV_32F) {
			//int channels = matrix.channels();
			for (int r = 0; r < matrix.rows; r++) {
				for (int c = 0; c < matrix.cols; c++) {
					//printf("(");
					//for( int cn=0 ; cn<channels ; cn++){
					printf("%6.2f ", matrix.at<float>(r, c)/*[cn]*/);
					//} printf(")");
				}
				printf("\n");
			}
		}
		else if (matrix.depth() == CV_8U) {
			//int channels = matrix.channels();
			for (int r = 0; r < matrix.rows; r++) {
				for (int c = 0; c < matrix.cols; c++) {
					//printf("(");
					//for( int cn=0 ; cn<channels ; cn++){
					printf("%6d ", (int)matrix.at<uchar>(r, c)/*[cn]*/);
					//} printf(")");
				}
				printf("\n");
			}
		}

	}
};
#endif
// _OPENCV_KCFTRACKER_HPP_