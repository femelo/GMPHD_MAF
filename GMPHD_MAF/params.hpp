/*
BSD 2-Clause License

Copyright (c) 2021, Young-min Song,
Machine Learning and Vision Lab (https://sites.google.com/view/mlv/),
Gwangju Institute of Science and Technology(GIST), South Korea.
All rights reserved.

This software is an implementation of the GMPHD_MAF tracker,
which not only refers to the paper entitled
"Online Multi-Object Tracking and Segmentation with GMPHD Filter and Mask-Based Affinity Fusion"
but also is available at https://github.com/SonginCV/GMPHD_MAF.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


// user-defined pre-processor,...
// user-defined containters, data-type,..

// #include <opencv2\opencv.hpp>

#ifndef PARAMS_HPP
#define PARAMS_HPP

// IOU between 3D Boxes
#include <boost/geometry.hpp>
// #include <boost/geometry/geometries/point_xy.hpp>
// #include <boost/geometry/geometries/polygon.hpp>
// #include <boost/geometry/geometries/adapted/c_array.hpp>

#include <unordered_map>
//#include "siamRPN_tracker.hpp"
//#include "DaSiamTracker.h"

#define MAX_OBJECTS	27	

#define SIZE_CONSTRAINT_RATIO	2

// Parameters for the GMPHD filter
#define PI_8				3.14159265
#define e_8				2.71828182
#define T_th				(0.0)
#define W_th				(0.0)
#define Q_TH_LOW_20			0.00000000000000000001f		// 0.1^20
#define Q_TH_LOW_20_INVERSE		100000000000000000000.0f	// 10^20
#define Q_TH_LOW_15			0.00000000000001f		// 0.1^15
#define Q_TH_LOW_15_INVERSE		1000000000000000.0f		// 10^15
#define Q_TH_LOW_10			0.0000000001f			// 0.1^10
#define Q_TH_LOW_10_INVERSE		10000000000.0f			// 10^10
#define Q_TH_LOW_8			0.00000001f			// 0.1^8
#define Q_TH_LOW_8_INVERSE		100000000.0f			// 10^8
#define P_SURVIVE_LOW			0.99				// object number >=2 : 0.99, else 0.95
#define P_SURVIVE_MID			0.95
#define VAR_X				25 //42.96 // 279.13 // 1116.520// 0.1 //25 // 171.87
#define VAR_Y				100//29.27// 47.685 // 190.740 //0.1 //100 // 117.09
#define VAR_D				25//0.1 //25
#define VAR_R				0.09 // 0.1, 0.04, 0.01
#define VAR_X_VEL			25// 42.96 // 279.13 // 1116.520// 0.1 //25 // 171.871
#define VAR_Y_VEL			100// 29.27//47.685 // 190.740//0.1//100 // 117.090
#define VAR_D_VEL			25//0.1 // 25	// 1, 4, 9, 16, 25, 100, 225, 400, 900
#define VAR_R_VEL			100 // 0.1, 0.04, 0.01
#define VAR_WIDTH			100
#define VAR_HEIGHT			400
#define VAR_X_MID			100
#define VAR_Y_MID			400
#define VAR_X_VEL_MID			100
#define VAR_Y_VEL_MID			400

//#define VELOCITY_UPDATE_ALPHA	0.4f // moving camera scene with low fps ( <= 15)
#define CONFIDENCE_UPDATE_ALPHA	0.95f

// Parameters for Data Association
#define TRACK_ASSOCIATION_FRAME_DIFFERENCE	0	// 0: equal or later, 1:later
#define ASSOCIATION_STAGE_1_GATING_ON		0
#define ASSOCIATION_STAGE_2_GATING_ON		0
#define AFFINITY_COST_L1_OR_L2_NORM_FW		1	// 0: L1-norm, 1: L2-norm, frame-wise
#define AFFINITY_COST_L1_OR_L2_NORM_TW		1	// 0: L1-norm, 1: L2-norm, tracklet-wise
// SOT Tracker Options
#define SOT_USE_KCF_TRACKER			1
#define SOT_USE_SIAMRPN_TRACKER			2
#define SOT_USE_DASIAMRPN_TRACKER		3
#define SOT_TRACK_OPT				SOT_USE_KCF_TRACKER

// (D2TA) GMPHD-KCF Fusion (D2TA)
#define APPEARANCE_STRICT_UPDATE_ON		0
#define APPEARANCE_UPDATE_DET_TH		0.85f

// (T2TA) GMPHD-KCF Fusion (T2TA)
//#define SAF_MASK_T2TA_ON			1

// #define KCF_SOT_T2TA_ON			0
//#define IOU_UPPER_TH_T2TA_ON			1

// MOTS
#define COV_UPDATE_T2TA				0

#define USE_LINEAR_MOTION			0
#define USE_KALMAN_MOTION			1
#define T2TA_MOTION_OPT				USE_LINEAR_MOTION			
#define KALMAN_INIT_BY_LINEAR			1

// Parameters for Occlusion Group Management (Merge and Occlusion Group Energy Minimization)
#define MERGE_DET_ON			1
#define MERGE_TRACK_ON			1
#define MERGE_METRIC_OPT		2	// 0: SIOA, 1:IOU, 2:sIOU
#define MERGE_THRESHOLD_RATIO	0.4f
#define MERGE_METRIC_SIOA		0
#define MERGE_METRIC_IOU		1
#define MERGE_METRIC_mIOU		2

#define GROUP_MANAGEMENT_FRAME_WISE_ON	1

#define USE_GMPHD_NAIVE			0
#define USE_GMPHD_HDA			1
#define GMPHD_TRACKER_MODE		USE_GMPHD_HDA

// Tracking Results Writing Option
#define EXCLUDE_BBOX_OUT_OF_FRAME	0

// Visualization Option (main function)
#define VISUALIZATION_MAIN_ON		1
#define SKIP_FRAME_BY_FRAME		0
#define VISUALIZATION_RESIZE_ON		0
#define BOUNDING_BOX_THICK		5
#define TRAJECTORY_THICK		5
#define	ID_CONFIDENCE_FONT_SIZE		2
#define	ID_CONFIDENCE_FONT_THICK	2
#define FRAME_COUNT_FONT_SIZE		3
#define FRAME_COUNT_FONT_THICK		3
// Visualization Option (tracker class)
#define VIS_D2TA_DETAIL			0
#define VIS_T2TA_DETAIL			0
#define VIS_TRACK_MERGE			0

// Image Save Option (main function)
#define SAVE_IMG_ON				0

// Dataset Indices
#define DB_TYPE_MOT15			0	// MOT Challenge 2015 Dataset (-1)
#define DB_TYPE_MOT17			1	// MOT Challenge 2017 Dataset
#define DB_TYPE_CVPR19			2   // MOT Challenge 2019 (CVPR 2019) Dataset
#define DB_TYPE_MOT20			2   // MOT Challenge 2020
#define DB_TYPE_KITTI			3	// KITTI 	(2D box, 3D box, 3D Point Cloud)
#define DB_TYPE_KITTI_MOTS		4	// KITTI-MOTS 	(2D Instance Segment)
#define DB_TYPE_MOTS20			5	// MOTS20	(2D Instacne Segment)

#define DEBUG_PRINT				0

/*--------------------------------------------------------------------------------------*/

// If you define functions and initialize variables in a header file,
// and want to use those functions/variables from other source files, the only way is to use `static`.
// Alternatively, declare them in the header file, define and initialize them in a .cpp file,
// and then use the `extern` keyword.
// Reference: https://zerobell.tistory.com/22
// Anyway, when the value doesn't change, using `static` is the most convenient.
// For functions, `extern` is implicitly assumed (the same applies to global variables within a file).

namespace sym {
	enum MODEL_VECTOR {
		XY, XYWH, XYRyx
	};

	static int DIMS_STATE[3] = { 4, 6, 6};
	static int DIMS_OBS[3] = { 2, 4, 3};

	enum OBJECT_TYPE {
		CAR, VAN,
		PEDESTRIAN, PERSON_SITTING,
		CYCLIST, TRUCK, TRAM,
		MISC, DONTCARE, BUS
	};
	static std::string OBJECT_STRINGS[10] = \
	{ "Car", "Van",
		"Pedestrian", "Person_sitting",
		"Cyclist", "Truck", "Tram",
		"Misc", "DontCare", "Bus" };

	static std::string DB_NAMES[6] = { "MOT15", "MOT17", "MOT20", "KITTI", "KITTI", "MOTS20" };
	static int FRAME_OFFSETS[6] = { 1,1,1,0,0,1 };

	static cv::Scalar OBJECT_TYPE_COLORS[9] = {
		cv::Scalar(255, 255, 0),	/*Car: Lite Blue (Mint)*/
		cv::Scalar(255, 0, 0),	/*Van: Blue*/
		cv::Scalar(255, 0, 255),	/*Pedestrian: Pink*/
		cv::Scalar(0, 0, 255),	/*Person_sitting: Red*/
		cv::Scalar(42, 42, 165),	/*Cyclist: Brown*/
		cv::Scalar(0, 255, 0),	/*Truck: Lite Green*/
		cv::Scalar(0, 255, 255),	/*Tram: Yellow*/
		cv::Scalar(64, 64, 64),	/*Misc: Gray*/
		cv::Scalar(0, 0, 0)
	};	/*DontCare: Black*/

	// Parameters: 11*10*7 = 770 cases * num of scenes * num of detectors * num of object classes, too many.
	static float DET_SCORE_THS[15] = { -100.0, 0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0 };
	//static float DET_SCORE_THS[11] = { 0.0,0.52,0.54,0.55,0.56,0.58,0.62,0.64,0.65,0.66,0.68};
	static int TRACK_INIT_MIN[10] = { 1,2,3,4,5,6,7,8,9,10 };
	static int TRACK_T2TA_MAX[8] = { 5,10,15,20,30,60,80,100 };
	static float VEL_UP_ALPHAS[12] = { 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 };
	static float MERGE_THS[8] = { 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 };
	static float DET_SCORE_THS_LOWER[7] = { 0.0,0.1,0.2,0.3,0.4,0.5,0.6 };

	static std::string MERGE_METRICS_STR[3] = { "SIOA","IOU", "mIOU" };

	enum AFFINITY_OPT { GMPHD, KCF, MAF };
}

#define IS_DONTCARE(x) (((int)x)==sym::OBJECT_TYPE::DONTCARE)
#define IS_VEHICLE_EVAL(x) (((int)x==sym::OBJECT_TYPE::CAR) ||((int)x==sym::OBJECT_TYPE::VAN))
#define IS_VEHICLE_ALL(x) (((int)x==sym::OBJECT_TYPE::CAR) ||((int)x==sym::OBJECT_TYPE::VAN)||((int)x==sym::OBJECT_TYPE::TRUCK)||((int)x==sym::OBJECT_TYPE::BUS))
#define IS_PERSON_KITTI(x) (((int)x==sym::OBJECT_TYPE::PEDESTRIAN) ||((int)x==sym::OBJECT_TYPE::PERSON_SITTING))
#define IS_PERSON_EVAL(x) (((int)x==sym::OBJECT_TYPE::PEDESTRIAN) ||((int)x==sym::OBJECT_TYPE::PERSON_SITTING)||((int)x==sym::OBJECT_TYPE::CYCLIST))


typedef struct MOTparams {
	/*TARGET CLASS*/
	int OBJ_TYPE = sym::OBJECT_TYPE::PEDESTRIAN;
	/*PARAMS*/
	float DET_MIN_CONF = 0.0f; // upper bound
	int TRACK_MIN_SIZE = 1;
	int QUEUE_SIZE = 10;
	int FRAMES_DELAY_SIZE = TRACK_MIN_SIZE - 1;
	int T2TA_MAX_INTERVAL = 10;
	float VEL_UP_ALPHA = 0.5;
	int MERGE_METRIC = MERGE_METRIC_OPT;
	float MERGE_RATIO_THRESHOLD = MERGE_THRESHOLD_RATIO;
	int GROUP_QUEUE_SIZE = TRACK_MIN_SIZE * 10;
	int FRAME_OFFSET = 0;
	/*Gating */
	bool GATE_D2TA_ON = true;
	bool GATE_T2TA_ON = true;

	/*Simple Affinity Fusion*/
	/// 0: GMPHD (Position and Motion) only, 1: KCF (Appearance) Only, 2: Simple Affinity Fuion On
	int SAF_D2TA_MODE = 2;
	int SAF_T2TA_MODE = 2;
	bool SAF_MASK_D2TA = true; // compute appearance considering pixel-wise mask
	bool SAF_MASK_T2TA = true;
	cv::Vec2f KCF_BOUNDS_D2TA;
	cv::Vec2f KCF_BOUNDS_T2TA;
	cv::Vec2f IOU_BOUNDS_D2TA;
	cv::Vec2f IOU_BOUNDS_T2TA;
	/*Simple Affinity Fusion*/

	MOTparams() {};
	MOTparams(int obj_type, float dConf_th, int trk_min, int t2ta_max, int mg_metric, float mg_th, float vel_alpha, int group_q_size, int frames_offset,
		int saf_d2ta = 2, int saf_t2ta = 2, bool mask_d2ta = true, bool mask_t2ta = true,
		const cv::Vec2f& kcf_bounds_d2ta = cv::Vec2f(0.5, 0.9), const cv::Vec2f& kcf_bounds_t2ta = cv::Vec2f(0.5, 0.9),
		const cv::Vec2f& iou_bounds_d2ta = cv::Vec2f(0.1, 0.9), const cv::Vec2f& iou_bounds_t2ta = cv::Vec2f(0.1, 0.9),
		const cv::Vec2b& GATES_ONOFF = cv::Vec2b(true, true)) {

		this->OBJ_TYPE = obj_type;

		this->DET_MIN_CONF = dConf_th;
		this->TRACK_MIN_SIZE = trk_min;
		this->FRAMES_DELAY_SIZE = trk_min - 1;
		this->T2TA_MAX_INTERVAL = t2ta_max;
		this->VEL_UP_ALPHA = vel_alpha;
		this->MERGE_METRIC = mg_metric;
		this->MERGE_RATIO_THRESHOLD = mg_th;
		this->GROUP_QUEUE_SIZE = group_q_size;
		this->FRAME_OFFSET = frames_offset;

		this->GATE_D2TA_ON = GATES_ONOFF[0];
		this->GATE_T2TA_ON = GATES_ONOFF[1];

		this->SAF_D2TA_MODE = saf_d2ta;
		this->SAF_T2TA_MODE = saf_t2ta;
		this->SAF_MASK_D2TA = mask_d2ta;
		this->SAF_MASK_T2TA = mask_t2ta;

		this->KCF_BOUNDS_D2TA = kcf_bounds_d2ta;
		this->KCF_BOUNDS_T2TA = kcf_bounds_t2ta;
		this->IOU_BOUNDS_D2TA = iou_bounds_d2ta;
		this->IOU_BOUNDS_T2TA = iou_bounds_t2ta;
	}

	MOTparams& operator=(const MOTparams& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->OBJ_TYPE = copy.OBJ_TYPE;
		this->DET_MIN_CONF = copy.DET_MIN_CONF;
		this->TRACK_MIN_SIZE = copy.TRACK_MIN_SIZE;
		this->FRAMES_DELAY_SIZE = copy.FRAMES_DELAY_SIZE;
		this->T2TA_MAX_INTERVAL = copy.T2TA_MAX_INTERVAL;
		this->VEL_UP_ALPHA = copy.VEL_UP_ALPHA;
		this->MERGE_METRIC = copy.MERGE_METRIC;
		this->MERGE_RATIO_THRESHOLD = copy.MERGE_RATIO_THRESHOLD;
		this->GROUP_QUEUE_SIZE = copy.GROUP_QUEUE_SIZE;
		this->FRAME_OFFSET = copy.FRAME_OFFSET;

		this->GATE_D2TA_ON = copy.GATE_D2TA_ON;
		this->GATE_T2TA_ON = copy.GATE_T2TA_ON;

		this->SAF_D2TA_MODE = copy.SAF_D2TA_MODE;
		this->SAF_T2TA_MODE = copy.SAF_T2TA_MODE;
		this->SAF_MASK_D2TA = copy.SAF_MASK_D2TA;
		this->SAF_MASK_T2TA = copy.SAF_MASK_T2TA;

		this->KCF_BOUNDS_D2TA = copy.KCF_BOUNDS_D2TA;
		this->KCF_BOUNDS_T2TA = copy.KCF_BOUNDS_T2TA;
		this->IOU_BOUNDS_D2TA = copy.IOU_BOUNDS_D2TA;
		this->IOU_BOUNDS_T2TA = copy.IOU_BOUNDS_T2TA;

		return *this;
	}
} GMPHDMAFparams;

#endif // PARAMS_HPP
