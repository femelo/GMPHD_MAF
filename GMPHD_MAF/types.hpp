#ifndef TYPES_HPP
#define TYPES_HPP


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>	// Kalman Filtering
#include <opencv2/core/utility.hpp>

// #include "VOT.h"
#include "kcftracker.hpp"


typedef struct boundingbox_id {
	int id;
	int idx;
	int min_id;		// minimum value ID in a group (is considered as group ID)
	double weight = 0.0;
	cv::Rect rec;	// t, t-1, t-2
	boundingbox_id() {
	}
	boundingbox_id(int id, cv::Rect occRect = cv::Rect()) :id(id) {
		// Deep Copy
		id = id;
		rec = occRect;
	}
	boundingbox_id& operator=(const boundingbox_id& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->idx = copy.idx;
		this->id = copy.id;
		this->min_id = copy.min_id;
		this->rec = copy.rec;
		this->weight = copy.weight;
	}
	bool operator<(const boundingbox_id& rect) const {
		return (id < rect.id);
	}
} RectID;

typedef struct track_id_bb_mask {
	int id;
	int fn;
	int object_type;
	cv::Rect bb;
	cv::Mat mask;

	//track_id_bb_mask() {

	//}
	//track_id_bb_mask(int id_, int fn_, cv::Rect bb, cv::Mat mask_=cv::Mat()) {
	//	this->id = id_;
	//	this->fn = fn_;
	//	this->bb = bb;

	//	this->mask = mask_.clone();
	//}

	track_id_bb_mask& operator=(const track_id_bb_mask& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->id = copy.id;
		this->fn = copy.fn;
		this->object_type = copy.object_type;
		this->bb = copy.bb;

		if (!this->mask.empty()) this->mask.release();
		this->mask = copy.mask.clone();

		return *this;
	}
} track_info;

typedef struct bbTrack {
	//VOT *papp ;
	KCFTracker papp;
	cv::Mat segMask;
	std::string segMaskRle;
	std::vector<float> reid_features;

	int fn;
	int id;
	int id_associated = -1; // (index) it is able to be used in Tracklet-wise association
	/*----MOTS Tracking Only---*/
	int det_id;
	float det_confidence;
	/*----MOTS Tracking Only---*/
	int fn_latest_T2TA = 0;
	cv::Rect rec;
	cv::Rect rec_corr;
	float rec6D[6];
	std::vector<cv::Vec3f> top_corners;
	std::vector<cv::Vec3f> bottom_corners;
	float vx;
	float vy;
	float vd;
	float vr;
	float weight;
	float conf;
	cv::KalmanFilter KF;
	cv::Mat cov;
	cv::Mat tmpl;
	cv::Mat hist;
	float density;
	float depth;
	float ratio_yx;
	float rotation_y;
	bool isNew;
	bool isAlive;
	bool isMerged = false;
	int isInterpolated = 0; // 0: Online, 1: Interpolated, -1:Ignored (Falsely Interpolated)
	int iGroupID = -1;
	bool isOcc = false;
	int size = 0;
	int objType;
	std::vector<RectID> occTargets;

	bbTrack() {}

	bbTrack(int fn, int id, int isOcc, cv::Rect rec, cv::Mat obj = cv::Mat(), cv::Mat hist = cv::Mat()) :
		fn(fn), id(id), isOcc(isOcc), rec(rec) {
		if (!hist.empty()) {
			this->hist.release();
			this->hist = hist.clone(); // deep copy
		}
		else {
			//this->hist.release();
			//printf("[ERROR]target_bb's parameter \"hist\" is empty!\n");
			this->hist = hist;
		}
		if (!obj.empty()) {
			this->tmpl.release();
			this->tmpl = obj.clone(); // deep copy
		}
		else {
			//this->obj_tmpl.release();
			this->tmpl = obj;
		}
		//isOccCorrNeeded = false; // default

	}

	bool operator<(const bbTrack& trk) const {
		return (id < trk.id);
	}

	bbTrack& operator=(const bbTrack& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		//this->KCFfeatures = copy.KCFfeatures;
		this->papp = copy.papp;
		if (!this->segMask.empty()) this->segMask.release();
		if (!copy.segMask.empty())	this->segMask = copy.segMask;
		if (!this->reid_features.empty()) this->reid_features.clear();
		this->reid_features = copy.reid_features;
		
		this->fn = copy.fn;
		this->id = copy.id;
		this->id_associated = copy.id_associated;
		this->size = copy.size;
		this->fn_latest_T2TA = copy.fn_latest_T2TA;
		this->rec = copy.rec;
		this->rec6D[0] = copy.rec6D[0];
		this->rec6D[1] = copy.rec6D[1];
		this->rec6D[2] = copy.rec6D[2];
		this->rec6D[3] = copy.rec6D[3];
		this->rec6D[4] = copy.rec6D[4];
		this->rec6D[5] = copy.rec6D[5];
		this->top_corners = copy.top_corners;
		this->bottom_corners = copy.bottom_corners;
		this->vx = copy.vx;
		this->vy = copy.vy;
		this->vd = copy.vd;
		this->vr = copy.vr;
		this->rec_corr = copy.rec_corr;
		this->density = copy.density;
		this->depth = copy.depth;
		this->rotation_y = copy.rotation_y;
		this->ratio_yx = copy.ratio_yx;
		this->isNew = copy.isNew;
		this->isAlive = copy.isAlive;
		this->isMerged = copy.isMerged;
		this->isInterpolated = copy.isInterpolated;
		this->iGroupID = copy.iGroupID;
		this->isOcc = copy.isOcc;
		this->weight = copy.weight;
		this->conf = copy.conf;
		this->objType = copy.objType;
		this->segMaskRle = copy.segMaskRle;

		if (!this->cov.empty()) this->cov.release();
		if (!this->tmpl.empty()) this->tmpl.release();
		if (!this->hist.empty()) this->hist.release();
	
		this->cov = copy.cov.clone();
		this->tmpl = copy.tmpl.clone();
		this->hist = copy.hist.clone();

		/*----MOTS Tracking Only---*/
		this->det_id = copy.det_id;
		this->det_confidence = copy.det_confidence;
		/*----MOTS Tracking Only---*/

		return *this;
	}

	void CopyTo(bbTrack& dst) {
		dst.papp = this->papp;
		if (!this->segMask.empty()) dst.segMask = this->segMask.clone();
		if (!this->reid_features.empty()) dst.reid_features = this->reid_features;

		dst.fn = this->fn;
		dst.id = this->id;
		dst.id_associated = this->id_associated;
		dst.rec = this->rec;
		dst.rec6D[0] = this->rec6D[0];
		dst.rec6D[1] = this->rec6D[1];
		dst.rec6D[2] = this->rec6D[2];
		dst.rec6D[3] = this->rec6D[3];
		dst.rec6D[4] = this->rec6D[4];
		dst.rec6D[5] = this->rec6D[5];
		dst.top_corners = this->top_corners;
		dst.bottom_corners = this->bottom_corners;
		dst.vx = this->vx;
		dst.vy = this->vy;
		dst.vd = this->vd;
		dst.vr = this->vr;
		dst.rec_corr = this->rec_corr;
		dst.density = this->density;
		dst.depth = this->depth;
		dst.rotation_y = this->rotation_y;
		dst.isNew = this->isNew;
		dst.isAlive = this->isAlive;
		dst.isMerged = this->isMerged;
		dst.isInterpolated = this->isInterpolated;
		dst.iGroupID = this->iGroupID;
		dst.isOcc = this->isOcc;
		dst.weight = this->weight;
		dst.conf = this->conf;
		dst.objType = this->objType;
		dst.segMaskRle = this->segMaskRle;

		if (!this->cov.empty()) dst.cov = this->cov.clone();
		if (!this->tmpl.empty()) dst.tmpl = this->tmpl.clone();
		if (!this->hist.empty()) dst.hist = this->hist.clone();

		/*----MOTS Tracking Only-- - */
		dst.det_id = this->det_id;
		dst.det_confidence = this->det_confidence;
		/*----MOTS Tracking Only---*/
	}

	void Destroy() {
		if (!this->cov.empty()) this->cov.release();
		if (!this->tmpl.empty()) this->tmpl.release();
		if (!this->hist.empty()) this->hist.release();
	}

	void InitKF(int nStats, int mObs, float x_var, float y_var) {
		this->KF = cv::KalmanFilter(nStats, mObs, 0);

		// F, constant
		setIdentity(KF.transitionMatrix);
		this->KF.transitionMatrix.at<float>(0, 2) = 1;
		this->KF.transitionMatrix.at<float>(1, 3) = 1;

		// H, constant
		this->KF.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

		setIdentity(this->KF.processNoiseCov, cv::Scalar::all(x_var / 2));	// Q, constant, 4^2/2, 4x4
		this->KF.processNoiseCov.at<float>(1, 1) = y_var / 2;
		this->KF.processNoiseCov.at<float>(3, 3) = y_var / 2;
		//cout << this->KF.processNoiseCov.rows << "x" << this->KF.processNoiseCov.cols << endl;

		setIdentity(this->KF.measurementNoiseCov, cv::Scalar::all(x_var));	// R, constant, 4^2, 2x2
		this->KF.measurementNoiseCov.at<float>(1, 1) = y_var;
		//cout << this->KF.measurementNoiseCov.rows << "x" << this->KF.measurementNoiseCov.cols << endl;

		setIdentity(this->KF.errorCovPre, cv::Scalar::all(1));				// P_t|t-1, predicted from P_t-1
		setIdentity(this->KF.errorCovPost, cv::Scalar::all(1));				// P_0 or P_t,	just for init (variable)
	}
} BBTrk;

typedef struct bbDet {
	int fn;
	int object_type = -1;
	cv::Rect rec;
	float depth;		// used to represent the 2.5D Bounding Box (2D + Depth from Point Cloud)
	float rotation_y;
	float ratio_yx;
	float rec6D[6];
	std::vector<cv::Vec3f> top_corners;
	std::vector<cv::Vec3f> bottom_corners;
	std::vector<float> distances;
	float confidence;
	float weight;		// normalization value of confidence at time t
	int id;				// Used in Looking Back Association
	int input_type;		// 2D Box, 3D Box, 3D Point Cloud, 2D Instance 
	cv::Mat segMask;
	std::string segMaskRle;
	std::vector<float> reid_features;

	bbDet& operator=(const bbDet& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		if (!this->segMask.empty()) this->segMask.release();
		this->segMask = copy.segMask;
		this->segMaskRle = copy.segMaskRle;

		if (!this->reid_features.empty()) this->reid_features.clear();
		this->reid_features = copy.reid_features;
		if (!this->distances.empty()) this->distances.clear();
		this->distances = copy.distances;

		this->fn = copy.fn;
		this->object_type = copy.object_type;
		this->rec = copy.rec;
		this->rec6D[0] = copy.rec6D[0];
		this->rec6D[1] = copy.rec6D[1];
		this->rec6D[2] = copy.rec6D[2];
		this->rec6D[3] = copy.rec6D[3];
		this->rec6D[4] = copy.rec6D[4];
		this->rec6D[5] = copy.rec6D[5];

		if (!this->top_corners.empty()) this->top_corners.clear();
		this->top_corners = copy.top_corners;
		if (!this->bottom_corners.empty()) this->bottom_corners.clear();
		this->bottom_corners = copy.bottom_corners;

		this->depth = copy.depth;
		this->rotation_y = copy.rotation_y;
		this->ratio_yx = copy.ratio_yx;
		this->confidence = copy.confidence;

		this->weight = copy.weight;
		this->id = copy.id;
		this->input_type = copy.input_type;

		return *this;
	}
} BBDet;

typedef std::vector<std::vector<std::vector<std::vector<float>>>> VECx4xFLT;
typedef std::vector<std::vector<std::vector<float>>> VECx3xFLT;
typedef std::vector<std::vector<std::vector<cv::Mat>>> VECx3xMAT;
typedef std::vector<std::vector<std::vector<std::string>>> VECx3xSTR;
typedef std::vector<std::vector<std::vector<BBDet>>> VECx3xBBDet;
typedef std::vector<std::vector<BBDet>> VECx2xBBDet;
typedef std::vector<std::vector<std::vector<BBTrk>>> VECx3xBBTrk;
typedef std::vector<std::vector<BBTrk>> VECx2xBBTrk;

#endif // TYPES_HPP
