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
// GMPHD_MAF.h
#pragma once

#include "OnlineTracker.h"
// #include <iostream>
// #include <algorithm>
#include <vector>
#include <unordered_map>
// #include <list>
// //#include <map>
// #include <numeric>
// #include <functional>

// #include <boost/format.hpp>

#include <opencv2/core.hpp>
#include "local_types.hpp"
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>
// // #include <opencv2/opencv.hpp>

// #include "drawing.hpp"
// // #include "io_mots.hpp" // boost.

/**
* @brief	A Class for the GMPHD_MAF tracker
* @details	Input: images and object detection results / Output: tracking results of the GMPHD-MAF tracker
			The Kalman Filtering based Motion Estimation was added for Track-to-Track Association in Online Tracker Class (v 0.1.0)
* @author	Young-min Song
* @date		2021-06-16
* @version	0.1.0
*/
class GMPHD_MAF :
	public OnlineTracker

{
public:
	GMPHD_MAF();
	~GMPHD_MAF();

	//* Multiple Object Tracking and Segmentation (MOTS) *//
	int RunMOTS(const int& iFrmCnt, const cv::Mat& img, const std::vector<BBDet>& dets, std::vector<BBTrk>& out_tracks);	// 2D Instance Segmentation

	void Destory();
private:

	bool isInitialization;

	// Moved to OnlineTracker
	//std::vector<BBTrk> liveTrkVec;		// live tracks at time t (now)
	//std::vector<BBTrk> *liveTracksBatch;
	//std::vector<BBTrk> lostTrkVec;		// lost tracks at time t (was live tracks before time t)
	//std::vector<BBTrk> *lostTracksBatch;

	// Moved to OnlineTracker
	// The container for Group management, index (0:t-d-2, 1:t-d-1, 2:t-d), d: delayed time
	//std::unordered_map<int, std::vector<RectID>> *groupsBatch;

	std::unordered_map<int, std::vector<BBTrk>> tracks_reliable;
	std::unordered_map<int, std::vector<BBTrk>> tracks_unreliable;


private:
	// Init Containters, Matrices, and Utils
	// void InitializeTrackletsContainters(); // Moved to OnlineTracker
	void InitializeMatrices(cv::Mat &F, cv::Mat &Q, cv::Mat &Ps, cv::Mat &R, cv::Mat &H, int MODEL_TYPE = sym::MODEL_VECTOR::XY);

	// Segmentation Refinement
	std::vector<BBDet> MergeDetInstances(std::vector<BBDet>& obss, const bool& IS_MOTS = false, const float& sOCC_TH = 0.1);
	std::vector<BBTrk> MergeTrkInstances(std::vector<BBTrk>& stats, const float& sOCC_TH = 0.1);

	std::vector<int> BuildMergeGroupsMinID(const std::vector<BBTrk>& frmTracks, std::vector<std::vector<int>>& mergeIdxList);
	std::vector<int> BuildMergeGroupsMaxConfID(const std::vector<BBDet>& frmDets, std::vector<std::vector<int>>& mergeIdxList);
	void MergeSegMasksRects(const std::vector<cv::Mat>& in_masks, const std::vector<cv::Rect>& in_rects, cv::Mat& out_mask, cv::Rect& out_rect);
	int FindGroupMinIDRecursive(const int& i, std::vector<bool>& visitTable, const std::vector<BBTrk>& frmTracks, const std::vector<std::vector<int>>& mergeIdxList, const int& cur_min_idx);
	int FindGroupMaxConfIDRecursive(const int& i, std::vector<bool>& visitTable, const std::vector<BBDet>& frmDets, const std::vector<std::vector<int>>& mergeIdxList, const int& cur_min_idx);

	// Initialize states (tracks)
	void InitTracks(const int& iFrmCnt, const cv::Mat& img, const std::vector<BBDet>& dets, const int& MODEL_TYPE);

	// Prediction of state_k|k-1 from state_k-1 (x,y,vel_x,vel_y,width, height) using Kalman filter
	void PredictFrmWise(int iFrmCnt, const cv::Mat& img, std::vector<BBTrk>& stats, const int MODEL_TYPE);

	// Methods for D2T (FrmWise) and T2T (TrkWise) Association (Hierarchical Data Association)
	void DataAssocFrmWise(int iFrmCnt, const cv::Mat& img, std::vector<BBTrk>& stats, std::vector<BBDet>& obss, double P_survive = 0.99, const int &MODEL_TYPE = sym::MODEL_VECTOR::XY);
	void DataAssocTrkWise(int iFrmCnt, cv::Mat& img, std::vector<BBTrk>& stats_lost, std::vector<BBTrk>& obss_live, const int &MODEL_TYPE = sym::MODEL_VECTOR::XY);

	// Affinity (Cost) Calculation
	float FrameWiseAffinity(BBDet ob, BBTrk& stat_temp, const int MODEL_TYPE);
	float TrackletWiseAffinity(BBTrk &stat_pred, const BBTrk& obs, const int MODEL_TYPE);
	float TrackletWiseAffinityKF(BBTrk &stat_pred, const BBTrk& obs, const int MODEL_TYPE);

	// Score-level Fussion based Affinity Calcuation
	void FusionMinMaxNorm(const int& nObs, const int& mStats,
		std::vector<std::vector<double>> &m_cost, std::vector<std::vector<double>>& gmphd_cost, std::vector<std::vector<double>>& kcf_cost, const std::vector<std::vector<float>>& IOUs,
		const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON = false);

	std::vector<double> FusionZScoreNorm(const int& nObs, const int& mStats,
		std::vector<std::vector<double>> &m_cost, std::vector<std::vector<double>>& gmphd_cost, std::vector<std::vector<double>>& kcf_cost, const std::vector<std::vector<float>>& IOUs, double conf_interval,
		const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON = false);

	void FusionDblSigmoidNorm(const int& nObs, const int& mStats,
		std::vector<std::vector<double>> &m_cost, std::vector<std::vector<double>>& gmphd_cost, std::vector<std::vector<double>>& kcf_cost, const std::vector<std::vector<float>>& IOUs,
		const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON = false);

	void FusionTanHNorm(const int& nObs, const int& mStats,
		std::vector<std::vector<double>> &m_cost, std::vector<std::vector<double>>& gmphd_cost, std::vector<std::vector<double>>& kcf_cost, const std::vector<std::vector<float>>& IOUs,
		const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON = false);

	void GetMeanStdDev(const int& nObs, const int& mStats, \
		std::vector<std::vector<double>>&gmphd_cost, double &mean_gmphd, double& stddev_gmphd,
		std::vector<std::vector<double>>&kcf_cost, double &mean_kcf, double& stddev_kcf);

	// Tracklets Managements (Categorization, state transition, memory deallocation..)
	void ArrangeTargetsVecsBatchesLiveLost();
	void PushTargetsVecs2BatchesLiveLost();
	void ClassifyTrackletReliability(int iFrmCnt, std::unordered_map<int, std::vector<BBTrk>>& tracksbyID, std::unordered_map<int, std::vector<BBTrk>>& reliables, std::unordered_map<int, std::vector<BBTrk>>& unreliables);
	void ClassifyReliableTracklets2LiveLost(int iFrmCnt, const std::unordered_map<int, std::vector<BBTrk>>& reliables, std::vector<BBTrk>& liveReliables, std::vector<BBTrk>& LostReliables, std::vector<BBTrk>& obss);
	void ArrangeRevivedTracklets(std::unordered_map<int, vector<BBTrk>>& tracks, std::vector<BBTrk>& lives);

	// Return the Tracking Results
	/// Return live and reliable tracks
	std::vector<BBTrk> GetTrackingResults(const int& iFrmCnt, std::vector<BBTrk>& liveReliables, const int& MODEL_TYPE);

private:
	cv::Mat F_xy;		// transition matrix state_t-1 to state_t 	
	cv::Mat Q_xy;		// process noise covariance
	cv::Mat Ps_xy;		// covariance of states's Gaussian Mixtures for Survival
	cv::Mat R_xy;		// the covariance matrix of measurement
	cv::Mat H_xy;		// transition matrix state_t to observation_t

	double P_survive = P_SURVIVE_LOW;		// Probability of Survival	(User Parameter)(Constant)
	double P_survive_mid = P_SURVIVE_MID;
};


