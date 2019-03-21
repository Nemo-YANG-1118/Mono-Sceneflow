
#ifndef DEPTH_FIELD_INTEGRATOR_H_
#define DEPTH_FIELD_INTEGRATOR_H_

#include <opencv2/core.hpp>

#include <tuple>
#include <vector>
#include <map>

#include "DepthFieldCommon/PhotoCamera.h"


namespace stereoscene{


	struct DepthFieldIntegrator{

	public:

		DepthFieldIntegrator(
			std::map<int , PhotoCamera>& _map_PCams ,
			cv::Mat& _CurDepthMap , 
			cv::Mat& _CurDepthMask , 
			std::vector<float>& _CurBoundBox , 
			std::map<int , cv::Mat>& _map_uCurFlow , 
			std::map<int , cv::Mat>& _map_vCurFlow) 
			: _map_PCams_(_map_PCams) ,
			_CurDepthMap_(_CurDepthMap) ,
			_CurDepthMask_(_CurDepthMask) , 
			_CurDepthBox_(_CurBoundBox) ,
			_map_uCurFlow_(_map_uCurFlow) ,
			_map_vCurFlow_(_map_vCurFlow) {}



		bool
			operator() (
			std::map<int , std::tr1::tuple<PhotoCamera , cv::Mat , cv::Mat>>& _map_DepthFields ,
			const cv::Mat& CurImage , 
			const int KthCurPCam ,
			const int KthFormPCam=0) ;



	private:

		std::map<int , PhotoCamera>& _map_PCams_ ;

		std::vector<float> &_CurDepthBox_ ;

		cv::Mat &_CurDepthMap_ , &_CurDepthMask_ ;

		std::map<int , cv::Mat>& _map_uCurFlow_ , &_map_vCurFlow_ ;



	private:

		cv::Mat _InternFormDepthMap_ , _InternFormDepthMask_  ;

		std::vector<float> _InternFormDepthBox_ ;

		std::vector<int> _x2ExternFormPoints_ ;

		std::vector<float> _x2ExternCurPoints_, _x2ExternDepth_ , _x3MapPoints_ ;


	private:

		bool
			integrateDepthMaps(
			const int NBorderPad ,
			const cv::Mat &_CurImage,
			const bool BDoBilateralFilter=false) ;


		bool 
			remapFormerDepth(
			std::tr1::tuple<PhotoCamera , cv::Mat , cv::Mat>& _FormDepthFields ,
			const int KthCurPCam , 
			const int NBorderPad) ;


	} ;


}


#endif