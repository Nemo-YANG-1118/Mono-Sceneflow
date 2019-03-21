
#ifndef DEPTH_FIELD_OPTIMIZER_H_
#define DEPTH_FIELD_OPTIMIZER_H_

#include <opencv2/core.hpp>


#include <vector>
#include <map>
#include <string>


#include "../SceneConfiguration.h"
#include "../PhotoCamera.h"


namespace stereoscene{


	struct DepthFieldOptimizer{

	public:

		DepthFieldOptimizer(
			cv::Mat& _DepthMap ,
			cv::Mat& _DepthMask , 
			cv::Mat& _ErrorMap ,
			std::vector<float>& _DepthBox ,
			std::map<int , PhotoCamera>& _map_PCams , 
			std::map<int , cv::Mat>& _map_uFlow=std::map<int , cv::Mat>() ,
			std::map<int , cv::Mat>& _map_vFlow=std::map<int , cv::Mat>() , 
			cv::Mat& _RefImage=cv::Mat())
			: _DepthMap_(_DepthMap) ,
			_DepthMask_(_DepthMask) ,
			_ErrorMap_(_ErrorMap) ,
			_DepthBox_(_DepthBox) ,
			_map_PCams_(_map_PCams) ,
			_map_uFlow_(_map_uFlow) ,
			_map_vFlow_(_map_vFlow) ,
			_RefImage_(_RefImage) {}



		bool 
			operator() (
			const std::string &_sdir_Images ,
			const int KthRefPCam ,
			const bool BMetric_L1 , 
			const bool BStatisticError , 
			const int KScene , 
			const int NIters=2) ;



	private:

		cv::Mat& _DepthMap_ , &_DepthMask_ , &_ErrorMap_ ,  &_RefImage_ ;

		std::vector<float> &_DepthBox_ ;

		std::map<int , PhotoCamera>& _map_PCams_ ;



	private:

		int _NDepthDataType_ ;

		///PCameras and x3Points within the local bundle
		std::vector<PhotoCamera*> _vec_pbPCams_ ;

		std::vector<float> _vec_x3Points_ , _vec_x3Rays_ ;

		///optical flow between reference frame and compared frame
		std::map<int , cv::Mat> _map_uFlow_ , _map_vFlow_ ;



	private:

		bool
			approximateSceneFlow(
			const bool BSceneFlow_L1 ,
			const bool BInitiating , 
			const bool BStatisticError , 
			const std::string& _sdir_Image , 
			const int KthRefPCam) ;


		bool 
			denoiseDepthField(
			const std::string &sdir_Images ,
			const int KthRefPCam ,
			const bool BGWeighted) ;



		bool
			reconstructPoissonField(
			cv::Mat& dDepthMap ,
			const std::string& sdir_Save=std::string()) ;

	} ;


}


#endif