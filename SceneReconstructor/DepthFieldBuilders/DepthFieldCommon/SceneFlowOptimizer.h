
#ifndef SCENE_FLOW_DRIVER_H_
#define SCENE_FLOW_DRIVER_H_

#include <opencv2/core.hpp>


#include <string>
#include <vector>
#include <map>


#include "PhotoCamera.h"


namespace stereoscene{


	struct SceneFlowOptimizer{

	public:

		enum{SCENEFLOW_L2=0 , SCENEFLOW_L1=1};


		SceneFlowOptimizer(
			std::vector<PhotoCamera*>& _vec_pPCams ,
			std::vector<float>& _vec_x3Points , 
			std::vector<float>& _vec_x3Rays , 
			std::map<int , cv::Mat>& _map_uFlow ,
			std::map<int , cv::Mat>& _map_vFlow ,
			cv::Mat& _DepthMap , 
			cv::Mat& _ErrorMap ,
			cv::Mat& _DepthMask ,
			const std::vector<float>& _DepthBox)
			: _vec_pPCams_(_vec_pPCams) ,
			_vec_x3Points_(_vec_x3Points) ,
			_vec_x3Rays_(_vec_x3Rays) ,
			_map_uFlow_(_map_uFlow) , 
			_map_vFlow_(_map_vFlow) ,
			_DepthMap_(_DepthMap) ,
			_ErrorMap_(_ErrorMap) ,
			_DepthMask_(_DepthMask) , 
			_DepthBox_(_DepthBox) {}


		bool
			operator() (
			const int OPTIMIZER_TYPE , 
			const bool BInitiating , 
			const bool BStatisticError , 
			const std::string& _sdir_Image , 
			const int KthRefPCam , 
			cv::Mat& _RefImage=cv::Mat() ,
			const bool BBoundOptFlow=true  ,
			const double _DTopBoundRatio=20.0 , 
			const double _DBottomBoundRatio=20.0 ) ;



	private:

		std::vector<float>& _vec_x3Points_ , &_vec_x3Rays_ ;

		std::vector<PhotoCamera*>& _vec_pPCams_ ;

		std::map<int , cv::Mat>& _map_uFlow_ , &_map_vFlow_ ;

		cv::Mat& _DepthMap_ , &_ErrorMap_ ;

		cv::Mat& _DepthMask_ ;

		const std::vector<float>& _DepthBox_ ;



	private:

		bool
			approximateMotionField_CUDA() ;


		bool
			approximateMotionField_L1(
			const cv::Mat& _GrayRefImage) ;


		bool
			approximateMotionField_L2() ;


		bool
			boundDenseDisparity(
			const double DTopRatio ,
			const double DBottomRatio) ;


		bool
			calculateDenseDisparity(
			std::map<int , cv::Mat>& _map_sImages , 
			const int KUseMethod=10) ; ///10 refer to brox , 11 refer to tvl1 , 12 refer to conjuncture



	public:

		static bool
			triangulateDepthMap(
			std::vector<float>& x3Points ,
			std::vector<float>& x3Rays,
			const std::vector<PhotoCamera*>& vec_pPCams ,
			const cv::Mat& DepthMap ,
			const cv::Mat& DepthMask ,
			const bool BUpdateRay) ;


		static bool
			statisticReprojectiveError(
			cv::Mat& ErrorMap , 
			const cv::Mat& DepthMask ,
			const std::vector<float>& x3Points , 
			const std::vector<float>& x3Rays ,
			const std::map<int , cv::Mat>& map_uFlows ,
			const std::map<int , cv::Mat>& map_vFlows ,
			const std::vector<PhotoCamera*> &vec_pbPCams ) ;

	};

}


#endif