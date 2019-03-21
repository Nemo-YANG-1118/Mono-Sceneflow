
#ifndef OPTICAL_FLOW_MATCHER_H_
#define OPTICAL_FLOW_MATCHER_H_

#include <opencv2/core.hpp>

#include <vector>
#include <map>


namespace stereoscene{


	struct OpticalFlowMatcher{

	public:

		static bool
			denseTracking_SOR(
			std::map<int , cv::Mat>& _map_uFlow ,
			std::map<int , cv::Mat>& _map_vFlow ,
			const std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>& _map_ImagesPair) ;


		static bool
			denseTracking_BROX(
			std::map<int , cv::Mat>& _map_uFlow ,
			std::map<int , cv::Mat>& _map_vFlow ,
			const std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat>>& _map_ImagesPair) ;


		static bool
			denseTracking_BROX_NCV(
			std::map<int , cv::Mat>& _map_uFlow ,
			std::map<int , cv::Mat>& _map_vFlow ,
			const std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>& _map_ImagesPair) ;



		static bool
			denseTracking_TVL1(
			std::map<int , cv::Mat>& _map_uFlow ,
			std::map<int , cv::Mat>& _map_vFlow ,
			const std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat>>& _map_ImagesPair) ;



		static bool
			denseTracking_SIMPLE(
			std::map<int , cv::Mat>& _map_uFlow ,
			std::map<int , cv::Mat>& _map_vFlow ,
			const std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>& _map_ImagesPair) ;
			
	};


}


#endif