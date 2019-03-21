
#ifndef DRPTH_DENOISER_H_
#define DRPTH_DENOISER_H_


#include <opencv2/core.hpp>


namespace stereoscene{


	struct DepthDenoiser{

	public:

		static bool 
			denoiseDepth(
			cv::Mat& _Dst ,
			const cv::Mat& _Src , 
			const cv::Mat& _SrcMask , 
			const int NIters ,
			const double DLambda , 
			const double DTheta , 
			const double DTau , 
			const bool BGWeighted ,
			const cv::Mat _Grad=cv::Mat()) ;

	} ;


}


#endif