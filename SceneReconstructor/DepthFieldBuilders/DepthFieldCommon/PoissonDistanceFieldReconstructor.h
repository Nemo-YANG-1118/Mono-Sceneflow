
#ifndef POISSON_DISTANCE_FIELD_RECONSTRUCTOR_H_
#define POISSON_DISTANCE_FIELD_RECONSTRUCTOR_H_


#include <opencv2/core.hpp>

#include <vector>
#include <map>
#include <string>


#include "PhotoCamera.h"

#ifdef _BORDER_NGAP_ 
#undef _BORDER_NGAP_
#endif
#define _BORDER_NGAP_ 2


namespace stereoscene{


	struct PoissonDistanceFieldReconstructor{

	public:

		bool
			operator() (
			cv::Mat& dDepthMap ,
			const cv::Mat &sDepthMap , 
			const cv::Mat &sDepthMask ,
			const std::string& sname_save=std::string()) {
		

				std::vector<float> sx3Points  ;

				convertDepthMapToDistanceField(sx3Points , sDepthMap, sDepthMask);
		

				reconstructPoisson(dDepthMap , sx3Points , sDepthMask ,sname_save );



				return true ;
		}



	public:

		static bool
			preprocessTriMesh(
			std::vector<float>& dx3P,
			const bool BInverse ,
			const std::vector<float>& sx3P) ;


		static bool
			PoissonDistanceFieldReconstructor::
			orthoTriangulateDepthMap(
			std::vector<float>& dx3Points ,
			const PhotoCamera& PCam ,
			const cv::Mat& DepthMap ,
			const cv::Mat& DepthMask ) ;


		static bool
			convertDepthMapToDistanceField(
			std::vector<float>& dx3Points ,
			const cv::Mat& sDepthMap , 
			const cv::Mat& sDepthMask);


		static bool 
			reconstructPoisson(
			cv::Mat& DepthMap ,
			const std::vector<float>& sx3Points ,
			const cv::Mat& DepthMask , 
			const std::string& sname_save) ;


	};


}


#endif