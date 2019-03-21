
#ifndef SCENE_HELPER_H_
#define SCENE_HELPER_H_

#include <opencv2/core.hpp>

#include <vector>
#include <map>
#include <string>


#include "PhotoCamera.h"


namespace stereoscene{


	struct SceneHelper{

	public:
		/// Functions for finding visual hull
		///
		static bool
			generateConvexMask(
			cv::Mat &_ConvexMask ,
			const cv::Mat& _BinMask) ;


		static bool
			buildBinaryMask(
			cv::Mat& _BinMask,
			const cv::Mat& _SrcMask,
			const int NSize=3) ;


		static bool 
			generateVisualMask(
			cv::Mat& _VisHullMask , 
			const cv::Mat& _SrcImage , 
			const int KScene , 
			const bool BSave=false ,
			const std::string& _sdir=std::string() ,
			const std::string& _sname=std::string()) ;


		static bool
			blackMapBorder(
			cv::Mat& _SrcImage , 
			const int NPixels=-1) ;


	public:
		/// Functions for loading and saving
		///
		static int 
			findStructurefromMotion(
			std::vector<std::vector<float>>& vec_x3Points, 
			std::map<int , PhotoCamera>& map_PCams , 
			const int kCurPCams ,
			const int NPads ,
			const cv::Size ImageSize ,
			const std::vector<std::vector<float>>& vec_StructurePoints ,
			const std::map<int, PhotoCamera>& map_MotionPCams) ;


		static int 
			loadStructureAndMotion(
			std::vector<std::vector<float>>& vec_StructurePoints , 
			std::map<int , PhotoCamera>& map_MotionPCams ,
			const std::string& sdir) ;


		static int
			loadx3Points(
			std::vector<std::vector<float>>& _vec_x3Points ,
			const std::string& _sdir , 
			const int NThresh=0) ;


		static int
			loadPCams(
			std::map<int , PhotoCamera>& _map_PCams ,
			const std::string& _sdir , 
			const int NThresh=0) ;


		static bool
			loadImageSpecifiedSize(
			cv::Mat& _DstImage , 
			const std::string& sdir_Image , 
			const cv::Size& _ImageSize ,
			const int Kth ,
			const bool BGray=false,
			const bool BResave=false , 
			const std::string _sdir_Resave=std::string()) ;


		static bool 
			saveDepthMap(
			const int KthCurPCam ,
			const bool BTxTFormat , 
			const bool BEntireMapSaved , 
			const std::string& _sdir ,
			const std::string& _sname_DepthMap ,
			const cv::Mat& _DepthMap ,
			const cv::Mat& _DepthMask) ;



		static bool
			saveReprojectiveErrorMap(
			const int KthCurPCam ,
			const cv::Mat& _ErrorMap ,
			const std::string& _sdir , 
			const std::string& _sname) ;



		static bool 
			saveImagesAndMapsVideo(
			const cv::Size& _MapSize ,
			const std::string& _sname_Video ,
			const std::string& _sdir ,
			const std::string& _sdir_TexImages,
			const std::string& _sname_Image1 , 
			const std::string& _sname_Image2=std::string() ,
			const std::string& _sname_Image3=std::string(),
			const std::string& _sname_Image4=std::string()) ; 


		static bool
			statisticPhotoConsistency(
			const cv::Mat& DepthMap ,
			const cv::Mat& DepthMask ,
			const std::vector<float>& DepthBox ,
			const std::map<int , PhotoCamera>& map_PCams ,  
			const std::string& sdir_Save , 
			const std::string& sname ,
			const std::string& sdir_Image , 
			const int KRefPCam) ;


		static bool
			deleteSpecificFile(
			const std::string& sdir_save ,
			const std::string& sname , 
			const int KRefPCam) ;


	public:
		/// Functions for displaying
		///
		static bool
			seePoints(
			const std::vector<std::vector<float>>& _vec_x3Points , 
			const bool BInverse=true , 
			const std::string& _sname_x3Points=std::string()) ;					


		static bool
			seePoints(
			std::vector<float>& _x3Points , 
			const bool BInverse=true ,
			const std::string& _sname_x3Points=std::string()) ;


		static bool
			seeTwoPointsets(
			const std::vector<float>& _x3Points0 ,
			const std::vector<float>& _x3Points1 ,
			const bool BInverse = true,
			const std::string& _sname_x3Points=std::string()) ;


		static bool
			seeCameras( 
			const std::vector<PhotoCamera>& _vec_PCams , 
			const std::vector<std::vector<float>>& _vec_Points , 
			const double DScaledCam=3.0);


		static bool
			seeHistogram(
			const cv::Mat& _Hist , 
			const std::string& _sname_hist=std::string()) ;


		static bool
			seeStructureAndMotion(
			std::vector<std::vector<float>>& vec_x3Points,
			std::map<int , PhotoCamera>& map_PCams , 
			const bool BSaveOffline , 
			const std::string& sdir_Save=std::string() , 
			const std::string& sname_Save=std::string() ) ;

	} ;


}


#endif
