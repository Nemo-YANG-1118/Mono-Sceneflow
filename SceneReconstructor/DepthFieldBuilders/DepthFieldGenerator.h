
#ifndef DEPTH_FIELD_GENERATOR_H_
#define DEPTH_FIELD_GENERATOR_H_

#include <opencv2/core.hpp>


#include <vector>
#include <map>


#include "../SceneConfiguration.h"
#include "../PhotoCamera.h"


namespace stereoscene{


	struct DepthFieldGenerator{

	public:

		DepthFieldGenerator(
			cv::Mat& _DepthMap ,
			cv::Mat& _DepthMask ,
			std::vector<float>& _DepthBox ,
			const std::vector<std::vector<float>>& _vec_x3Points ,
			const std::map<int , PhotoCamera>& _map_PCams)
			: _DepthMap_(_DepthMap) ,
			_DepthMask_(_DepthMask) ,
			_DepthBox_(_DepthBox) ,
			_vec_x3Points_(_vec_x3Points.begin() , _vec_x3Points.end()) , 
			_map_PCams_(_map_PCams.begin() , _map_PCams.end()) {}


		bool 
			operator() (
			const bool BCleaning  ,
			const int KthRefPCam ,
			const cv::Size& MapSize ,
			const bool BGenerateFullMap ,
			const int KScene,
			const bool BSaveCleanPoints=false ,
			const std::string& _sdir_Save=std::string()) ;



	public:

		static bool
			generateMapTrimesh(
			std::vector<int>& _TriCells ,
			const cv::Size& _ImageSize ,
			const cv::Mat& _ImageMask) ;


		static bool
			generateVerticesTrimesh(
			std::vector<cind>& _x3TriCells , 
			std::vector<float>& _x3Normals , 
			const std::vector<float>& _x3Points ,
			const bool BComputeNormals=false ,
			const bool BInverse=false) ;


		static bool
			orthoBackprojection(
			const cv::Mat& _DepthMap , 
			const cv::Mat& _DepthMask ,
			const cv::Mat& _TexImage , 
			const PhotoCamera& CurPCam ,
			const cv::Size& MapSize ,
			const int KthCurPCam , 
			const bool BVisualized=false ,
			const bool BUseTexture=false ,
			const bool BSaveMeshPhoto=false,
			const std::string& _sdir=std::string() ,
			const std::string& _sname_Save=std::string()) ;



	private:

		bool
			orthoTriangulatePoints(
			const PhotoCamera& PCam);


		bool
			unorthoTriangulatePoints(
			const PhotoCamera& PCam);


		int
			cleanPointsCloud() ;


		bool
			calculateDepthBox(
			std::vector<float>& DepthBox ,
			const cv::Size& MapSize ,
			const PhotoCamera& PCam ) ;


		bool
			generateDepthField(
			const int KthRefPCam ,
			const bool BGenerateFullMap ,
			const cv::Size& MapSize) ;



	private:

		cv::Mat& _DepthMap_ , & _DepthMask_ ;

		std::vector<float>& _DepthBox_ ;



	private:

		std::vector<std::vector<float>> _vec_x3Points_ ;

		std::vector<cind> _vec_TriCells_ ;

		std::map<int , PhotoCamera> _map_PCams_ ;


	} ;


}


#endif