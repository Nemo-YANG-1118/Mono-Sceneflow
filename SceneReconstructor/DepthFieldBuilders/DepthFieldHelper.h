
#ifndef DEPTH_FIELD_HELPER_H_
#define DEPTH_FIELD_HELPER_H_


#include "../SceneHelper.h"
#include "../SceneConfiguration.h"


namespace stereoscene{


	struct DepthFieldHelper : public SceneHelper{

	public:
		/// Functions for synthesize the scene
		static bool
			synthesizeScene(
			const bool BOfflineSynthesize , 
			const bool BOnlyStructure , 
			const bool BSaveScene,
			const std::string& sdir_Save ,
			const std::string& sname_Save ,
			std::vector<float>& x3P ,
			std::map<int , PhotoCamera>& map_PCam ,
			const int kthRef ,
			const int NPad ,
			const cv::Mat& DepthMap=cv::Mat() ,
			const bool BSynthTexture=false ,
			const std::vector<std::string> &vec_sTexName=std::vector<std::string>() ,
			std::vector<unsigned char>& x3T=std::vector<unsigned char>()) ;


		static bool
			transformPCamsI2J(
			std::map<int , PhotoCamera>& map_newPCams,
			const int Indi ,
			const int Indj ,
			std::map<int , PhotoCamera>& map_oldPCams ) ;


		static bool
			triangulateDepthMap(
			std::vector<float>& x3P,
			const PhotoCamera& PCams ,
			const cv::Mat& DepthMap , 
			const cv::Mat& DepthMask,
			const std::vector<float>& DepthBox,
			const bool BGetTexture=false ,
			const cv::Mat& TexImage=cv::Mat(),
			std::vector<unsigned char>& x3T=std::vector<unsigned char>() ,
			const int BorderBound=12) ;


		static bool
			extractValidLinesFromCells(
			std::vector<cind>& x3L,
			const std::vector<cind>& x3C,
			const std::vector<float>& x3P,
			const float FThreshCell=10.0f);


		static bool
			extractValidTrianglesFromCells(
			std::vector<cind>& x3Tri,
			const std::vector<cind>& x3C,
			const std::vector<float>& x3P,
			const float FThreshCell=10.0f);


		static bool
			findConcaveMaskfromPointsVector(
			cv::Mat& ConcaveMask ,
			const cv::Size& MapSize ,
			const std::vector<float>& Pvec,
			const std::vector<std::vector<float> >& vec_x3Points) ;


		static bool
			findConcaveMaskfromPointsArray(
			cv::Mat& ConcaveMask ,
			const cv::Size& MapSize ,
			const std::vector<float>& Pvec,
			const std::vector<float>& x3P) ;


		static bool
			findConcaveMaskfromMaskMap(
			cv::Mat& ConcaveMask ,
			const cv::Size& MapSize ,
			const cv::Mat& MaskMap) ;



	public:
		///Functions for visualization
		static bool
			seeTrimeshTextured(
			float *const _px3Points ,
			const int NPoints , 
			cind *const _px3TriCells , 
			const int NTriCells ,
			unsigned char *const _px3Texture , 
			const int NChannels , 
			float *const _px3Normals=NULL ,
			const bool BInverse=true , 
			const std::string& _sname_Window=std::string(),
			const double DScale=1.0,
			const bool BSaveMeshPhoto=false,	
			const std::string& _sdir=std::string() ,
			const std::string& _sname_Save=std::string());


		static bool
			seeTrimesh(			 
			float *const _x3Points , 
			const int NPoints , 
			cind *const _x3TriCells , 
			const int NTriCells , 
			float *const _px3Normals=NULL , 
			const bool BInverse=true ,
			const std::string& _sname_Window=std::string(),
			const double DScale=1.0 ,
			const bool BSaveMeshPhoto=false,
			const std::string& _sdir=std::string() ,
			const std::string& _sname_Save=std::string()) ;



	public:
		/// Functions for loading and saving
		static bool 
			loadSpecificImage(
			cv::Mat& _DstImage ,
			const std::string& sdir_Image , 
			const cv::Size& ImageSize ,
			const int Kth ,
			const bool BGray=true) ;


		static bool
			savePoints(
			const std::vector<std::vector<float>>& _vec_x3Points , 
			const std::string &_sdir ,
			const std::string &_sname) ;


	public:
		/// Functions for assistency
		static bool
			generateDepthBoundingMask(
			cv::Mat& _BoundMask , 
			const cv::Mat& _DepthMap ,
			const cv::Mat& _DepthMask , 
			const double DFarRatio=2.0 , 
			const double DNearRatio=20.0 ) ;



		static bool
			generateOrthoBoundingBoxfromCleanedSource(
			std::vector<float>& x3OrthoBox ,
			std::vector<float>& x2OrthoBox , 
			const PhotoCamera & PCam , 
			const std::vector<std::vector<float>>& vec_sx3) ;



		static bool 
			filterOrthoPointsArray(
			std::vector<float>& dx3P ,
			const std::vector<float>& sx3P ,
			const std::vector<float>& x3OrthoBox ,
			const PhotoCamera& PCam , 
			const double DZNearRatio=1.0 , 
			const double DZFarRatio=1.0) ; 


		static bool
			filterOrthoDepthMap(
			cv::Mat& nDepthMap ,
			cv::Mat& nDepthMask ,
			const PhotoCamera& PCam ,
			const cv::Mat& sDepthMap ,
			const cv::Mat& sDepthMask ,
			const std::vector<float>& x3OrthoBox ,
			const std::vector<float>& x2OrthoBox , 
			const double DZNearRatio=1.0 , 
			const double DZFarRatio=1.0) ;


	} ;

}


#endif