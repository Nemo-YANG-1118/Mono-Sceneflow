
#include <Fade2D/Fade_2D.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <utility>
#include <iterator>
#include <cmath>

#include <omp.h>


#include "../SceneConfiguration.h"
#include "DepthFieldIntegrator.h"
#include "DepthFieldHelper.h"
#include "DepthFieldCommon/DepthMapper.hpp"
#include "DepthFieldCommon/TrimeshGenerator.h"
#include "DepthFieldCommon/PrimalDualSolver.h"


#ifdef _SEE_IMAGE_
#undef _SEE_IMAGE_
#endif
#define _SEE_IMAGE_(img_ , wname_)\
	cv::namedWindow(wname_ , 0);\
	if(img_.cols<1680 && img_.rows<1050){\
	cv::resizeWindow(wname_ , img_.cols , img_.rows);}\
	else{\
	cv::resizeWindow(wname_ , img_.cols/2 , img_.rows/2);}\
	cv::imshow(wname_ , img_);\
	cv::waitKey(0);\
	cv::destroyWindow(wname_);

#undef _SEE_IMAGES_PAIR_
#define _SEE_IMAGES_PAIR_(img0_ , img1_ , wname0_ , wname1_)\
	cv::namedWindow(wname0_ , 0);\
	if(img0_.cols<1680 && img0_.rows<1050)\
	cv::resizeWindow(wname0_ , img0_.cols , img0_.rows);\
	else\
	cv::resizeWindow(wname0_ , img0_.cols/2 , img0_.rows/2);\
	cv::namedWindow(wname1_ , 0);\
	if(img1_.cols<1680 && img1_.rows<1050)\
	cv::resizeWindow(wname1_ , img1_.cols , img1_.rows);\
	else\
	cv::resizeWindow(wname1_ , img1_.cols/2 , img1_.rows/2);\
	cv::imshow(wname0_ , img0_);\
	cv::imshow(wname1_ , img1_);\
	cv::waitKey(0);\
	cv::destroyWindow(wname0_);\
	cv::destroyWindow(wname1_);


#ifdef _PRINT_FLOATVEC_
#undef _PRINT_FLOATVEC_
#endif
#define _PRINT_FLOATVEC_(_vec_data)\
	fflush(stdout);\
	std::copy(_vec_data.begin() , _vec_data.end() , (std::ostream_iterator<float>(std::cout , "\t"))) ;\
	std::cout<<std::endl ;\
	fflush(stdout);



#undef _DEBUG_SEE_FORM_DEPTH_MAP_PRE_INTEG_
#define _DEBUG_SEE_FORM_DEPTH_MAP_PRE_INTEG_ 0


#undef _DEBUG_SEE_CUR_DEPTH_MAP_PRE_INTEG_
#define _DEBUG_SEE_CUR_DEPTH_MAP_PRE_INTEG_ 0


#undef _DEBUG_SEE_CUR_DEPTH_MAP_AFT_INTEG_
#define _DEBUG_SEE_CUR_DEPTH_MAP_AFT_INTEG_ 0


#undef _DEBUG_SEE_DEPTH_TRIMESH_
#define _DEBUG_SEE_DEPTH_TRIMESH_ 0



namespace stereoscene{


	bool 
		DepthFieldIntegrator::
		operator() (
		std::map<int , std::tr1::tuple<PhotoCamera , cv::Mat , cv::Mat>>& _map_DepthFields ,
		const cv::Mat& CurImage ,
		const int KthCurPCam ,
		const int KthFormPCam ){


			if(KthCurPCam) {
				fflush(stdout) ;
				fprintf(stdout , "\n DepthFieldIntegrator: About to integrate the depth fields based on scene.%d\n" , KthCurPCam) ;
				fflush(stdout) ;
			}

			const int NBorderPad = 2 ;


			///step.1: remap former depth field onto current referenced camera to generate former depth map
			///
			fflush(stdout) ;
			fprintf(stdout , " --> step.1 : remap former depth field onto current reference camera\n") ;
			fflush(stdout) ;

			std::map<int , std::tr1::tuple<PhotoCamera , cv::Mat , cv::Mat>>::iterator it_FormDepthFields(_map_DepthFields.begin()) ;
			if(KthFormPCam<=0) {
				std::advance(it_FormDepthFields , _map_DepthFields.size()-1) ;
			} else {
				it_FormDepthFields = _map_DepthFields.find(KthFormPCam);
				fflush(stdout) ;
				fprintf(stdout , "\nError in <DepthFieldIntegrator/operator()>\n") ;
				fflush(stdout) ;
				exit(0);
			}


			remapFormerDepth(it_FormDepthFields->second , KthCurPCam , NBorderPad) ;



#if _DEBUG_SEE_CUR_DEPTH_MAP_PRE_INTEG_ 
			{
				cv::Mat TmpDepthMap0 ;
				cv::normalize(_CurDepthMap_ , TmpDepthMap0 , 0 , 1 , cv::NORM_MINMAX , CV_32FC1 , _InternFormDepthMask_) ;
				TmpDepthMap0.setTo(0 , 255-_CurDepthMask_);
				_SEE_IMAGE_(TmpDepthMap0 , "Original Current Depth Map");
			}
#endif


			///step.2: integrate current depth map with former to optimize current depth field
			///
			fflush(stdout) ;
			fprintf(stdout , " --> step.2 : integrate depth maps to globally optimize current depth field\n") ;
			fflush(stdout) ;

			const bool BDoBilateralFilter = false ;


			integrateDepthMaps(NBorderPad , CurImage , BDoBilateralFilter) ;



#if _DEBUG_SEE_CUR_DEPTH_MAP_AFT_INTEG_ 
			{
				cv::Mat TmpDepthMap1 ;
				cv::normalize(_CurDepthMap_ , TmpDepthMap1 , 0 , 1 , cv::NORM_MINMAX , CV_32FC1 , _CurDepthMask_) ;
				TmpDepthMap1.setTo(0 , 255-_CurDepthMask_);
				_SEE_IMAGE_(TmpDepthMap1 , "Integrated Current Depth Map");
			}
#endif



			return true ;
	}


}




namespace stereoscene{


	bool
		DepthFieldIntegrator::
		remapFormerDepth(
		std::tr1::tuple<PhotoCamera , cv::Mat , cv::Mat>& _FormDepthFields ,
		const int KthCurPCam , 
		const int NBorderPad ) {


			const cv::Size DMapSize(_CurDepthMap_.size());
			const int NMapPoints = DMapSize.width*DMapSize.height ,
				MapBorder = _BORDER_NGAP_+NBorderPad ;


			///step.1: calculate 2d reprojective points inside new mask
			///
			std::map<int , PhotoCamera>::const_iterator it_CurPCam = _map_PCams_.find(KthCurPCam);

			if(it_CurPCam==_map_PCams_.end()) {
				fflush(stdout) ;
				fprintf(stdout , "\nError in <DepthFieldIntegrator/remapOldDepth> :\n\t cann't find Current PCam\n") ;		
				fflush(stdout) ;
				exit(0);
			}

			const PhotoCamera &CurPCam = it_CurPCam->second;
			const std::vector<float> &CurC = CurPCam._Cvec_ , &CurP = CurPCam._P_;


			const PhotoCamera &FormPCam = std::get<0>(_FormDepthFields) ;
			const std::vector<float>& FormK = FormPCam._K_ , &FormR = FormPCam._R_ , &FormC = FormPCam._Cvec_ ;

			const cv::Mat &FormDepthMap = std::get<1>(_FormDepthFields);

			std::vector<float> x2InternFormPoints , x2InternFormDepth ;
			x2InternFormPoints.reserve(NMapPoints*3) ,
				x2InternFormDepth.reserve(NMapPoints) ;

			_x2ExternFormPoints_.reserve(NMapPoints*2) , 
				_x2ExternCurPoints_.reserve(NMapPoints*2) ,
				_x2ExternDepth_.reserve(NMapPoints);



#pragma omp parallel for schedule(dynamic, 1)
			for(int nh=MapBorder ; nh<DMapSize.height-MapBorder ; nh++) {

				const float *pFormDval = FormDepthMap.ptr<float>(nh)+MapBorder ;

				for(int nw=MapBorder ; nw<DMapSize.width-MapBorder ; nw++ ,pFormDval++) {

					const float &&ux = (nw-FormK[2])/FormK[0] , &&vx = (nh-FormK[5])/FormK[4] ;

					const float &&rayx = FormR[0]*ux+FormR[3]*vx+FormR[6] ,
						&&rayy = FormR[1]*ux+FormR[4]*vx+FormR[7] ,
						&&rayz = FormR[2]*ux+FormR[5]*vx+FormR[8] ,
						&&rayn = std::sqrt(rayx*rayx+rayy*rayy+rayz*rayz);


					const float vt3[3] = {*pFormDval*rayx/rayn+FormC[0] , *pFormDval*rayy/rayn+FormC[1] , *pFormDval*rayz/rayn+FormC[2]} ;
					const float &&x2w = CurP[8]*vt3[0]+CurP[9]*vt3[1]+CurP[10]*vt3[2]+CurP[11] ,
						&&x2u = (CurP[0]*vt3[0]+CurP[1]*vt3[1]+CurP[2]*vt3[2]+CurP[3])/x2w ,
						&&x2v = (CurP[4]*vt3[0]+CurP[5]*vt3[1]+CurP[6]*vt3[2]+CurP[7])/x2w ;

					const float &&vtdval = std::sqrt((vt3[0]-CurC[0])*(vt3[0]-CurC[0])+(vt3[1]-CurC[1])*(vt3[1]-CurC[1])+(vt3[2]-CurC[2])*(vt3[2]-CurC[2])) ;

					if( x2u>1 && x2v>1 && 
						x2u<DMapSize.width-1&& x2v<DMapSize.height-1) {
#pragma omp critical
							{
								x2InternFormPoints.push_back(x2u) ,
									x2InternFormPoints.push_back(x2v) ,
									x2InternFormPoints.push_back(1) ,

									x2InternFormDepth.push_back(vtdval) ; 

							}

					} else {
#pragma omp critical
						{

							_x2ExternFormPoints_.push_back(nw) ,
								_x2ExternFormPoints_.push_back(nh);

							_x2ExternCurPoints_.push_back(x2u) ,
								_x2ExternCurPoints_.push_back(x2v);

							_x2ExternDepth_.push_back(vtdval) ;
						}

					}

				}

			}

			std::vector<float> (x2InternFormPoints).swap(x2InternFormPoints) ,
				(x2InternFormDepth).swap(x2InternFormDepth) , 
				(_x2ExternCurPoints_).swap(_x2ExternCurPoints_);

			std::vector<int> (_x2ExternFormPoints_).swap(_x2ExternFormPoints_) ;


			///step.2: remap depth field of internal vertices on current view
			///		and remesh external vertices
			///
			DepthMapper(x2InternFormPoints , std::vector<cind>() , x2InternFormDepth)(
				_InternFormDepthMap_ , _InternFormDepthMask_ , _InternFormDepthBox_ , CurPCam , DMapSize , true) ;

			_InternFormDepthMask_ = _CurDepthMask_ ;


			///filling border with mas depth which means farest
			for(int nlev=0 ; nlev<MapBorder ; nlev++) {
				_InternFormDepthMap_.row(nlev).setTo(_InternFormDepthBox_[1]);
				_InternFormDepthMap_.row(DMapSize.height-nlev-1).setTo(_InternFormDepthBox_[1]) ;
				_InternFormDepthMap_.col(nlev).setTo(_InternFormDepthBox_[1]);
				_InternFormDepthMap_.col(DMapSize.width-nlev-1).setTo(_InternFormDepthBox_[1]) ;
			}

			cv::GaussianBlur(_InternFormDepthMap_ , _InternFormDepthMap_ , cv::Size(5 ,5) , 1.0 , 1.0 , cv::BORDER_REFLECT);


#if _DEBUG_SEE_FORM_DEPTH_MAP_PRE_INTEG_
			cv::Mat TmpDepth ;
			_InternFormDepthMap_.convertTo(TmpDepth , CV_32FC1);
			cv::normalize(TmpDepth , TmpDepth , 0 , 1 , cv::NORM_MINMAX , CV_32FC1 , _InternFormDepthMask_);
			TmpDepth.setTo(1 , 255-_InternFormDepthMask_);
			_SEE_IMAGE_(TmpDepth , "Form Depth Map Pre-Integration");
#endif


			return true ;
	}



	bool
		DepthFieldIntegrator::
		integrateDepthMaps(
		const int NBorderPad ,
		const cv::Mat& _CurImage , 
		const bool BDoBilateralFilter) {


			const bool BWeighted = true ;


			const double DLambda = 1.0/200 , ///[1.0/200 ]
				DTheta = 1.0 , 
				DTau = 0.02 ; ///[ 0.02 ] 


			const int NWidth = _CurDepthMap_.cols ,
				NHeight = _CurDepthMap_.rows ,
				NIters = 60 ,
				UInd = 1 ;


			///build source maps and mask maps
			_CurDepthMap_.convertTo(_CurDepthMap_ , CV_64FC1) ;
			for(int nlev=0; nlev<_BORDER_NGAP_+NBorderPad ; nlev++) {
				_CurDepthMap_.row(nlev).setTo(_CurDepthBox_[1]);
				_CurDepthMap_.row(NHeight-nlev-1).setTo(_CurDepthBox_[1]) ;
				_CurDepthMap_.col(nlev).setTo(_CurDepthBox_[5]);
				_CurDepthMap_.col(NWidth-nlev-1).setTo(_CurDepthBox_[1]) ;
			}
			cv::normalize(_CurDepthMap_ , _CurDepthMap_ , 0.0 , 1.0 , cv::NORM_MINMAX);


			_InternFormDepthMap_.convertTo(_InternFormDepthMap_ , CV_64FC1) ;
			cv::normalize(_InternFormDepthMap_ , _InternFormDepthMap_ , 0.0 , 1.0 , cv::NORM_MINMAX );
			//_InternFormDepthMap_.setTo(1.0 , 255-_InternFormDepthMask_);

			std::vector<const cv::Mat*> vec_SrcMaps , vec_VMasks ;
			vec_SrcMaps.push_back(&_InternFormDepthMap_) , vec_SrcMaps.push_back(&_CurDepthMap_) ;
			vec_VMasks.push_back(&_InternFormDepthMask_) , vec_VMasks.push_back(&_CurDepthMask_) ;



			///generate weighted map
			cv::Mat WMap0 , WMap1 ;
			if(BWeighted){

				if(_CurImage.channels()>1){
					cv::cvtColor(_CurImage , WMap0 , cv::COLOR_BGR2GRAY);
					WMap0.convertTo(WMap0 , CV_32FC1 , 1.0/255);
				} 
				else {
					_CurImage.convertTo(WMap0 , 1.0/255);
				}

				//cv::medianBlur(WMap0 , WMap1 , 3);
				//cv::GaussianBlur(WMap1 , WMap1 , cv::Size(5 , 5) , 1.0 , 1.0 , cv::BORDER_REFLECT);
				//cv::normalize(WMap0 , WMap0 , 0 , 1 , cv::NORM_MINMAX);

				if(1) {
					///d=[5,7,9] , sigma_c=[5.0 , 6.0]
					cv::bilateralFilter(WMap0 , WMap1 , 5 , 3.0 , 20); //[5 , 6.0]
					cv::Sobel(WMap1 , WMap1 , CV_32FC1 ,1 , 0 , 3);
					cv::Sobel(WMap1 , WMap1 , CV_32FC1 ,0 , 1 , 3);
					WMap1.convertTo(WMap1 , CV_64FC1);

					///alpha set to -10 , suggested by Newcombe
					cv::exp(-10.0*cv::abs(WMap1) , WMap1);
				}
				else {
					cv::bilateralFilter(WMap0 , WMap1 , 9 , 3.0 , 20);
					cv::Scharr(WMap1 , WMap1 , CV_32FC1 , 1 , 0);
					cv::Scharr(WMap1 , WMap1 , CV_32FC1 , 0 , 1) ;
					WMap1.convertTo(WMap1 , CV_64FC1);

					cv::exp(-5.0*cv::abs(WMap1) , WMap1);
				}

			}



			///implement TV-L1 depth integration
			PrimalDualSolver(_CurDepthMap_ , vec_SrcMaps , vec_VMasks , _CurDepthMask_  , 1) (
				NIters , DLambda , DTheta , DTau , BWeighted , WMap1);



			///smoothness : d=[5, 7 , 9] , sigma_c = [0.3 , 0.5]
			///shapeness : d=[5 , 7 ,9] , sigma_c = [5.0  , 6.0 ]
			///
			cv::Mat TmpDepthMap ;
			_CurDepthMap_.convertTo(TmpDepthMap  ,CV_32FC1);
			if(NWidth>1280 || NHeight>960){
				cv::bilateralFilter(TmpDepthMap  , _CurDepthMap_ , 9 , 5.0 , 50);
			}
			else {
				cv::bilateralFilter(TmpDepthMap , _CurDepthMap_ , 5 , 1.0 , 50);//key parameters[5 , 0.5]
			}


			cv::normalize(_CurDepthMap_, _CurDepthMap_ , _CurDepthBox_[0] , _CurDepthBox_[1] ,cv::NORM_MINMAX , CV_32FC1 , _CurDepthMask_);
			_CurDepthMap_.setTo(_CurDepthBox_[1] , 255-_CurDepthMask_);


			///clean up
			std::for_each(vec_SrcMaps.begin() , vec_SrcMaps.end() , 
				[](const cv::Mat* _pmat){
					_pmat = NULL;
			});

			std::for_each(vec_VMasks.begin() , vec_VMasks.end() , 
				[](const cv::Mat* _pmat){
					_pmat = NULL;
			});



			return true ;
	}

}