
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include <algorithm>
#include <iterator>
#include <iostream>
#include <utility>
#include <cmath>
#include <list>

#include <omp.h>


#include "../SceneConfiguration.h"

#include "DepthFieldOptimizer.h"
#include "DepthFieldHelper.h"

#include "DepthFieldCommon/SceneFlowOptimizer.h"




#ifdef _SEE_IMAGE_
#undef _SEE_IMAGE_
#endif
#define _SEE_IMAGE_(img_ , wname_)\
	cv::namedWindow(wname_ , 0);\
	if(img_.cols<1680 && img_.rows<1050)\
	cv::resizeWindow(wname_ , img_.cols , img_.rows);\
	else\
	cv::resizeWindow(wname_ , img_.cols/2 , img_.rows/2);\
	cv::imshow(wname_ , img_);\
	cv::waitKey(0);\
	cv::destroyWindow(wname_);


#undef _PRINT_FLOATVEC_
#define _PRINT_FLOATVEC_(_vec_data)\
	fflush(stdout);\
	std::copy(_vec_data.begin() , _vec_data.end() , (std::ostream_iterator<float>(std::cout , "\t\t"))) ;\
	std::cout<<std::endl ;\
	fflush(stdout);



#undef _DEBUG_SEE_DEPTH_BEFORE_SCENEFLOW_
#define _DEBUG_SEE_DEPTH_BEFORE_SCENEFLOW_ 0


#undef _DEBUG_SEE_DEPTH_AFTER_SCENEFLOW_ 
#define _DEBUG_SEE_DEPTH_AFTER_SCENEFLOW_ 0


#undef _DEBUG_SEE_CURRENT_DEPTH_
#define _DEBUG_SEE_CURRENT_DEPTH_ 0



#undef _CLEAN_BORDER_
#define _CLEAN_BORDER_(_img , NBorders , KVAL)\
	_img.colRange(0,NBorders).setTo(KVAL);\
	_img.colRange(_img.cols-1-NBorders , _img.cols-1).setTo(KVAL);\
	_img.rowRange(0,NBorders).setTo(KVAL);\
	_img.rowRange(_img.rows-1-NBorders , _img.rows-1).setTo(KVAL);\


#undef _DO_POISSON_
#define _DO_POISSON_ 0


namespace stereoscene{


	bool
		DepthFieldOptimizer::
		operator() (
		const std::string &_sdir_Images ,
		const int KthRefPCam ,
		const bool BMetric_L1 , 
		const bool BStatisticError , 
		const int KScene , 
		const int NIters) {


			if(KScene) {
				fflush(stdout) ;
				fprintf(stdout , "\n DepthFieldOptimizer: About to optimize the depth field of scene.%d\n" , KScene) ;
				fflush(stdout) ;
			}


			const bool BSceneFlow_L1 = BMetric_L1 ;


			const int NPad = _PER_BUNDLE_NFRAMES_/2 ;


			for(int niter=0; niter<NIters ; niter++) {

				fflush(stdout) ;
				fprintf(stdout , "\t-- Round of Iterations : ( %d ) --\n" , niter+1) ;
				fflush(stdout) ;


				//const bool BDenoise = !BSceneFlow_L1 ? true : !niter ;
				const bool BDenoise = true /*&& !BSceneFlow_L1*/ ;

				if( BDenoise ) {


					///step.1: denoise depth map to smooth the depth field and preserve sharp features
					///
					fflush(stdout) ;
					fprintf(stdout , " --> step.1 : denoising the depth field\n") ;
					fflush(stdout) ;


					denoiseDepthField(_sdir_Images , KthRefPCam , true);
				}


#if _DEBUG_SEE_DEPTH_BEFORE_SCENEFLOW_
				cv::Mat TmpDepth0 = cv::Mat::ones(_DepthMap_.size() , CV_8UC1) ;
				cv::normalize(_DepthMap_ , TmpDepth0 , 0 , 255 , cv::NORM_MINMAX , CV_32FC1 , _DepthMask_) ;
				_CLEAN_BORDER_(TmpDepth0 , 5 , 255);
				_SEE_IMAGE_(TmpDepth0 , "Depth Before Sceneflow");
#endif


				const bool BInitiating = (!niter) ;	

				if(KthRefPCam>=NPad){


					///step.2: approximate scene flow to optimize the coordinates of vertices
					///
					fflush(stdout) ;
					fprintf(stdout , " --> step.2 : optimizing the depth field by implementing scene flow\n") ;
					fflush(stdout) ;


					approximateSceneFlow(\
						BSceneFlow_L1 , BInitiating , BStatisticError , _sdir_Images , KthRefPCam ) ;

				}


#if _DEBUG_SEE_DEPTH_AFTER_SCENEFLOW_ 
				cv::Mat TmpDepth1 ;
				cv::normalize(_DepthMap_ , TmpDepth1 , 1.0/500 , 1.0 , cv::NORM_MINMAX , CV_32FC1 , _DepthMask_) ;
				cv::Mat TmpDepth1.setTo(0 , 255-_DepthMask_);
				_SEE_IMAGE_(TmpDepth1 , "Depth After Sceneflow");
#endif

			}


			//denoiseDepthField(_sdir_Images , KthRefPCam , true);


			///step.3: statistic re-projective error in sceneflow
			///
			fflush(stdout) ;
			fprintf(stdout , " --> step.3 : statistic reprojective errors of scene flow\n") ;
			fflush(stdout) ;


			if(BStatisticError) {

				SceneFlowOptimizer::statisticReprojectiveError(\
					_ErrorMap_ , _DepthMask_ , _vec_x3Points_ , _vec_x3Rays_ ,\
					_map_uFlow_ , _map_vFlow_ , _vec_pbPCams_) ;
			}


#if _DEBUG_SEE_CURRENT_DEPTH_
			///Note: before implement denosing, depth field should be normalized into [0 , 1]
			cv::Mat TmpDepthFinal ;
			cv::normalize(_DepthMap_ , TmpDepthFinal , 1.0/500 , 1.0 , cv::NORM_MINMAX , CV_32FC1 , _DepthMask_) ;
			_SEE_IMAGE_(TmpDepthFinal , "Current Depth Map");
#endif


			///clean up
			std::for_each(_vec_pbPCams_.begin() , _vec_pbPCams_.end() , 
				[](PhotoCamera* pPCam){
					pPCam = NULL ;
			});


			return true ;
	}


}



namespace stereoscene{


	bool
		DepthFieldOptimizer::
		approximateSceneFlow(
		const bool BSceneFlow_L1 ,
		const bool BInitiating , 
		const bool BStatisticError , 
		const std::string& _sdir_Image , 
		const int KthRefPCam) {


			const int NPad = _PER_BUNDLE_NFRAMES_/2;

			///Initiating including dense tracking and depth triangulating 
			///so that x3Points, x3Rays and flow maps would be updated
			///
			if(BInitiating) {
				_vec_pbPCams_.clear() ;

				for(int nc=KthRefPCam-NPad ; nc<=KthRefPCam+NPad; nc++) {
					std::map<int , PhotoCamera>::iterator it_PCam = _map_PCams_.find(nc);
					if(it_PCam==_map_PCams_.end()) {
						fflush(stdout);
						fprintf(stderr , "\nError in <DepthFieldOptimizer/calculateSceneFlow>\n");
						fflush(stdout);
						std::cout<<KthRefPCam<<"\t"<<nc<<std::endl;
						exit(0);
					}

					_vec_pbPCams_.push_back(&it_PCam->second) ;
				}


				///Before implement scene flow, depth field should be normalized into the real value range
				cv::normalize(_DepthMap_,_DepthMap_, _DepthBox_[0], _DepthBox_[1], cv::NORM_MINMAX, CV_32FC1, _DepthMask_);
				_DepthMap_.setTo(_DepthBox_[1] , 255-_DepthMask_);
			}		


			if(!BSceneFlow_L1) {

				///Implement original scene flow optimization using LS minimizer
				SceneFlowOptimizer(_vec_pbPCams_ , _vec_x3Points_  ,_vec_x3Rays_ , _map_uFlow_ , _map_vFlow_ ,  _DepthMap_ , _ErrorMap_ , _DepthMask_ , _DepthBox_) (
					SceneFlowOptimizer::SCENEFLOW_L2 , BInitiating , BStatisticError , _sdir_Image , KthRefPCam , _RefImage_) ;

			} else {

				///Implement scene flow optimization using TV-L1 integrator
				SceneFlowOptimizer(_vec_pbPCams_ , _vec_x3Points_ , _vec_x3Rays_ , _map_uFlow_ , _map_vFlow_ ,  _DepthMap_ , _ErrorMap_ ,  _DepthMask_ , _DepthBox_) (
					SceneFlowOptimizer::SCENEFLOW_L1 , BInitiating , BStatisticError ,_sdir_Image , KthRefPCam , _RefImage_) ;
			}



			return true ;
	}


}



#include <opencv2/imgproc.hpp>


#include "DepthFieldCommon/DepthDenoiser.h"
#include "DepthFieldHelper.h"


namespace stereoscene{


	bool
		DepthFieldOptimizer::
		denoiseDepthField(
		const std::string &sdir_Images ,
		const int KthRefPCam ,
		const bool BGWeighted) {



			cv::Size MapSize(_DepthMap_.cols , _DepthMap_.rows);

#if 0
			const int NIters = 30  ;
			const double DLambda = 1.0/200 , 
				DTheta = 1.0 ,
				DTau = 0.02 ;


			cv::Mat Grad ;
			if(BGWeighted) {
				if(_RefImage_.empty()) {
					DepthFieldHelper::loadSpecificImage(_RefImage_ , sdir_Images , MapSize , KthRefPCam , false);
				}

				cv::cvtColor(_RefImage_ , Grad , cv::COLOR_RGB2GRAY);

				Grad.convertTo(Grad , CV_32FC1 , 1.0/255) ;
				//_SEE_IMAGE_(Grad , "GRAD");

				cv::GaussianBlur(Grad , Grad , cv::Size(5 , 5) , 1.0 , 1.0 , cv::BORDER_REPLICATE) ;
				cv::Sobel(Grad , Grad , CV_32FC1 , 1 , 0 , 3);
				cv::Sobel(Grad , Grad , CV_32FC1 , 0 , 1 , 3);

				Grad.convertTo(Grad , CV_64FC1);
				cv::exp(-2*cv::abs(Grad) ,Grad);
			}


			_DepthMap_.convertTo(_DepthMap_ , CV_64FC1) ;

			///Before denoising, depth value should be normalized into [0,1]
			cv::normalize(_DepthMap_, _DepthMap_ , 0.0 , 1.0 , cv::NORM_MINMAX , CV_64FC1 , _DepthMask_);
			cv::GaussianBlur(_DepthMap_ , _DepthMap_ , cv::Size(5 , 5) , 1.0 , 1.0 , cv::BORDER_REFLECT);

			const cv::Mat InvDMask = 255-_DepthMask_ ;
			_DepthMap_.setTo(1.0 , InvDMask);


			DepthDenoiser::denoiseDepth(_DepthMap_ , _DepthMap_ , _DepthMask_ , NIters , DLambda , DTheta , DTau , BGWeighted , Grad) ;
#endif

			_DepthMap_.convertTo(_DepthMap_ , CV_32FC1);
			cv::Mat PatchDepth ;
			cv::Mat(_DepthMap_ , cv::Rect(_BORDER_NGAP_ , _BORDER_NGAP_ , MapSize.width-_BORDER_NGAP_*2 , MapSize.height-_BORDER_NGAP_*2)).copyTo(PatchDepth);


			///d=[7 , 9] , sigma_c = [0.3 , 0.5]
			cv::bilateralFilter(PatchDepth , 
				cv::Mat(_DepthMap_ , cv::Rect(_BORDER_NGAP_ , _BORDER_NGAP_ , MapSize.width-_BORDER_NGAP_*2 , MapSize.height-_BORDER_NGAP_*2)) , 
				9 , 2.0 , 20); //key parameters





			///normalized back
			cv::normalize(_DepthMap_ , _DepthMap_ , _DepthBox_[0] , _DepthBox_[1] , cv::NORM_MINMAX , CV_32FC1 , _DepthMask_) ;
			_DepthMap_.setTo(_DepthBox_[1] , 255-_DepthMask_);


			return true;
	}


}


//#include "DepthFieldCommon/PoissonDistanceFieldReconstructor.h"


namespace stereoscene{


	bool
		DepthFieldOptimizer::
		reconstructPoissonField(
		cv::Mat& dDepthMap , 
		const std::string& sdir_Save) {

#if 0
		PoissonDistanceFieldReconstructor()(dDepthMap , _DepthMap_ , _DepthMask_ , sdir_Save);
#endif


		return true ;
	}


}