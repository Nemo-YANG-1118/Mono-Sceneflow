
#include <stlplus3/filesystemSimplified/file_system.hpp>

#include <opencv2/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/cudalegacy.hpp>

#include <iostream>
#include <algorithm>
#include <iterator>
#include <utility>
#include <numeric>

#include <omp.h>


#include "SceneFlowOptimizer.h"
#include "OpticalFlowMatcher.h"
#include "../DepthFieldHelper.h"



#undef _CHECK_FOLDER_
#define _CHECK_FOLDER_(sdir_)\
	if(!stlplus::folder_exists(sdir_) || stlplus::folder_empty(sdir_)){\
	fflush(stderr);\
	fprintf(stderr , "\nError in <SceneFlowOptimizer> : ( %s )\ doesn't existed\n" , sdir_.c_str()) ;\
	fflush(stderr); }\


#undef _SEE_IMAGE_
#define _SEE_IMAGE_(img_ , wname_)\
	cv::namedWindow(wname_ , 0);\
	if(img_.cols<1680 && img_.rows<1050)\
	cv::resizeWindow(wname_ , img_.cols , img_.rows);\
	else\
	cv::resizeWindow(wname_ , img_.cols/2 , img_.rows/2);\
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


#undef _PRINT_VEC_
#define _PRINT_VEC_(_biter , _eiter , _typename)\
	fflush(stdout);\
	std::copy(_biter , _eiter , (std::ostream_iterator<_typename>(std::cout , "\t\t"))) ;\
	std::cout<<std::endl ;\
	fflush(stdout);



#undef _OPTFLOW_USE_CPU_SOR_
#define _OPTFLOW_USE_CPU_SOR_ 8


#undef _OPTFLOW_USE_CUDA_BROX_NCV_
#define _OPTFLOW_USE_CUDA_BROX_NCV_ 9


#undef _OPTFLOW_USE_CUDA_BROX_
#define _OPTFLOW_USE_CUDA_BROX_ 10


#undef _OPTFLOW_USE_CUDA_TVL1_
#define _OPTFLOW_USE_CUDA_TVL1_ 11


#undef _OPTFLOW_USE_CPU_SIMPLE_
#define _OPTFLOW_USE_CPU_SIMPLE_ 12


#undef _DEBUG_TRIANGULATED_POINTS_
#define _DEBUG_TRIANGULATED_POINTS_ 0


#undef _DEBUG_SEE_OPTICAL_FLOW_
#define _DEBUG_SEE_OPTICAL_FLOW_ 0


#undef _DEBUG_SEE_MULTIPLE_DMAPS_ 
#define _DEBUG_SEE_MULTIPLE_DMAPS_ 0 



#undef _USE_CUDA_VER_
#define _USE_CUDA_VER_ 0


namespace stereoscene{


	bool
		SceneFlowOptimizer::
		operator()(
		const int OPTIMIZER_TYPE , 
		const bool BIniating , 
		const bool BStatisticError , 
		const std::string& _sdir_Image , 
		const int KthRefPCam ,
		cv::Mat& _RefImage , 
		const bool BBoundOptFlow , 
		const double _DTopBoundRatio , 
		const double _DBottomBoundRatio ) {


			const int KMethod_OpticalFlow = _OPTFLOW_USE_CUDA_BROX_; 


			const int NCams = _vec_pPCams_.size() ,
				NPad = NCams/2 ;

			const cv::Size MapSize(_DepthMap_.cols , _DepthMap_.rows) ;


			if(BIniating) {

				_map_uFlow_.clear() , _map_vFlow_.clear() ;


				///step.1: calculate 2d-motion field by dense tracking
				///
				std::map<int , cv::Mat> map_sImages ;
				_CHECK_FOLDER_(_sdir_Image);


				std::vector<std::string> vec_fileNames = stlplus::folder_files(_sdir_Image);
				const int NFiles = vec_fileNames.size() ;

				for(int nf=0 ; nf<NFiles && nf<=KthRefPCam+NPad ; ) {
					std::string sext_ = stlplus::extension_part(vec_fileNames[nf]);
					if( sext_=="bmp"||sext_=="BMP" ||
						sext_=="jpg"||sext_=="JPG" ||
						sext_=="png"||sext_=="PNG" ) {

							if(nf>=KthRefPCam-NPad) {

								cv::Mat img_ ;
								if(nf!=KthRefPCam) {
									img_ = cv::imread(_sdir_Image+"/"+vec_fileNames[nf] , cv::IMREAD_GRAYSCALE) ;
									if(img_.cols!=MapSize.width || img_.rows!=MapSize.height) {
										cv::resize(img_ , img_ , MapSize) ;
									}

								} else {
									img_ = cv::imread(_sdir_Image+"/"+vec_fileNames[nf]) ;
									if(img_.cols!=MapSize.width || img_.rows!=MapSize.height) {
										cv::resize(img_ , img_ , MapSize) ;
									}
									if(_RefImage.empty()) {
										img_.copyTo(_RefImage);
									}

									cv::cvtColor(img_, img_ , cv::COLOR_BGR2GRAY) ;
								}

				
								img_.convertTo(img_ , CV_32FC1 , 1.0/255) ;


								cv::Mat ftImage;

								if( KMethod_OpticalFlow!=_OPTFLOW_USE_CUDA_BROX_ &&\
									KMethod_OpticalFlow!=_OPTFLOW_USE_CUDA_BROX_NCV_) {

									//cv::normalize(img_, img_, 0, 1.0, cv::NORM_MINMAX);
									img_.copyTo(ftImage);
									cv::bilateralFilter(ftImage , img_ , 5.0 , 1.0 , 10); 
								}

								//cv::medianBlur(img_ , img_ , 3);
								//cv::GaussianBlur(img_ , img_ , cv::Size(3 , 3) , 1.0 , 1.0 , cv::BORDER_REPLICATE) ;


								map_sImages.insert(std::make_pair(nf-(KthRefPCam-NPad) , img_)) ;

								//_SEE_IMAGE_(img_ , "source");
							}

							nf++ ;
					}
				}



				calculateDenseDisparity(map_sImages , KMethod_OpticalFlow) ;


				if(BBoundOptFlow) {
					boundDenseDisparity(_DTopBoundRatio , _DBottomBoundRatio);
				}
				
#if 0
				const int NFlows = _map_uFlow_.size() ;
				for(int nf=0 ; nf<NFlows ; nf++) {

					cv::Mat tflow;


					std::map<int , cv::Mat>::iterator it_uflow=_map_uFlow_.begin() ;
					std::advance(it_uflow , nf);

					///smoothness: d = 3 , sigma_c = [0.1 0.5]
					cv::bilateralFilter(it_uflow->second , tflow , 3 , 0.5 , 10);
					tflow.copyTo(it_uflow->second);


					std::map<int , cv::Mat>::iterator it_vflow=_map_vFlow_.begin() ;
					std::advance(it_vflow , nf);
					
					cv::bilateralFilter(it_vflow->second , tflow , 3 , 0.5 , 10) ;
					tflow.copyTo(it_vflow->second);
 				}
#endif

			}



			///step.2: triangulate depth map to get vertices and rays
			///		
			const bool BUpdateRay = true ;
			triangulateDepthMap(\
				_vec_x3Points_ , _vec_x3Rays_ , _vec_pPCams_ , _DepthMap_ , _DepthMask_ , BUpdateRay) ;


#if _DEBUG_TRIANGULATED_POINTS_
			DepthFieldHelper::seePoints(_vec_x3Points_ ,false) ;
#endif


			///step.3: approximate motion field by estimating the scalar of the ray
			///
			switch (OPTIMIZER_TYPE){

			case SCENEFLOW_L2 :
				{
#if _USE_CUDA_VER_
					approximateMotionField_CUDA() ;
#else
					approximateMotionField_L2() ;
#endif


#if _DEBUG_TRIANGULATED_POINTS_
					DepthFieldHelper::seePoints(_vec_x3Points_ ,false) ;
#endif
				}
				break ;



			case SCENEFLOW_L1:
				{
					cv::Mat GrayRefImage ;
					cv::cvtColor(_RefImage , GrayRefImage , cv::COLOR_BGR2GRAY) ;

					GrayRefImage.convertTo(GrayRefImage , 1.0/255);
					//medianBlur(GrayRefImage , GrayRefImage , 3);


					approximateMotionField_L1(GrayRefImage) ;


#if _DEBUG_TRIANGULATED_POINTS_
					DepthFieldHelper::seePoints(_vec_x3Points_ ,false) ;
#endif
				}
				break ;



			default:
				{
					fflush(stdout) ;
					fprintf(stdout , "\tSceneFlowOptimizer: Wrong type of optimizer\n");
					fflush(stdout) ;

					return false ;
				}
			}


			if(false && BStatisticError) {
				///static the reprojective error
				statisticReprojectiveError(\
					_ErrorMap_ , _DepthMask_ , _vec_x3Points_ , _vec_x3Rays_ ,\
					_map_uFlow_ , _map_vFlow_ , _vec_pPCams_) ;
			}



			return true ;
	}

}




namespace stereoscene{


	bool
		SceneFlowOptimizer::
		boundDenseDisparity(
		const double DTopRatio ,
		const double DBottomRatio) {

			const int NPoints = _vec_x3Points_.size()/3 ,
				NMaps = _map_uFlow_.size() ;


			const cv::Size MapSize = _DepthMap_.size() ;


			for(int nmap=0 ; nmap<NMaps ; nmap++) {

				std::map<int , cv::Mat>::iterator it_uflow = _map_uFlow_.begin() ,
					it_vflow = _map_vFlow_.begin() ;
				std::advance(it_uflow , nmap) ; 
				std::advance(it_vflow , nmap) ;


				cv::Mat& uflow = it_uflow->second , &vflow = it_vflow->second ;

				double maxu(0), minu(0) , maxv(0) , minv(0);
				cv::minMaxIdx(uflow , &minu , &maxu , NULL , NULL , _DepthMask_);
				cv::minMaxIdx(vflow , &minv , &maxv , NULL , NULL , _DepthMask_);

				cv::Mat mean_uflow , mean_vflow , stdv_uflow , stdv_vflow ;
				cv::meanStdDev(uflow , mean_uflow , stdv_uflow , _DepthMask_) ;
				cv::meanStdDev(vflow , mean_vflow , stdv_vflow , _DepthMask_) ;

				const double DTop_u =  std::min(mean_uflow.at<double>(0)+DTopRatio*stdv_uflow.at<double>(0) , maxu) ,
					DBottom_u = std::max(mean_uflow.at<double>(0) - DBottomRatio*stdv_uflow.at<double>(0) , minu),
					DTop_v = std::min(mean_vflow.at<double>(0) + DTopRatio*stdv_vflow.at<double>(0) , maxv),
					DBottom_v = std::max(mean_vflow.at<double>(0) - DBottomRatio*stdv_vflow.at<double>(0), minv) ;

				//std::cout<<DTop_u<<"\t"<<DBottom_u<<"\t"<<DTop_v<<"\t"<<DBottom_v<<"\n" ;

#pragma omp parallel for schedule(dynamic, 1)
				for(int nh=0 ; nh<MapSize.height ; nh++) {
					const unsigned char *pumask = _DepthMask_.ptr<unsigned char>(nh);

					float *puflow = uflow.ptr<float>(nh) , 
						*pvflow = vflow.ptr<float>(nh) ;

					for(int nw=0 ; nw<MapSize.width; nw++, pumask++ , puflow++ , pvflow++) {
						if(*pumask) {
							puflow[0] = std::min(std::max( puflow[0]- (*mean_uflow.ptr<double>(0)) , DBottom_u) , DTop_u) ;
							pvflow[0] = std::min(std::max( pvflow[0]- (*mean_vflow.ptr<double>(0)) , DBottom_v) , DTop_v) ;
						} else {
							puflow[0] = (*mean_uflow.ptr<double>(0)) , pvflow[0] = (*mean_vflow.ptr<double>(0)) ;
						}
					}
				}

			}


			return true ;
	}


	bool
		SceneFlowOptimizer::
		calculateDenseDisparity(
		std::map<int , cv::Mat>& _map_sImages , 
		const int KUseMethod) {


			///source images will be stored as  cv::cuda::GpuMat
			const int NImages = _map_sImages.size() ,
				KthRefPCam = NImages/2 ;

			std::map<int , cv::Mat>::const_iterator it_Image0 = _map_sImages.find(KthRefPCam);
			//std::cout<<_map_sImages.size()<<std::endl ;
			if(it_Image0==_map_sImages.end()) {
				fflush(stderr);
				fprintf(stderr , "\nError in <SceneFlowOptimizer/calculateDenseDisparity>\n") ;
				fflush(stderr);
				exit(0);
			}

			switch(KUseMethod) {

			case (_OPTFLOW_USE_CUDA_BROX_) :
				{

					std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat> > map_d_Images01 ;

					for(int nimg=0 ; nimg<NImages ; nimg++) {
						if(nimg!=KthRefPCam){
							std::map<int , cv::Mat>::iterator it_Image1 = _map_sImages.begin() ;
							std::advance(it_Image1 , nimg) ;

							cv::cuda::GpuMat d_Image0, d_Image1 ;
							d_Image0.upload(it_Image0->second , cv::cuda::Stream());
							d_Image1.upload(it_Image1->second , cv::cuda::Stream());

							map_d_Images01.insert(std::make_pair(nimg,  std::make_pair(d_Image0 , d_Image1))) ;
						}
					}


					///driving TV-L1 optical flow by using OpenCV
					OpticalFlowMatcher::denseTracking_BROX(_map_uFlow_ , _map_vFlow_ , map_d_Images01) ;

				} 
				break;


			case (_OPTFLOW_USE_CUDA_TVL1_) :
				{

				std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat> > map_d_Images01 ;

				for(int nimg=0 ; nimg<NImages ; nimg++) {
					if(nimg!=KthRefPCam){
						std::map<int , cv::Mat>::iterator it_Image1 = _map_sImages.begin() ;
						std::advance(it_Image1 , nimg) ;

						cv::cuda::GpuMat d_Image0,d_Image1 ;
						d_Image0.upload(it_Image0->second,cv::cuda::Stream());
						d_Image1.upload(it_Image1->second,cv::cuda::Stream());

						map_d_Images01.insert(std::make_pair(nimg,  std::make_pair(d_Image0 , d_Image1))) ;
					}
				}


				///driving TV-L1 optical flow by using OpenCV
				OpticalFlowMatcher::denseTracking_TVL1(_map_uFlow_ , _map_vFlow_ , map_d_Images01) ;

			} 
			break;
				

			case(_OPTFLOW_USE_CUDA_BROX_NCV_):
				{

					std::map<int , cv::Mat>::const_iterator it_Image0 = _map_sImages.find(KthRefPCam);

					std::map<int , std::pair<const cv::Mat* , const cv::Mat*> > map_pImagesPair ;

					for(int nimg=0 ; nimg<NImages ; nimg++) {
						if(nimg!=KthRefPCam){
							std::map<int , cv::Mat>::iterator it_Image1 = _map_sImages.begin() ;
							std::advance(it_Image1 , nimg) ;

							map_pImagesPair.insert(std::make_pair(nimg,  
								std::make_pair(&it_Image0->second , &it_Image1->second))) ;
						}
					}


					///driving Horn-Schunck optical flow by using NVDIA SDK
					OpticalFlowMatcher::denseTracking_BROX_NCV(_map_uFlow_ , _map_vFlow_ , map_pImagesPair) ;

				}
				break;


			case(_OPTFLOW_USE_CPU_SOR_):
				{

					std::map<int , cv::Mat>::const_iterator it_Image0 = _map_sImages.find(KthRefPCam);

					std::map<int , std::pair<const cv::Mat* , const cv::Mat*> > map_pImagesPair ;

					for(int nimg=0 ; nimg<NImages ; nimg++) {
						if(nimg!=KthRefPCam){
							std::map<int , cv::Mat>::iterator it_Image1 = _map_sImages.begin() ;
							std::advance(it_Image1 , nimg) ;


							map_pImagesPair.insert(std::make_pair(nimg,  
								std::make_pair(&it_Image0->second , &it_Image1->second))) ;
						}
					} 

					///driving Conjuncture optical flow by using Liuche's implementation
					OpticalFlowMatcher::denseTracking_SOR(_map_uFlow_ , _map_vFlow_ , map_pImagesPair) ;

				}
				break;


			case (_OPTFLOW_USE_CPU_SIMPLE_):
				{
					std::map<int , cv::Mat>::const_iterator it_Image0 = _map_sImages.find(KthRefPCam);

					std::map<int , std::pair<const cv::Mat* , const cv::Mat*> > map_pImagesPair ;

					for(int nimg=0 ; nimg<NImages ; nimg++) {
						if(nimg!=KthRefPCam){
							std::map<int , cv::Mat>::iterator it_Image1 = _map_sImages.begin() ;
							std::advance(it_Image1 , nimg) ;


							map_pImagesPair.insert(std::make_pair(nimg,  
								std::make_pair(&it_Image0->second , &it_Image1->second))) ;
						}
					} 

					///driving Conjuncture optical flow by using Liuche's implementation
					OpticalFlowMatcher::denseTracking_SIMPLE(_map_uFlow_ , _map_vFlow_ , map_pImagesPair) ;

				}
				break;


			default:
				{
					fflush(stdout);
					fprintf(stdout , "\nError : wrong case index for selecting dense method\n");
					fflush(stdout);
					exit(0);
				}

			}



#if _DEBUG_SEE_OPTICAL_FLOW_
			//fprintf(stdout , "flow data type = %d\n" , _map_uFlow_.begin()->second.type());

			for(int nimg=0 ; nimg<_map_uFlow_.size() ; nimg++) {
				std::map<int , cv::Mat>::const_iterator it_uFlow = _map_uFlow_.begin() ,
					it_vFlow = _map_vFlow_.begin() ;
				std::advance(it_uFlow , nimg) ;
				std::advance(it_vFlow , nimg) ;

				cv::Mat uFlow , vFlow ;
				cv::normalize(it_uFlow->second , uFlow , 0.0 , 1.0 , cv::NORM_MINMAX);
				cv::normalize(it_vFlow->second , vFlow , 0.0 , 1.0 , cv::NORM_MINMAX) ;

				_SEE_IMAGES_PAIR_(uFlow , vFlow , "Du" , "Dv");
			}
#endif

			return true ;
	}



	bool
		SceneFlowOptimizer::
		triangulateDepthMap(
		std::vector<float>& x3Points ,
		std::vector<float>& x3Rays,
		const std::vector<PhotoCamera*>& vec_pPCams ,
		const cv::Mat& DepthMap ,
		const cv::Mat& DepthMask ,
		const bool BUpdateRay) {

			const cv::Size MapSize(DepthMap.size());

			const int NPoints = MapSize.height*MapSize.width ,
				NCams = vec_pPCams.size() ,
				KthRefPCam = NCams/2 ;


			const PhotoCamera* const pRefPcam = vec_pPCams[KthRefPCam];
			const std::vector<float> &Kvec_ = pRefPcam->_K_ ,
				&Rvec_ = pRefPcam->_R_ ,
				&Cvec_ = pRefPcam->_Cvec_ ;

			std::vector<float> Rtvec_(9 ,0);
			PhotoCamera::transposeMatrix(Rtvec_ , Rvec_ , 3 , 3);


			///points should be anti-project with the new reference-coordinates system
			x3Points.resize(NPoints*3 , 0) ;

			if(BUpdateRay){
				x3Rays.resize(NPoints*3, 0);
			}


#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints ; np++) {

				const unsigned char *const p_dmask = DepthMask.ptr<unsigned char>(0)+np ;

				if(*p_dmask) {

					const float x2u= ((np%MapSize.width)-Kvec_[2])/Kvec_[0] , 
						x2v = ((np/MapSize.width) - Kvec_[5])/Kvec_[4] ;

					float &&rayx = Rtvec_[0]*x2u + Rtvec_[1]*x2v + Rtvec_[2] ,
						&&rayy = Rtvec_[3]*x2u + Rtvec_[4]*x2v + Rtvec_[5] ,
						&&rayz = Rtvec_[6]*x2u + Rtvec_[7]*x2v + Rtvec_[8] ;

					const float rayN2 = std::sqrt(rayx*rayx + rayy*rayy + rayz*rayz);

					rayx /= rayN2 , rayy /= rayN2 , rayz /= rayN2 ;

					if(BUpdateRay) {
						x3Rays[np*3]=rayx , x3Rays[np*3+1] = rayy , x3Rays[np*3+2] = rayz ;
					}

					/// x' = d*Ray' + Cvec'
					const float *const p_depth = DepthMap.ptr<float>(0)+np ;

					x3Points[np*3] = *p_depth*rayx + Cvec_[0] ,
						x3Points[np*3+1] = *p_depth*rayy + Cvec_[1] ,
						x3Points[np*3+2] = *p_depth*rayz + Cvec_[2] ;

				}
			}


			return true ;
	}


}


namespace stereoscene{


	bool
		SceneFlowOptimizer::
		statisticReprojectiveError(
		cv::Mat& ErrorMap , 
		const cv::Mat& DepthMask ,
		const std::vector<float>& x3Points , 
		const std::vector<float>& x3Rays ,
		const std::map<int , cv::Mat>& map_uFlows ,
		const std::map<int , cv::Mat>& map_vFlows ,
		const std::vector<PhotoCamera*> &vec_pbPCams ) {

			const cv::Size MapSize = DepthMask.size() ;
			const int NPoints = MapSize.width*MapSize.height ;

			if(ErrorMap.empty()) {
				ErrorMap.create(MapSize , CV_32FC1) ;
			}
			ErrorMap.setTo(0);


			const int NCams = vec_pbPCams.size() , 
				NPad = NCams/2 ,
				KthRefPCam = NCams/2 ; 


#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints; np++){

				const unsigned char *const p_dMask = DepthMask.ptr<unsigned char>(0)+np ;
				float *const pError = ErrorMap.ptr<float>(0)+np ;

				if(*p_dMask) {

					const float *const px3 = &x3Points[np*3] , 
						*const pray = &x3Rays[np*3];

					std::vector<float> vec_uvDir(2*NPad ,0) ;

					std::map<int , cv::Mat>::const_iterator it_uFlow = map_uFlows.begin() , 
						it_vFlow = map_vFlows.begin() ;


					int ncCnt(0);

					for(int nc=0; nc<NCams ; nc++) {
						if(nc!=KthRefPCam){

							const std::vector<float> &P_ = vec_pbPCams[nc]->_P_ ;

							const float x2hx = P_[0]*px3[0]+P_[1]*px3[1]+P_[2]*px3[2]+P_[3] ,
								x2hy = P_[4]*px3[0]+P_[5]*px3[1]+P_[6]*px3[2]+P_[7] ,
								x2hz = P_[8]*px3[0]+P_[9]*px3[1]+P_[10]*px3[2]+P_[11] ,
								x2hz2 = x2hz*x2hz ;

							const float J_[6] = { (P_[0]*x2hz-x2hx*P_[8])/x2hz2 , (P_[1]*x2hz-x2hx*P_[9])/x2hz2 , (P_[2]*x2hz-x2hx*P_[10])/x2hz2  ,
								(P_[4]*x2hz-x2hx*P_[8])/x2hz2 , (P_[5]*x2hz-x2hy*P_[9])/x2hz2 , (P_[6]*x2hz-x2hy*P_[10])/x2hz2 } ;


							const float uvMot[2] = { J_[0]*pray[0]+J_[1]*pray[1]+J_[2]*pray[2] ,
								J_[3]*pray[0]+J_[4]*pray[1]+J_[5]*pray[2] } ;

							const float *puflow = it_uFlow->second.ptr<float>(0)+np , 
								*pvflow = it_vFlow->second.ptr<float>(0)+np ;

							pError[0] += std::sqrt((uvMot[0]-puflow[0])*(uvMot[0]-pvflow[0])+(uvMot[1]-pvflow[0])*(uvMot[1]-pvflow[0]));


							++it_uFlow , ++it_vFlow ;
						}
					}

					pError[0]/=(NCams-1) ;

					pError[0] = pError[0]>1e-6 ? pError[0] : 0;

					//fprintf(stdout , "%.3f\t" , pError[0]);
				}
			}


			return true;
	}


	bool 
		SceneFlowOptimizer::
		approximateMotionField_L2() {


			const int NPoints = _vec_x3Points_.size()/3 ,
				NCams = _vec_pPCams_.size() , 
				NPad = NCams/2 ,
				KthRefPCam = NCams/2 ; 

			const cv::Size PixSize(1 , 1);

			std::vector<float>& Cvec_ = _vec_pPCams_[KthRefPCam]->_Cvec_ ;

#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints; np++){

				const unsigned char *const p_dMask = _DepthMask_.ptr<unsigned char>(0)+np ;

				if(*p_dMask) {

					float *const px3 = &_vec_x3Points_[np*3] , 
						*const pray = &_vec_x3Rays_[np*3];

					std::vector<float> vec_uvDir(2*NPad ,0) ;

					std::map<int , cv::Mat>::const_iterator it_uFlow = _map_uFlow_.begin() , 
						it_vFlow = _map_vFlow_.begin() ;

					float fuFlow(0) , fvFlow(0) , sx3Mot(0) , sx3MotN2(0) ;
					cv::Mat uFlow(1,1,CV_32FC1 , &fuFlow) , vFlow(1,1,CV_32FC1 , &fvFlow) ;

					for(int nc=0 ; nc<NCams ; nc++) {
						if(nc!=KthRefPCam){

							const std::vector<float> &P_ = _vec_pPCams_[nc]->_P_ ;

							const float &&x2hx = P_[0]*px3[0]+P_[1]*px3[1]+P_[2]*px3[2]+P_[3] ,
								&&x2hy = P_[4]*px3[0]+P_[5]*px3[1]+P_[6]*px3[2]+P_[7] ,
								&&x2hz = P_[8]*px3[0]+P_[9]*px3[1]+P_[10]*px3[2]+P_[11] ,
								&&x2hz2 = x2hz*x2hz ;

							cv::Point2f x2uv(x2hx/x2hz , x2hy/x2hz) ;


							///step.1: calculate Jacobian matrix by reproject vertices on image plane viewed by compared camera
							///
							const float Jacob[6] = {(P_[0]*x2hz-P_[8]*x2hx)/x2hz2 , 
								(P_[1]*x2hz-P_[9]*x2hx)/x2hz2 , 
								(P_[2]*x2hz-P_[10]*x2hx)/x2hz2 ,
								(P_[4]*x2hz-P_[8]*x2hy)/x2hz2 , 
								(P_[5]*x2hz-P_[9]*x2hy)/x2hz2 , 
								(P_[6]*x2hz-P_[10]*x2hy)/x2hz2 } ;


							///step.2: calculate scalar of directional vector by minimizing ||s*uvDir - uvFlow||
							///		which can be solved by using normal equation suggested by Newcombe(CVPR'2010)
							///
							const float &&x2uDir = Jacob[0]*pray[0]+Jacob[1]*pray[1]+Jacob[2]*pray[2] , 
								&&x2vDir = Jacob[3]*pray[0]+Jacob[4]*pray[1]+Jacob[5]*pray[2] ;
#if 0 
							///linear interpolation
							cv::getRectSubPix(it_uFlow->second , PixSize , x2uv , uFlow) ;
							cv::getRectSubPix(it_vFlow->second , PixSize , x2uv , vFlow) ;
#else
							const int &&nw = np%_DepthMap_.cols , &&nh = np/_DepthMap_.cols ;
							fuFlow = it_uFlow->second.ptr<float>(0)[np] , fvFlow = it_vFlow->second.ptr<float>(0)[np] ;
#endif


							///s = uvDirT*(uvDir*uvDirT)^-1 * uvFlow
							//sx3Mot += (x2uDir*fuFlow + x2vDir*fvFlow) ;
							sx3Mot += (x2uDir*(it_uFlow->second.ptr<float>(0)[np]) + x2vDir*(it_vFlow->second.ptr<float>(0)[np]));
							sx3MotN2 += (x2uDir*x2uDir + x2vDir*x2vDir) ;


							++it_uFlow , ++it_vFlow ;
						}
					}

					sx3Mot /= sx3MotN2 ;


					///step.3: update vertices and depth map
					///
					px3[0] += sx3Mot*pray[0] , px3[1] += sx3Mot*pray[1] , px3[2] += sx3Mot*pray[2] ;

					float *const p_Depth = _DepthMap_.ptr<float>(0) + np ;
					p_Depth[0] = std::sqrt( (px3[0]-Cvec_[0])*(px3[0]-Cvec_[0]) + 
						(px3[1]-Cvec_[1])*(px3[1]-Cvec_[1]) +  
						(px3[2]-Cvec_[2])*(px3[2]-Cvec_[2])) ;
				}
			}

			cv::Mat tDmap;
			cv::bilateralFilter(_DepthMap_ , tDmap , 5 , 1.0 , 10);
			tDmap.copyTo(_DepthMap_);


			return true ;
	}



	static bool 
		solvePrimalDualL1(
		cv::Mat& _DepthMap , 
		const std::vector<cv::Mat>& _vec_Depth ,
		const cv::Mat& _DepthMask , 
		const int UInd ,
		const cv::Mat& _Grad , 
		const bool BGWeighted ,
		const int NIters ,
		const double Lambda ,
		const double Theta ,
		const double Tau) {


			const int NCols = _DepthMask.cols , 
				NRows = _DepthMask.rows  ,
				NImages = _vec_Depth.size() , 
				NPad = 1 ;


			const double Sigma = 0.1/Tau ;


			cv::Mat U_ , P_ = cv::Mat::zeros(NRows, NCols, CV_64FC2) ;
			const cv::Mat &UMask = _DepthMask ;
			const std::vector<cv::Mat>& vec_Src = _vec_Depth ;
			vec_Src[UInd].copyTo(U_) ;


			std::vector<cv::Mat> vec_V_(NImages);
			std::for_each(
				vec_V_.begin() , 
				vec_V_.end() , 
				[&NRows , &NCols](cv::Mat& _Rs){
					_Rs.create(NRows , NCols , CV_64FC1);
					_Rs.setTo(0);
			});


			for(int niter = 0; niter < NIters; niter++ ){

				///updating P:
				///
				const double SigmaP = (!niter) ? (1 + Sigma) : Sigma;


				///P' = P + Sigma*Grad(U)
				///P(x,y) = P'(x,y)/max(||P(x,y)||,1)
#pragma omp parallel for 
				for(int ny=NPad; ny <NRows-NPad; ny++ ){

					const double* p_U11 = U_.ptr<double>(ny)+NPad, 
						* p_U21 = p_U11+1 ,
						* p_U12 = U_.ptr<double>(ny+1)+NPad;

					cv::Point2d* p_P = P_.ptr<cv::Point2d>(ny)+NPad;

					if(BGWeighted){

						const double *p_G = _Grad.ptr<double>(ny)+NPad ;

						for(int nx =NPad; nx < NCols-NPad; nx++ , 
							p_U11++ , p_U12++ , p_U21++ , p_P++ , p_G++){

								///forward difference: P = P + Sigma*Div(P)
								///
								const double &&px_ = (p_U21[0] - p_U11[0])*SigmaP + p_P[0].x;
								const double &&py_ = (p_U12[0] - p_U11[0])*SigmaP + p_P[0].y;

								///inv-g-weighted normal: 1/(1+|Div(P(x))|/G(x))
								const double &&DInvPNorm = 1.0/(1.0 + std::sqrt(px_*px_ + py_*py_)/p_G[0]) ;

								p_P[0].x = px_ * DInvPNorm;
								p_P[0].y = py_ * DInvPNorm;
						}

					} else {

						for(int nx =NPad; nx < NCols-NPad; nx++ , 
							p_U11++ , p_U12++ , p_U21++ , p_P++){

								///forward difference: P = P + Sigma*Div(P)
								///
								const double &&px_ = (p_U21[0] - p_U11[0])*SigmaP + p_P[0].x;
								const double &&py_ = (p_U12[0] - p_U11[0])*SigmaP + p_P[0].y;

								const double &&DInvPNorm = 1.0/(std::sqrt(px_*px_ + py_*py_)+ 1.0) ;

								p_P[0].x = px_ * DInvPNorm;
								p_P[0].y = py_ * DInvPNorm;
						}
					}

				}


				for(int nimg=0; nimg<NImages ;nimg++){

					///updating V: 
					///
					///	 V := {-lambda , v+sigma*(u-f) , lambda}
					///		=>	lambda			; iff: v+sigma*(u-s)>lambda
					///			v+sigma*(u-s)	; iff: -lambda<v+sigma<lambda
					///			-lambda			; iff: v+sigma*(u-s)<-lambda
					///
#pragma omp parallel for
					for(int ny=NPad ; ny<NRows-NPad ; ny++) {

						double *p_V = vec_V_[nimg].ptr<double>(ny)+NPad ;
						const double *p_U = U_.ptr<double>(ny)+NPad , 
							*p_Src = vec_Src[nimg].ptr<double>(ny)+NPad;

						const unsigned char* p_Mask =  UMask.ptr<unsigned char>(ny)+NPad;

						for(int nx=NPad ; nx<NCols-NPad ; nx++) {

							if(p_Mask[nx]) {

								const double && Vxyn = p_V[nx]+Sigma*(p_U[nx]-p_Src[nx]);

								p_V[nx] = std::max(std::min(Vxyn , Lambda) , -Lambda) ;
							}

						}

					}
				}


#pragma omp parallel for
				for(int ny = NPad; ny < NRows-NPad; ny++ ){

					///updating U:
					///
					double* p_U = U_.ptr<double>(ny)+NPad ;
					const cv::Point2d* p_P11 = P_.ptr<cv::Point2d>(ny)+NPad ,
						*p_P10 = P_.ptr<cv::Point2d>(ny-1)+NPad , 
						*p_P01 = p_P11 - 1;

					const unsigned char* p_UMask = UMask.ptr<unsigned char>(ny)+NPad ;


					///U1 = U + Tau*(-nablaT(P))
					for(int nx = NPad; nx < NCols-NPad; nx++ ,
						p_U++ , p_P01++ , p_P10++ , p_P11++ , p_UMask++){

							if(*p_UMask) {
								double vs_(0.0) ;
								for(int nimg=0; nimg<NImages ; nimg++){
									vs_ += vec_V_[nimg].ptr<double>(ny)[nx] ;	
								}

								///U1 = U + Tau*(-nablaT(P))
								const double&& Unew = *p_U + Tau*(p_P11->x - p_P01->x + p_P11->y - p_P10->y - vs_);

								///U = U2 + Theta*(U2 - U)
								*p_U = Unew + Theta*(Unew - *p_U);
							}
					}
				}
			}


			U_.convertTo(_DepthMap , CV_32FC1);


			return true ;
	}



	bool 
		SceneFlowOptimizer::
		approximateMotionField_L1(
		const cv::Mat& _GrayRefImage) {


			const int NPoints = _vec_x3Points_.size()/3 ,
				NCams = _vec_pPCams_.size() , 
				NPad = NCams/2 ,
				KthRefPCam = NCams/2 ; 


			std::vector<cv::Mat> vec_DepthMaps(NPad*2 , cv::Mat(_DepthMap_.rows , _DepthMap_.cols ,CV_64FC1 )) ;


			const std::vector<float> &Cvec_ = _vec_pPCams_[KthRefPCam]->_Cvec_ ;

			const cv::Size PixSize(1 , 1);


#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints; np++){

				const unsigned char *const p_dMask = _DepthMask_.ptr<unsigned char>(0)+np ;

				if(*p_dMask) {

					float *const px3 = &_vec_x3Points_[np*3] , 
						*const pray = &_vec_x3Rays_[np*3];

					std::vector<float> vec_uvDir(2*NPad ,0) ;

					std::map<int , cv::Mat>::const_iterator it_uFlow = _map_uFlow_.begin() , 
						it_vFlow = _map_vFlow_.begin() ;


					int ncCnt(0);

					for(int nc=0; nc<NCams ; nc++) {
						if(nc!=KthRefPCam){

							const std::vector<float> &P_ = _vec_pPCams_[nc]->_P_ ;

							const float &&x2hx = P_[0]*px3[0]+P_[1]*px3[1]+P_[2]*px3[2]+P_[3] ,
								&&x2hy = P_[4]*px3[0]+P_[5]*px3[1]+P_[6]*px3[2]+P_[7] ,
								&&x2hz = P_[8]*px3[0]+P_[9]*px3[1]+P_[10]*px3[2]+P_[11] ,
								&&x2hz2 = x2hz*x2hz ;

							cv::Point2f x2uv(x2hx/x2hz , x2hy/x2hz) ;


							///step.1: calculate Jacobian matrix by reproject vertices on image plane viewed by compared camera
							///
							const float Jacob[6] = {(P_[0]*x2hz-P_[8]*x2hx)/x2hz2 , 
								(P_[1]*x2hz-P_[9]*x2hx)/x2hz2 , 
								(P_[2]*x2hz-P_[10]*x2hx)/x2hz2 ,
								(P_[4]*x2hz-P_[8]*x2hy)/x2hz2 , 
								(P_[5]*x2hz-P_[9]*x2hy)/x2hz2 , 
								(P_[6]*x2hz-P_[10]*x2hy)/x2hz2 } ;


							///step.2: calculate scalar of directional vector by minimizing ||s*uvDir - uvFlow||
							///		which can be solved by using normal equation suggested by Newcombe(CVPR'2010)
							///
							const float &&x2uDir = Jacob[0]*pray[0]+Jacob[1]*pray[1]+Jacob[2]*pray[2] , 
								&&x2vDir = Jacob[3]*pray[0]+Jacob[4]*pray[1]+Jacob[5]*pray[2] ;

							const float &fuFlow = it_uFlow->second.ptr<float>(0)[np] , &fvFlow = it_vFlow->second.ptr<float>(0)[np] ;


							///s = uvDirT*(uvDir*uvDirT)^-1 * uvFlow
							const float &&sx3Mot = (x2uDir*fuFlow + x2vDir*fvFlow)/(x2uDir*x2uDir + x2vDir*x2vDir) ;


							///step.3: calculate depth on local scene flow generated from current compared frame and referenced frame
							///
							double *const p_Depth = vec_DepthMaps[ncCnt].ptr<double>(0)+np ;

							*p_Depth = std::sqrt((sx3Mot*pray[0]+px3[0] - Cvec_[0])*(sx3Mot*pray[0]+px3[0] - Cvec_[0]) + 
								(sx3Mot*pray[1]+px3[1] - Cvec_[1])*(sx3Mot*pray[1]+px3[1] - Cvec_[1]) + 
								(sx3Mot*pray[2]+px3[2] - Cvec_[2])*(sx3Mot*pray[2]+px3[2] - Cvec_[2])) ;


							ncCnt++ ;

							++it_uFlow , ++it_vFlow ;
						}
					}

				}
			}




			///step.4: integrate depth maps from each compared camera to optimize the referenced depth map
			///		by solving primal dual problem with TV-L1 model
			///
			const cv::Mat InvDMask = 255-_DepthMask_ ;

			for(int npad=0; npad<NPad*2 ; npad++) {

				cv::normalize(vec_DepthMaps[npad] , vec_DepthMaps[npad] , 0.0 , 1.0 , cv::NORM_MINMAX , CV_64FC1 , _DepthMask_) ;
				vec_DepthMaps[npad].setTo(0 , InvDMask) ;

				cv::GaussianBlur(vec_DepthMaps[npad] , vec_DepthMaps[npad] , cv::Size(3 , 3) , 1.0 , 1.0 , cv::BORDER_REFLECT);

#if _DEBUG_SEE_MULTIPLE_DMAPS_
				char wname[256];
				sprintf(wname , "%d-depth\0" , npad) ;
				_SEE_IMAGE_(vec_DepthMaps[npad] ,  std::string(wname));
#endif
			}


			cv::Mat Grad0 , Grad ;
			if(_GrayRefImage.type()!=CV_32FC1 && _GrayRefImage.type()!=CV_32F){
				_GrayRefImage.convertTo(Grad0 , CV_32FC1 ,  1.0/255);
			} else {
				_GrayRefImage.copyTo(Grad0);
			}
		
			
			cv::bilateralFilter(Grad0 , Grad , 5 , 3.0 , 20);
			

			//cv::medianBlur(Grad , Grad , 3);
			//cv::GaussianBlur(Grad , Grad , cv::Size(5 , 5) , 1.0 , 1.0, cv::BORDER_REFLECT);
			cv::Sobel(Grad , Grad , CV_32FC1 , 1 , 0 , 3);
			cv::Sobel(Grad , Grad , CV_32FC1 , 0 , 1 , 3);
			Grad.convertTo(Grad , CV_64FC1) ;

			cv::exp(-10.0*cv::abs(Grad) , Grad) ;


			const int UInd = NPad ;
			const bool BGWeighted = true ;


			solvePrimalDualL1(_DepthMap_ , vec_DepthMaps , _DepthMask_ , UInd , Grad , BGWeighted ,
				50 , 1.0/1000 , 1.0 , 0.10);


			//cv::GaussianBlur(_DepthMap_ , _DepthMap_ , cv::Size(5 , 5) , 1.0 , 1.0 , cv::BORDER_REFLECT);
			cv::Mat filterDepthMap ;
			cv::bilateralFilter(_DepthMap_ , filterDepthMap , 5 , 1.0 , 50);

			cv::normalize(filterDepthMap , _DepthMap_ , _DepthBox_[0] , _DepthBox_[1] , cv::NORM_MINMAX , CV_32FC1 , _DepthMask_ );
			_DepthMap_.setTo(_DepthBox_[1] , InvDMask);



			///step.5: update vertices by using rays and new depth map
			///
			const unsigned char *const pDepthMask = _DepthMask_.ptr<unsigned char>(0) ;

#pragma omp parallel for schedule(dynamic, 1) 
			for(int np=0 ; np<NPoints ; np++) {
				if(pDepthMask[np]) {

					const float *const pray = &_vec_x3Rays_[np*3] ;
					float *const px3 = &_vec_x3Points_[np*3] , 
						*const p_Depth = _DepthMap_.ptr<float>(0)+np ;

					px3[0] = *p_Depth*pray[0] + Cvec_[0] , 
						px3[1] = *p_Depth*pray[1] + Cvec_[1] , 
						px3[2] = *p_Depth*pray[2] + Cvec_[2] ;
				}
			}


			return true ;
	}


}