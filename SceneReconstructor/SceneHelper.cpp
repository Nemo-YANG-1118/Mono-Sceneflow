
#include <stlplus3/filesystemSimplified/file_system.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <iterator>
#include <utility>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <cmath>

#include "SceneHelper.h"
#include "SceneConfiguration.h"


#undef _SEE_IMAGE_
#define _SEE_IMAGE_(img_ , sWinName_)\
	cv::namedWindow(sWinName_ , 1) ;\
	cv::imshow(sWinName_ , img_);\
	cv::waitKey(0);\
	cv::destroyWindow(sWinName_);


#ifndef _CHECK_FOLDER_
#undef _CHECK_FOLDER_
#endif
#define _CHECK_FOLDER_(sdir_ , chFunctionName_)\
	if(!stlplus::folder_exists(sdir_) || stlplus::folder_empty(sdir_)){\
	fflush(stderr) ;\
	fprintf(stderr, "Error in <SceneHelper/%s>" , chFunctionName_);\
	fflush(stderr);\
	exit(0);}\


#undef _PRINT_FLOATVEC_
#define _PRINT_FLOATVEC_(_vec_data)\
	fflush(stdout);\
	std::copy(_vec_data.begin() , _vec_data.end() , (std::ostream_iterator<float>(std::cout , "\t\t"))) ;\
	std::cout<<std::endl ;\
	fflush(stdout);\


#undef _SET_BORDER_0_
#define _SET_BORDER_0_(_img , _npix)\
	_img.colRange(0 , _npix).setTo(0) ;\
	_img.colRange(_img.cols-_npix , _img.cols).setTo(0) ;\
	_img.rowRange(0 , _npix).setTo(0) ;\
	_img.rowRange(_img.rows-_npix , _img.rows).setTo(0);\



namespace stereoscene{


	bool
		SceneHelper::
		blackMapBorder(
		cv::Mat& _SrcImage ,
		const int NPixels) {

			_SET_BORDER_0_(_SrcImage , std::max(NPixels , _BORDER_NGAP_));

			return true ;
	}


	bool
		SceneHelper::
		generateConvexMask(
		cv::Mat &_ConvexMask ,
		const cv::Mat& _BinMask) {


			const int NWidth = _BinMask.cols , NHeight = _BinMask.rows ;
			const int NPoints = NWidth*NHeight ;

			if(_ConvexMask.empty()) {
				_BinMask.copyTo(_ConvexMask) ;
			}


			std::vector<cv::Point> Pointset ;
			Pointset.reserve(NWidth*NHeight);

			for(int nh=0 ; nh<NHeight ; nh++) {
				const unsigned char* pmask = _BinMask.ptr<unsigned char>(nh) ;
				for(int nw=0 ; nw<NWidth ; nw++ , pmask++) {
					if(*pmask) {
						Pointset.push_back(cv::Point(nw , nh));
					}
				}
			}
			std::vector<cv::Point>(Pointset).swap(Pointset);


			///generate a convex mask in order to turn the full depth map into
			///scalar field existed on surface
			std::vector<cv::Point> hull ;
			cv::convexHull(Pointset , hull);

			const cv::Point* phull = hull.data() ;
			cv::fillConvexPoly(_ConvexMask , phull , hull.size() , cv::Scalar(255 , 255 , 255));


			return true ;
	}


	bool
		SceneHelper::
		buildBinaryMask(
		cv::Mat& _BinMask,
		const cv::Mat& _SrcMask,
		const int NSize) {


			const cv::Size MaskSize = _SrcMask.size() ;
			const int NWidth = MaskSize.width ,
				NHeight = MaskSize.height ;


			cv::Mat VerMask(MaskSize , CV_8UC1) , HorMask(MaskSize , CV_8UC1) ;
			VerMask.setTo(0) , HorMask.setTo(0) ;

#pragma omp parallel for schedule(dynamic , 1)
			for(int nh=0 ; nh<NHeight ; nh++) {

				const unsigned char* phor = _SrcMask.ptr<unsigned char>(nh) ;

				unsigned int kl_(0) ,kr_(0);
				for(int nw=0 ; nw<NWidth ; nw++ , phor++) {
					if(*phor) {
						if(!kl_) {
							kl_=nw ;
						} else {
							kr_=nw ;
						}
					}
				}

				if(kr_>kl_) {
					memset(HorMask.ptr<unsigned char>(nh)+kl_ , 255 , sizeof(unsigned char)*(kr_-kl_));
				}
			}

			//
			cv::dilate(HorMask, HorMask ,cv::getStructuringElement(cv::MORPH_CROSS , cv::Size(NSize ,NSize)));
			cv::erode(HorMask, HorMask ,cv::getStructuringElement(cv::MORPH_CROSS , cv::Size(NSize , NSize)));


			cv::Mat SrcMask_1 = _SrcMask.t() ;

			cv::transpose(VerMask , VerMask) ;

#pragma omp parallel for schedule(dynamic , 1)
			for(int nh=0 ; nh<NWidth ; nh++) {

				const unsigned char* phor = SrcMask_1.ptr<unsigned char>(nh) ;

				unsigned int kl_(0) ,kr_(0);
				for(int nw=0 ; nw<NHeight ; nw++ , phor++) {
					if(*phor) {
						if(!kl_) {
							kl_=nw ;
						} else {
							kr_=nw ;
						}
					}
				}

				if(kr_>kl_) {
					memset(VerMask.ptr<unsigned char>(nh)+kl_ , 255 , sizeof(unsigned char)*(kr_-kl_));
				}
			}

			cv::transpose(VerMask , VerMask);
			//cv::flip(VerMask ,VerMask , 0);

			//
			cv::dilate(VerMask, VerMask ,cv::getStructuringElement(cv::MORPH_CROSS , cv::Size(NSize , NSize)));
			cv::erode(VerMask, VerMask ,cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(NSize , NSize)));

			cv::bitwise_and(VerMask , HorMask , _BinMask);


			return true ;
	}



	bool
		SceneHelper::
		generateVisualMask(
		cv::Mat& _VisHullMask , 
		const cv::Mat& _SrcImage , 
		const int KScene , 
		const bool BSave ,
		const std::string& _sdir ,
		const std::string& _sname ) {


			const cv::Size ImageSize = _SrcImage.size() ;
			const int NHeight = ImageSize.height , 
				NWidth = ImageSize.width ,
				NChannels = _SrcImage.channels() ;


			cv::Mat SrcRGB , SrcHSV , SrcGray ;

			cv::medianBlur(_SrcImage , SrcRGB , 3);
			if(NChannels>1) {
				cv::cvtColor(SrcRGB , SrcHSV , cv::COLOR_BGR2HSV) ;
				cv::cvtColor(SrcRGB , SrcGray , cv::COLOR_BGR2GRAY);
			}


			///gray mask
			cv::Mat	GMask(ImageSize , CV_8UC1) ;
			cv::threshold(SrcGray , GMask , 30 , 255 , cv::THRESH_BINARY) ;

			SceneHelper::buildBinaryMask(GMask , GMask , 3);

			
			if(NChannels>1) {

				///hsv mask
				///
				cv::Mat HSVMask(ImageSize , CV_8UC1) ;


				const cv::Vec3b *const phsv = SrcHSV.ptr<cv::Vec3b>(0) ;
				unsigned char *const pvmask = HSVMask.ptr<unsigned char>(0);

#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0; np<NWidth*NHeight ; np++) {
					pvmask[np] = (phsv[np].val[2]>30)? unsigned char(255) : unsigned char(0);  
				}

				SceneHelper::buildBinaryMask(HSVMask , HSVMask);

				//cv::bitwise_and(GMask , HSVMask , _VisHullMask) ;

				_VisHullMask &= (GMask&HSVMask);

			} else {
				GMask.copyTo(_VisHullMask) ;
			}


			cv::erode(_VisHullMask , _VisHullMask , cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3)));


			_SET_BORDER_0_(_VisHullMask , _BORDER_NGAP_);


			//_SEE_IMAGE_(_VisHullMask , "vmask");			

			if(BSave) {

				if(!stlplus::folder_exists(_sdir)) {
					stlplus::folder_create(_sdir) ;
				} 


				std::string sdir_File ;
				{
					char fname[256]={0};
					if(KScene<10){
						sprintf(fname , "%s/00%d\0" , _sdir.c_str() , KScene) ;
					} else if(KScene>=10 && KScene<100) {
						sprintf(fname , "%s/0%d\0" , _sdir.c_str() , KScene) ;
					} else if(KScene>=100 && KScene<1000){
						sprintf(fname , "%s/%d\0" , _sdir.c_str() , KScene) ;
					} else {
						fflush(stdout) ;
						fprintf(stdout , "\nError <SceneHelper.cpp/saveVisMask> : too much data to save\n");
						fflush(stderr) ;

						return false ;
					}

					sdir_File.assign(fname) ;
				}


				if(!stlplus::folder_exists(sdir_File)) {
					stlplus::folder_create(sdir_File) ;
				} 

				std::string sname_VisMask(sdir_File);
				sname_VisMask.append("/"+_sname) ;

				cv::Mat ConvexMask ;
				generateConvexMask(ConvexMask , _VisHullMask) ;

				cv::imwrite(sname_VisMask , ConvexMask);

			}


			return true ;
	}



}



namespace stereoscene{


	int
		SceneHelper::
		findStructurefromMotion(
		std::vector<std::vector<float>>& vec_x3Points, 
		std::map<int , PhotoCamera>& map_PCams , 
		const int kCurPCams ,
		const int NPads ,
		const cv::Size ImageSize , 
		const std::vector<std::vector<float>>& vec_StructurePoints ,
		const std::map<int, PhotoCamera>& map_MotionPCams) {


			const int NPoints = vec_StructurePoints.size() ,
				NPCams = map_MotionPCams.size() ,
				NWidth = ImageSize.width ,
				NHeight = ImageSize.height ;


			if(kCurPCams+NPads>=NPCams){
				fflush(stdout);
				fprintf(stdout , "\nError : Required cameras' number is much more than existed\n") ;
				fflush(stdout);
				exit(0);
			}

			vec_x3Points.clear() , map_PCams.clear();

			for(int nc=0; nc<=kCurPCams+NPads ; nc++) {
				std::map<int , PhotoCamera>::const_iterator it_MotPCams = map_MotionPCams.begin() ;
				std::advance(it_MotPCams , nc) ;

				map_PCams.insert(std::make_pair(it_MotPCams->first , 
					PhotoCamera(it_MotPCams->second._K_ , it_MotPCams->second._R_ , it_MotPCams->second._Tvec_)));
			}


			std::map<int , PhotoCamera>::const_iterator it_rPCams = map_PCams.find(kCurPCams) ;
			if(it_rPCams==map_PCams.end()) {
				fflush(stdout);
				fprintf(stdout , "\nError : <SceneHelper/findStructurefromMotion>\n") ;
				fflush(stdout);
				exit(0);
			}

			const std::vector<float> &Pvec = it_rPCams->second._P_ ;
			const std::vector<float> &Cvec = it_rPCams->second._Cvec_ ;
			const std::vector<float> &Kvec = it_rPCams->second._K_ ;

			vec_x3Points.reserve(NPoints) ;

#pragma omp parallel for schedule(dynamic , 1)
			for(int np3=0 ; np3<NPoints ; np3++){
				const float *const px3 = &vec_StructurePoints[np3][0] ;

				const float wx = Pvec[8]*px3[0]+Pvec[9]*px3[1]+Pvec[10]*px3[2]+Pvec[11];
				const float ux = (Pvec[0]*px3[0]+Pvec[1]*px3[1]+Pvec[2]*px3[2]+Pvec[3])/wx ;
				const float vx = (Pvec[4]*px3[0]+Pvec[5]*px3[1]+Pvec[6]*px3[2]+Pvec[7])/wx ;

				if(ux<NWidth-2 && ux>1 && vx<NHeight-2 && vx>1){
#pragma omp critical
					{
						vec_x3Points.push_back(std::vector<float>(px3 , px3+3));
					}
				}
			}


			return vec_x3Points.size() ;
	}



	int 
		SceneHelper::
		loadStructureAndMotion(
		std::vector<std::vector<float>>& vec_StructurePoints , 
		std::map<int , PhotoCamera>& map_MotionPCams ,
		const std::string& sdir) {

			if(!stlplus::folder_exists(sdir)) {
				fflush(stdout);
				fprintf(stdout , "\nError: x3Points and PCameras' folder is wrong : %s\n"  ,sdir.c_str()) ;
				fflush(stdout);
				exit(0) ;
			}

			const std::string sname_x3Points = sdir + "/x3Points.txt" , 
				sname_PCameras = sdir + "/PCameras.txt" ;

			if(!stlplus::file_exists(sname_x3Points)) {
				fflush(stdout);
				fprintf(stdout , "\nError: x3Points' data is null: %s\n"  ,sname_x3Points.c_str()) ;
				fflush(stdout);
				exit(0) ;
			}

			if(!stlplus::file_exists(sname_PCameras)) {
				fflush(stdout);
				fprintf(stdout , "\nError: PCameras' data is null: %s\n"  ,sname_x3Points.c_str()) ;
				fflush(stdout);
				exit(0) ;
			}

			std::fstream fs;

			fs.open(sname_x3Points , std::ios::in) ;
			int NPoints(0);
			fs>>NPoints ;
			vec_StructurePoints.resize(NPoints ,std::vector<float>(3)) ;

			std::for_each(vec_StructurePoints.begin() , vec_StructurePoints.end() , 
				[&fs](std::vector<float>& structurePoints) mutable {
					fs>>structurePoints[0]>>structurePoints[1]>>structurePoints[2] ;
			});
			fs.close() ;



			fs.open(sname_PCameras , std::ios::in) ;
			int NCams(0);
			fs>>NCams ;
			map_MotionPCams.clear() ;

			for(int ncam=0 ; ncam<NCams ; ncam++) {

				std::vector<float> Kvec_(9) , Rvec_(9), tvec_(3) ;

				std::for_each(Kvec_.begin() ,Kvec_.end() , [&fs](float& _k){
					fs>>_k ;
				});

				std::for_each(Rvec_.begin() , Rvec_.end() , [&fs](float& _r){
					fs>>_r ;
				}) ;

				std::for_each(tvec_.begin() , tvec_.end() , [&fs](float& _t){
					fs>>_t ;
				});

				map_MotionPCams.insert(std::make_pair(ncam , PhotoCamera(Kvec_ , Rvec_ , tvec_))) ;
			}

			fs.close() ;


			return vec_StructurePoints.size() ;
	}



	int
		SceneHelper::
		loadPCams(
		std::map<int , PhotoCamera>& _map_PCams , 
		const std::string& _sdir , 
		const int NThresh) {


			if(!stlplus::folder_exists(_sdir)) {
				fflush(stderr) ;
				fprintf(stderr , "Error in SceneHelper.cpp : < %s > is wrong\n" , _sdir.c_str()) ;
				fflush(stderr) ;

				return false ;
			}

			const std::string sname_Cams = _sdir+"/PCameras.txt" ;

			if(!stlplus::file_exists(sname_Cams)) {
				fflush(stderr) ;
				fprintf(stderr , "Error in SceneHelper.cpp : < %s > dosen't existed\n" , sname_Cams.c_str()) ;
				fflush(stderr) ;

				return false ;
			}


			int NCams(0) ;
			std::fstream fs_Cams(sname_Cams , std::ios::in) ;
			fs_Cams>>NCams ; 

			if(NCams<NThresh) {
				fs_Cams.close();
				return false ;
			}

			for(int ncam=0 ; ncam<NCams ; ncam++) {
				std::vector<float> Kvec_(9) , Rvec_(9), tvec_(3) ;

				std::for_each(Kvec_.begin() ,Kvec_.end() , [&fs_Cams](float& _k){
					fs_Cams>>_k ;
				});

				std::for_each(Rvec_.begin() , Rvec_.end() , [&fs_Cams](float& _r){
					fs_Cams>>_r ;
				}) ;

				std::for_each(tvec_.begin() , tvec_.end() , [&fs_Cams](float& _t){
					fs_Cams>>_t ;
				});

				_map_PCams.insert(std::make_pair(ncam , PhotoCamera(Kvec_ , Rvec_ , tvec_))) ;

#if 0
				///to see P
				PhotoCamera PCam(Kvec_ , Rvec_ , tvec_) ;
				_PRINT_FLOATVEC_(PCam._P_) ;
#endif
			}


			fs_Cams.close();


			return NCams ;
	}



	int 
		SceneHelper::
		loadx3Points(
		std::vector<std::vector<float>>& _vec_x3Points ,
		const std::string& _sdir , 
		const int NThresh) {

			if(!stlplus::folder_exists(_sdir)) {
				fflush(stderr) ;
				fprintf(stderr , "Error: < %s > is wrong\n" , _sdir.c_str()) ;
				fflush(stderr) ;

				return false;
			}

			const std::string sname_x3Points = _sdir+"/x3Points.txt" ;

			if(!stlplus::file_exists(sname_x3Points)) {
				fflush(stderr) ;
				fprintf(stderr , "Error: < %s > dosen't existed\n" , sname_x3Points.c_str()) ;
				fflush(stderr) ;

				return false ;
			}

			int NPoints(0);
			std::fstream fs_x3Points(sname_x3Points , std::ios::in) ;
			fs_x3Points>>NPoints ;


			if(NPoints<NThresh) {
				fs_x3Points.close();
				return false ;
			}


			_vec_x3Points.resize(NPoints , std::vector<float>(3));
			std::for_each(_vec_x3Points.begin() , _vec_x3Points.end() , 
				[&fs_x3Points](std::vector<float>& _x3){
					fs_x3Points>>_x3[0]>>_x3[1]>>_x3[2];
			}) ;

			fs_x3Points.close() ;


			return NPoints ;
	}


	bool
		SceneHelper::
		loadImageSpecifiedSize(
		cv::Mat& _DstImage , 
		const std::string& sdir_Image , 
		const cv::Size& _ImageSize ,
		const int Kth ,
		const bool BGray , 
		const bool BResave , 
		const std::string _sdir_Resave){


			_CHECK_FOLDER_(sdir_Image , "loadImageSpecifiedSize") ;

			std::vector<std::string> &vec_fileNames = stlplus::folder_files(sdir_Image);

			const int NFiles = vec_fileNames.size() ;

			for(int nf=0 , imgCnt=0 ; nf<NFiles ; nf++) {

				std::string& sext_ = stlplus::extension_part(vec_fileNames[nf]) ;

				if(sext_=="bmp"||sext_=="BMP"||sext_=="jpg"||sext_=="JPG"||sext_=="png"||sext_=="PNG") {
					if(imgCnt==Kth) {
						if(BGray) {
							(cv::imread(sdir_Image+"/"+vec_fileNames[nf], cv::IMREAD_GRAYSCALE)).copyTo(_DstImage) ;
						} else {
							(cv::imread(sdir_Image+"/"+vec_fileNames[nf] )).copyTo(_DstImage);
						}

						if(_ImageSize.height!=_DstImage.rows || _ImageSize.width!=_DstImage.cols){
							cv::resize(_DstImage , _DstImage , _ImageSize , 0, 0 , cv::INTER_CUBIC) ;
						}

						if(BResave) {
							char chdir[256];
							if(nf>=0 && nf<10) {
								sprintf(chdir , "%s/00%d\0" , _sdir_Resave.c_str() , nf);
							} else if(nf<100 && nf>=10) {
								sprintf(chdir , "%s/0%d\0" , _sdir_Resave.c_str() , nf);
							} else if(nf<1000 && nf>=100) {
								sprintf(chdir , "%s/%d\0" , _sdir_Resave.c_str() , nf);
							}

							std::string sdir_folder(chdir);
							if(!stlplus::folder_exists(sdir_folder)) {
								stlplus::folder_create(sdir_folder);
							}

							std::string sname_Resave = sdir_folder+"/TexImage.bmp" ;

							cv::imwrite(sname_Resave , _DstImage) ;
						}

						break; 
					}

					imgCnt++ ;
				}

			}


			return true ;
	}



	bool
		SceneHelper::
		saveDepthMap(
		const int KthCurPCam ,
		const bool BTxTFormat , 
		const bool BEntireMapSaved ,
		const std::string& _sdir ,
		const std::string& _sname_DepthMap ,
		const cv::Mat& _DepthMap ,
		const cv::Mat& _DepthMask) {


			if(!stlplus::folder_exists(_sdir)) {
				stlplus::folder_create(_sdir) ;
			} 


			std::string sdir_File ;
			{
				char fname[256]={0};
				if(KthCurPCam<10){
					sprintf(fname , "%s/00%d\0" , _sdir.c_str() , KthCurPCam) ;
				} else if(KthCurPCam>=10 && KthCurPCam<100) {
					sprintf(fname , "%s/0%d\0" , _sdir.c_str() , KthCurPCam) ;
				} else if(KthCurPCam>=100 && KthCurPCam<1000){
					sprintf(fname , "%s/%d\0" , _sdir.c_str() , KthCurPCam) ;
				} else {
					fflush(stdout) ;
					fprintf(stdout , "\nError <SceneHelper.cpp/saveDepthMap> : too much data to save\n");
					fflush(stderr) ;

					return false ;
				}

				sdir_File.assign(fname) ;
			}


			if(!stlplus::folder_exists(sdir_File)) {
				stlplus::folder_create(sdir_File) ;
			} 


			std::string sname_DMap(sdir_File) ;
			if(BTxTFormat) {
				sname_DMap.append("/"+_sname_DepthMap) ;

				std::fstream fs(sname_DMap , std::ios::out) ;

				std::copy(_DepthMap.ptr<float>(0) , 
					_DepthMap.ptr<float>(0)+_DepthMap.cols*_DepthMap.rows , 
					(std::ostream_iterator<float>(fs , "\t")));

				fs.close();

			} else {
				sname_DMap.append("/"+_sname_DepthMap) ;

				const int NWidth = _DepthMap.cols , NHeight = _DepthMap.rows ;

				cv::Mat NormDepthMap , UDMap=cv::Mat::zeros(NHeight, NWidth , CV_8UC1) ;

				if(BEntireMapSaved) {
					cv::normalize(_DepthMap , NormDepthMap , 2.0  ,253.0 , cv::NORM_MINMAX , CV_32FC1 , _DepthMask) ;
					NormDepthMap.convertTo(UDMap , CV_8UC1);
					UDMap.setTo(0 , 255-_DepthMask) ;

				} else {

					cv::Mat ConvexBoundMask ;
					generateConvexMask(ConvexBoundMask , _DepthMask);

					cv::normalize(_DepthMap , NormDepthMap , 2.0 , 253.0 , cv::NORM_MINMAX , CV_32FC1 , ConvexBoundMask);
					NormDepthMap.convertTo(UDMap , CV_8UC1) ;
					UDMap.setTo(0 , 255-ConvexBoundMask) ;
				}

				cv::imwrite(sname_DMap , UDMap);
			}


			return true;
	}



	bool
		SceneHelper::
		saveReprojectiveErrorMap(
		const int KthCurPCam ,
		const cv::Mat& _ErrorMap ,
		const std::string& _sdir , 
		const std::string& _sname) {


			if(!stlplus::folder_exists(_sdir)) {
				stlplus::folder_create(_sdir);
			}

			std::string sdir_File ;
			{
				char fname[256]={0};
				if(KthCurPCam<10){
					sprintf(fname , "%s/00%d\0" , _sdir.c_str() , KthCurPCam) ;
				} else if(KthCurPCam>=10 && KthCurPCam<100) {
					sprintf(fname , "%s/0%d\0" , _sdir.c_str() , KthCurPCam) ;
				} else if(KthCurPCam>=100 && KthCurPCam<1000){
					sprintf(fname , "%s/%d\0" , _sdir.c_str() , KthCurPCam) ;
				} else {
					fflush(stdout) ;
					fprintf(stdout , "\nError <SceneHelper.cpp/saveDepthMap> : too much data to save\n");
					fflush(stderr) ;

					return false ;
				}

				sdir_File.assign(fname) ;
			}


			if(!stlplus::folder_exists(sdir_File)) {
				stlplus::folder_create(sdir_File) ;
			} 


			const int NPoints = _ErrorMap.rows*_ErrorMap.cols ;

			cv::Mat ErrorMask(_ErrorMap.size() , CV_8UC1) ;
			unsigned char* pemask = ErrorMask.ptr<unsigned char>(0);

			std::for_each(_ErrorMap.ptr<float>(0) , _ErrorMap.ptr<float>(0)+NPoints ,
				[pemask](const float& _ferror) mutable{
					*pemask++ = (_ferror>1e-5) ? unsigned char(255) : unsigned char(0) ;
			}) ;

			cv::Mat MeanError , StdvError ;
			cv::meanStdDev(_ErrorMap , MeanError , StdvError , ErrorMask);

			fflush(stdout) ;
			fprintf(stdout , "\t-- Mean and stdvar reprojective-error is : %.3f , %.3f --\n",
				MeanError.at<double>(0) , StdvError.at<double>(0)) ;
			fflush(stdout) ;


			std::string sname ;

			std::string& sext = stlplus::extension_part(_sname) ;
			if(sext!="txt"){
				if(sext.empty()){
					sname = _sname+std::string(".txt") ;
				} else {
					sname = stlplus::basename_part(_sname).append(".txt") ;
				}
			} else {
				sname = _sname ;
			}

			std::fstream fs(sdir_File+"/"+sname , std::ios::out) ;
			fs<<_ErrorMap.cols*_ErrorMap.rows<<"\n";
			fs<<MeanError.at<double>(0)<<"\t"<<StdvError.at<double>(0)<<"\n" ;
			std::copy(_ErrorMap.ptr<float>(0) , _ErrorMap.ptr<float>(0)+NPoints , 
				(std::ostream_iterator<float>(fs , "\t"))) ;
			fs.close() ;


			return true ;
	}



	bool 
		SceneHelper::
		saveImagesAndMapsVideo(
		const cv::Size& _MapSize ,
		const std::string& _sname_Video ,
		const std::string& _sdir ,
		const std::string& _sdir_TexImages,
		const std::string& _sname_Image1 , 
		const std::string& _sname_Image2 ,
		const std::string& _sname_Image3,
		const std::string& _sname_Image4) {


			if(!stlplus::folder_exists(_sdir)) {
				fflush(stdout) ;
				fprintf(stdout , "\nError <SceneHelper.cpp/saveImagesAndMapsVideo> : data dosen't exist\n");
				fflush(stderr) ;

				return false ;
			}


			cv::Size MapSize=_MapSize ;
			if(MapSize.width>640 || MapSize.height>480){
				MapSize.width /= 2 , MapSize.height /= 2 ;
			}

			int cntw(2),cnth(0);
			if(!_sname_Image1.empty()) {
				cntw++,cnth++ ;
			}
			if(!_sname_Image2.empty()) {
				cntw++ ;
			}
			if(!_sname_Image3.empty()) {
				cnth++ ;
			}



			cv::Size VideoSize = cv::Size(cntw*MapSize.width , cnth*MapSize.height) ;

			fflush(stdout) ;
			fprintf(stdout,"Frame Size in Video : %d x %d\n" , VideoSize.width , VideoSize.height);
			fflush(stderr) ;


			const std::vector<std::string>& vec_Dirs = stlplus::folder_subdirectories(_sdir);

			//const std::vector<std::string>& vec_Dirs = stlplus::folder_files(_sname_Image1);
			const int NDirs = vec_Dirs.size();


			cv::VideoWriter VidWriter(
				std::string(_sdir+"/"+_sname_Video) , 
				-1 ,
				10 ,
				VideoSize);

			cv::namedWindow("VideoImage" , 1);
			for(int nd=2+int(_PER_BUNDLE_NFRAMES_/2) ; nd<NDirs ; nd++) {
				const std::string sdirSub =  _sdir +"/"+vec_Dirs[nd];

				cv::Mat VideoImage(cv::Mat::zeros(VideoSize , CV_8UC3 ));
	
				if(!_sdir_TexImages.empty()) {
					//std::cout<<sdirSub + "/" + _sdir_TexImages<<"\n" ;
					cv::Mat &Image0 = cv::Mat(VideoImage , cv::Rect(0 , 0 ,  MapSize.width*2 ,MapSize.height*2)) ;
					cv::resize(cv::imread(sdirSub + "/" + _sdir_TexImages) , Image0 ,cv::Size(MapSize.width*2 , MapSize.height*2),  0 , 0 , cv::INTER_CUBIC) ;
				}

				if(!_sname_Image1.empty()) {
					cv::Mat &Image1 = cv::Mat(VideoImage,cv::Rect(MapSize.width*2,0,MapSize.width,MapSize.height)) ;
					cv::resize(cv::imread(sdirSub+"/"+_sname_Image1),Image1,MapSize,0,0,cv::INTER_CUBIC) ;
				}

				if(!_sname_Image2.empty()) {
					cv::Mat	&Image2 = cv::Mat(VideoImage,cv::Rect(MapSize.width*3 ,0,MapSize.width,MapSize.height));
					cv::resize(cv::imread(sdirSub+"/"+_sname_Image2),Image2,MapSize,0,0,cv::INTER_CUBIC) ;
				}

				if(!_sname_Image3.empty()) {

					cv::Mat &Image3 = cv::Mat(VideoImage,cv::Rect(MapSize.width*2 ,MapSize.height,MapSize.width,MapSize.height));
					cv::resize(cv::imread(sdirSub+"/"+_sname_Image3),Image3,MapSize,0,0,cv::INTER_CUBIC) ;
				}

				if(!_sname_Image4.empty()) {
					cv::Mat &Image4 = cv::Mat(VideoImage,cv::Rect(MapSize.width*3,MapSize.height,MapSize.width,MapSize.height));
					cv::resize(cv::imread(sdirSub+"/"+_sname_Image4),Image4,MapSize,0,0,cv::INTER_CUBIC)  ;
				}


				VidWriter.write(VideoImage);

				cv::imshow("VideoImage" , VideoImage);

				cv::waitKey(20);
			}

			VidWriter.release() ;

			cv::destroyWindow("VideoImage");

			return true ;
	}



	bool
		SceneHelper::
		seeHistogram(
		const cv::Mat& _Hist , 
		const std::string& _sname_hist) {


			const int NBins = _Hist.rows , 
				NDims = _Hist.cols ;

			double minhval(0) , maxhval(0) ;
			cv::minMaxLoc(_Hist , &minhval , &maxhval , NULL, NULL);


			const int NStep = 30 ,
				NStack = 500/maxhval ;


			cv::Mat HistImage(600 , 20+(30*NBins) , CV_8UC3) ;
			HistImage.setTo(255);


			for(int nbin=0 ; nbin<NBins ; nbin++) {
				const float *const phist = _Hist.ptr<float>(nbin) ;

				cv::line(HistImage , cv::Point2i(20+NStep*nbin , 600) , cv::Point2i(20+NStep*nbin , 600-NStack*(phist[0])) , cv::Scalar(155 , 255 , 0) , 10) ;
			}

			if(_sname_hist.empty()) {
				_SEE_IMAGE_(HistImage , "See Hist Distribution");
			} else {
				_SEE_IMAGE_(HistImage , _sname_hist);
			}


			return true ;
	}


}


#include <VTK/vtkSmartPointer.h>
#include <VTK/vtkPoints.h>
#include <VTK/vtkPolyData.h>
#include <VTK/vtkPolyDataMapper.h>
#include <VTK/vtkActor.h>
#include <VTK/vtkRenderer.h>
#include <VTK/vtkRenderWindow.h>
#include <VTK/vtkRenderWindowInteractor.h>
#include <VTK/vtkInteractorStyleTrackballCamera.h>
#include <VTK/vtkVertexGlyphFilter.h>
#include <VTK/vtkConeSource.h>
#include <VTK/vtkProperty.h>
#include <VTK/vtkFloatArray.h>
#include <VTK/vtkProperty.h>
#include <VTK/vtkCamera.h>
#include <VTK/vtkBMPWriter.h>
#include <VTK/vtkWindowToImageFilter.h>


#include "SceneHelper.h"


#undef _VSP_
#define _VSP_(ClassName_ , ObjName_)\
	vtkSmartPointer<ClassName_> ObjName_ = vtkSmartPointer<ClassName_>::New() ;



#ifdef _DISP_POLYMAPPER_
#undef _DISP_POLYMAPPER_
#endif
#define _DISP_POLYMAPPER_(PolyMapper_ , BInverse ,sname_)\
	_VSP_(vtkActor , Actor_) ;\
	Actor_->SetMapper(Mapper_) ;\
	if(BInverse)Actor_->SetScale(1.0 , -1.0 , -1.0); \
	_VSP_(vtkRenderer , Ren_) ;\
	Ren_->AddActor(Actor_) ;\
	_VSP_(vtkRenderWindow , RenWin_) ;\
	RenWin_->AddRenderer(Ren_);\
	RenWin_->SetSize(800 , 600);\
	RenWin_->Render() ;\
	RenWin_->SetWindowName(_sname_x3Points.c_str()) ;\
	_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;\
	_VSP_(vtkRenderWindowInteractor , IRen_) ;\
	IRen_->SetInteractorStyle(Trackball_);\
	IRen_->SetRenderWindow(RenWin_);\
	IRen_->Initialize() ;\
	IRen_->Start();\



namespace stereoscene{


	bool
		SceneHelper::
		seeTwoPointsets(
		const std::vector<float>& _x3Points0 ,
		const std::vector<float>& _x3Points1 ,
		const bool BInverse ,
		const std::string& _sname_x3Points){


			const int NPoints0 = _x3Points0.size()/3 , 
				NPoints1 = _x3Points1.size()/3 ;

			_VSP_(vtkPoints , Points0) ;
			Points0->SetNumberOfPoints(NPoints0) ;
			Points0->SetDataTypeToFloat() ;
			memcpy(Points0->GetVoidPointer(0) , _x3Points0.data()  , sizeof(float)*3*NPoints0);				
			_VSP_(vtkPoints , Points1) ;
			Points1->SetNumberOfPoints(NPoints1) ;
			Points1->SetDataTypeToFloat() ;
			memcpy(Points1->GetVoidPointer(0) , _x3Points1.data()  , sizeof(float)*3*NPoints1);


			_VSP_(vtkPolyData , Polys0) ;
			Polys0->SetPoints(Points0) ;
			_VSP_(vtkPolyData , Polys1) ;
			Polys1->SetPoints(Points1) ;


			_VSP_(vtkVertexGlyphFilter , Vertex0) ;
			Vertex0->SetInputData(Polys0) ;
			Vertex0->Update() ;
			_VSP_(vtkVertexGlyphFilter , Vertex1) ;
			Vertex1->SetInputData(Polys1) ;
			Vertex1->Update() ;


			_VSP_(vtkPolyDataMapper , Mapper0);
			Mapper0->SetInputConnection(Vertex0->GetOutputPort());
			Mapper0->Update();
			_VSP_(vtkPolyDataMapper , Mapper1);
			Mapper1->SetInputConnection(Vertex1->GetOutputPort());
			Mapper1->Update();


			_VSP_(vtkActor , Actor0) ;
			Actor0->SetMapper(Mapper0);
			Actor0->GetProperty()->SetColor(255 , 0 , 0);
			if(BInverse){
				Actor0->SetScale(1 , -1 , -1) ;
			}

			_VSP_(vtkActor , Actor1) ;
			Actor1->SetMapper(Mapper1) ;
			Actor1->GetProperty()->SetColor(0 , 255 , 0) ;
			if(BInverse){
				Actor1->SetScale(1 , -1 , -1);
			}


			_VSP_(vtkRenderer , Ren) ;
			Ren->AddActor(Actor0) ;
			Ren->AddActor(Actor1) ;



			_VSP_(vtkRenderWindow , RenWin) ;
			RenWin->AddRenderer(Ren);
			RenWin->SetSize(800 , 600);
			RenWin->Render() ;
			if(!_sname_x3Points.empty()) {
				RenWin->SetWindowName(_sname_x3Points.c_str()) ;
			}
			else {
				RenWin->SetWindowName("Red is Points0 , Green is Points1") ;
			}


			_VSP_(vtkInteractorStyleTrackballCamera , Trackball) ;
			_VSP_(vtkRenderWindowInteractor , IRen) ;
			IRen->SetInteractorStyle(Trackball);
			IRen->SetRenderWindow(RenWin);
			IRen->Initialize() ;
			IRen->Start();



			return true ;
	}



	bool
		SceneHelper::
		seePoints(
		const std::vector<std::vector<float>>& _vec_x3Points ,
		const bool BInverse , 
		const std::string& _sname_x3Points) {

			const int NPoints = _vec_x3Points.size() ;


			_VSP_(vtkPoints , Points_) ;
			Points_->SetNumberOfPoints(NPoints) ;


			if(BInverse) {
#pragma omp parallel for
				for(int np=0 ; np<NPoints ; np++) {
					Points_->InsertPoint(np , _vec_x3Points[np][0] , -_vec_x3Points[np][1] , -_vec_x3Points[np][2]);
				}
			} else {
#pragma omp parallel for
				for(int np=0 ; np<NPoints ; np++) {
					Points_->InsertPoint(np , &_vec_x3Points[np][0]);
				}
			}


			//memcpy(Points_->GetVoidPointer(0) , _x3Points.data() , sizeof(T)*3*NPoints) ;


			_VSP_(vtkPolyData , Polys_) ;
			Polys_->SetPoints(Points_) ;


			_VSP_(vtkVertexGlyphFilter , Vertex_) ;
			Vertex_->SetInputData(Polys_) ;
			Vertex_->Update() ;


			_VSP_(vtkPolyDataMapper , Mapper_) ;
			Mapper_->SetInputConnection(Vertex_->GetOutputPort());
			Mapper_->Update() ;


			if(_sname_x3Points.empty()) {
				_DISP_POLYMAPPER_(Mapper_ , !(BInverse) , std::string("See Points")) ;
			} else{
				_DISP_POLYMAPPER_(Mapper_ , !(BInverse) , _sname_x3Points) ;
			}


			return true ;
	}



	bool 
		SceneHelper::
		seePoints(
		std::vector<float>& _x3Points ,
		const bool BInverse ,
		const std::string& _sname_x3Points) {


			const int NPoints = _x3Points.size()/3 ;

			_VSP_(vtkFloatArray , FPointsCoordinates) ;
			FPointsCoordinates->SetNumberOfComponents(3);
			FPointsCoordinates->SetArray(_x3Points.data() , _x3Points.size() , 1);

			_VSP_(vtkPoints , Points_) ;
			Points_->SetData(FPointsCoordinates);


			_VSP_(vtkPolyData , Polys_) ;
			Polys_->SetPoints(Points_) ;


			_VSP_(vtkVertexGlyphFilter , Vertex_) ;
			Vertex_->SetInputData(Polys_) ;
			Vertex_->Update() ;


			_VSP_(vtkPolyDataMapper , Mapper_) ;
			Mapper_->SetInputConnection(Vertex_->GetOutputPort());
			Mapper_->Update() ;


			if(_sname_x3Points.empty()) {
				_DISP_POLYMAPPER_(Mapper_ , BInverse , std::string("See Points")) ;
			} else{
				_DISP_POLYMAPPER_(Mapper_ , BInverse , _sname_x3Points) ;
			}


			return false ;
	}


	bool
		SceneHelper::
		seeCameras(
		const std::vector<PhotoCamera>& _vec_PCams , 
		const std::vector<std::vector<float>>& _vec_Points , 
		const double DScaledCam) {

			_VSP_(vtkRenderer , Ren_);

			_VSP_(vtkPoints , Points_);
			Points_->SetNumberOfPoints(_vec_Points.size());
			Points_->SetDataTypeToFloat();
			for(int np=0 ; np<_vec_Points.size() ; np++) {
				Points_->InsertPoint(np , _vec_Points[np][0] , -_vec_Points[np][1] , -_vec_Points[np][2]);
			}

			_VSP_(vtkPolyData , Polys_);
			Polys_->SetPoints(Points_);


			_VSP_(vtkVertexGlyphFilter ,Vertex_);
			Vertex_->SetInputData(Polys_);
			Vertex_->Update();


			_VSP_(vtkPolyDataMapper , Mapper_);
			Mapper_->SetInputConnection(Vertex_->GetOutputPort()) ;
			Mapper_->Update() ;


			_VSP_(vtkActor , Actor_) ;
			Actor_->SetMapper(Mapper_);

			Ren_->AddActor(Actor_);

			double cent[3] ;
			Polys_->GetCenter(cent);

			std::for_each(_vec_PCams.begin() , _vec_PCams.end() , 
				[&_vec_PCams , &Ren_ , &cent , &DScaledCam](const PhotoCamera& _camera){

					const std::vector<float>& Cen = _camera._Cvec_ ;

					_VSP_(vtkConeSource , Cam_);
					Cam_->SetCenter(DScaledCam*Cen[0] ,DScaledCam*Cen[1] , DScaledCam*Cen[2]);
					Cam_->SetDirection(DScaledCam*(Cen[0]-cent[0]) , DScaledCam*(Cen[1]-cent[1]) , DScaledCam*(Cen[2]-cent[2]));
					Cam_->Update() ;

					_VSP_(vtkPolyDataMapper , Mapper_) ;
					Mapper_->SetInputConnection(Cam_->GetOutputPort());
					Mapper_->Update() ;

					_VSP_(vtkActor , Actor_);
					Actor_->SetMapper(Mapper_);
					Actor_->SetScale(0.3);
					Actor_->GetProperty()->SetColor(1.0  ,0.6 , 0.0);
					Actor_->GetProperty()->SetRepresentationToWireframe() ;

					Ren_->AddActor(Actor_);
			});




			_VSP_(vtkRenderWindow , RenWin_);
			RenWin_->AddRenderer(Ren_);
			RenWin_->SetSize(800 , 600);
			RenWin_->Render() ;


			_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;
			_VSP_(vtkRenderWindowInteractor , IRen_) ;
			IRen_->SetInteractorStyle(Trackball_);
			IRen_->SetRenderWindow(RenWin_);
			IRen_->Initialize() ;
			IRen_->Start();


			return true ;
	}


	static bool
		transformPCamsI2J(
		std::map<int , PhotoCamera>& map_newPCams,
		const int Indi ,
		const int Indj ,
		std::map<int , PhotoCamera>& map_oldPCams ) {

			PhotoCamera &iPCam = map_oldPCams.find(Indi)->second , 
				&jPCam = map_oldPCams.find(Indj)->second;


			const cv::Mat Ri = cv::Mat(3 , 3 , CV_32FC1 , iPCam._R_.data()) ,
				Rj = cv::Mat(3 , 3, CV_32FC1 , jPCam._R_.data()) ,
				Ti = cv::Mat(3 , 1 , CV_32FC1 , iPCam._Tvec_.data()) , 
				Tj = cv::Mat(3 , 1 , CV_32FC1 , jPCam._Tvec_.data()) ;


			/// i is defined as reference camera and j is the camera need to be working on :
			///
			/// Rij = Rj * RiT 
			///	Tij = Tj - Rij * Ti
			///
			const cv::Mat Rij = Rj*(Ri.t()) ;
			const cv::Mat Tij = Tj - Rij*Ti ;

			for(std::map<int , PhotoCamera>::iterator it_old=map_oldPCams.begin() ; 
				it_old!=map_oldPCams.end() ; ++it_old) {

					cv::Mat Rold(3 , 3, CV_32FC1 , it_old->second._R_.data()), 
						Told(3, 1 , CV_32FC1, it_old->second._Tvec_.data()) ;

					/// Rj' = Rij * Rj
					/// Tj' = Rij * Tj + Tij
					///
					cv::Mat &&Rnew = Rij * Rold ;
					cv::Mat	&&Tnew = Rij*Told + Tij ;

					std::vector<float> R_(Rnew.ptr<float>(0) ,Rnew.ptr<float>(0)+9) ,
						Tvec_(Tnew.ptr<float>(0) , Tnew.ptr<float>(0)+3);

					map_newPCams.insert(std::make_pair(it_old->first , PhotoCamera(it_old->second._K_ , R_ , Tvec_))) ;
			}



			return true ;
	}



	bool
		SceneHelper::
		seeStructureAndMotion(
		std::vector<std::vector<float>>& vec_x3Points, 
		std::map<int , PhotoCamera>& map_PCams , 
		const bool BSaveOffline ,
		const std::string& sdir_Save ,
		const std::string& sname_Save ) {


			const int NPoints3 = vec_x3Points.size() ,
				NCams = map_PCams.size() ;

			std::vector<float> x3Points(NPoints3*3) ;
			{
				std::map<int , PhotoCamera>::const_iterator it_PCams = map_PCams.begin() ;
				std::advance(it_PCams , NCams- int(_PER_BUNDLE_NFRAMES_/2)-1 );

				const std::vector<float>& P_ = it_PCams->second._P_ ;
				const std::vector<float>& K_ = it_PCams->second._K_ ;
				const std::vector<float>& Cvec_ = it_PCams->second._Cvec_ ;

#pragma omp parallel for schedule(dynamic , 1)
				for(int np3=0 ; np3<NPoints3 ; np3++) {

					std::vector<float>& x3_ = vec_x3Points[np3] ;

					const float wx  = P_[8]*x3_[0]+P_[9]*x3_[1]+P_[10]*x3_[2]+P_[11] ;
					const float ux = ((P_[0]*x3_[0]+P_[1]*x3_[1]+P_[2]*x3_[2]+P_[3])/wx - K_[2])/K_[0] ;
					const float vx = ((P_[4]*x3_[0]+P_[5]*x3_[1]+P_[6]*x3_[2]+P_[7])/wx - K_[5])/K_[4] ;

					const float dval = std::sqrt( (x3_[0]-Cvec_[0])*(x3_[0]-Cvec_[0]) +
						(x3_[1]-Cvec_[1])*(x3_[1]-Cvec_[1]) + 
						(x3_[2]-Cvec_[2])*(x3_[2]-Cvec_[2]) ) ;



					const float rn = std::sqrt(ux*ux+vx*vx+1.0);

					x3Points[np3*3] = dval*ux/rn ;
					x3Points[np3*3+1] = dval*vx/rn ;
					x3Points[np3*3+2] = dval/rn ;
				}
			}


			const int NPoints = x3Points.size()/3 ;


			_VSP_(vtkPolyData , Polys) ;


			_VSP_(vtkFloatArray , FPointsCoordinates) ;
			FPointsCoordinates->SetNumberOfComponents(3);
			FPointsCoordinates->SetArray(x3Points.data() , x3Points.size() , 1);

			_VSP_(vtkPoints , Points) ;
			Points->SetData(FPointsCoordinates);
			Polys->SetPoints(Points) ;

			_VSP_(vtkVertexGlyphFilter , Vert) ;
			Vert->SetInputData(Polys) ;
			Vert->Update() ;

			_VSP_(vtkPolyDataMapper , Mapper) ;
			Mapper->SetInputConnection(Vert->GetOutputPort());
			Mapper->Update() ;


			_VSP_(vtkActor , PActor) ;
			PActor->SetMapper(Mapper);
			PActor->SetOrigin(0 , 0 ,0);
			PActor->RotateX(180);


			//double pcen[3];
			//PActor->RotateY(6);

			//PActor->SetScale(1.5);

			//const double *pcen = PActor->GetCenter() ;
			//const double *pbox = PActor->GetBounds() ;

			double pcen[3];
			Polys->GetCenter(pcen);
			double pbox[6] ;
			Polys->GetBounds(pbox);

			_VSP_(vtkRenderer , Ren_);
			Ren_->AddActor(PActor);



			////vtkCamera *pCamera = Ren_->GetActiveCamera() ;
			////pCamera->SetParallelProjection(1) ;
			////pCamera->SetPosition(0 , 0 , 1);
			////pCamera->SetFocalPoint(0 , 0 , 0) ;
			////pCamera->SetParallelProjection((pbox[3]-pbox[2])/2) ;
			////pCamera->SetClippingRange(0.001 , 100);


			vtkCamera* RenCam = Ren_->GetActiveCamera();
			RenCam->SetParallelProjection(1);
			RenCam->SetPosition( 0 , std::fabs(pbox[3]-pbox[2])*(1.2) , (pbox[5]-pbox[4])*0.3 );
			RenCam->SetFocalPoint( 0 , 0 , - std::fabs(pbox[5]-pbox[4])*0.75 );
			RenCam->SetParallelScale( (pbox[3]-pbox[2])/0.9 );
			//RenCam->SetClippingRange(0.001 , 100);


			const int Indi = NCams-1-_PER_BUNDLE_NFRAMES_/2 ;
			const int Indj = 0 ;
			//std::map<int ,PhotoCamera> map_NewPCams ; 
			//transformPCamsI2J(map_NewPCams , Indi , Indj , map_PCams);

#if 1 
			const std::map<int ,PhotoCamera>::iterator it_iPCam = map_PCams.find( Indi );
			PhotoCamera& iPCam = it_iPCam->second ;
			const cv::Mat Ri = cv::Mat(3 , 3 , CV_32FC1 , iPCam._R_.data()) ,
				Ti = cv::Mat(3 , 1 , CV_32FC1 , iPCam._Tvec_.data()) ;

			cv::Mat Rj = cv::Mat(3,3,CV_32FC1) ;
			cv::setIdentity(Rj);
			cv::Mat Cj = cv::Mat::zeros(3,1, CV_32FC1);
			Cj.at<float>(2) = (pbox[5]-pbox[4])*0.05  ;
			cv::Mat Tj = -Rj*Cj;


			/// i is defined as reference camera and j is the camera need to be working on :
			///
			/// Rij = Rj * RiT 
			///	Tij = Tj - Rij * Ti
			///
			const cv::Mat Rij = Rj*(Ri.t()) ;
			const cv::Mat Tij = Tj - Rij*Ti ;


			for(int nc=0 ; nc<NCams ; nc++) {

				std::map<int ,PhotoCamera>::iterator it_PCams = map_PCams.begin();
				std::advance(it_PCams, nc) ;

				cv::Mat Rold = cv::Mat(3 ,3 , CV_32FC1 , it_PCams->second._R_.data()) ;
				cv::Mat Told = cv::Mat(3 ,1 , CV_32FC1 , it_PCams->second._Tvec_.data());
				cv::Mat Cold = cv::Mat(3 ,1 , CV_32FC1 , it_PCams->second._Cvec_.data());

				///
				cv::Mat Rnew = Rij * Rold ;
				cv::Mat	Tnew = Rij*Told + Tij ;

				//C = -Rt * T
				cv::Mat Cnew = -Rnew.t()*Tnew ;

				double camcen[3] = {Cnew.at<float>(0) , Cnew.at<float>(1)-0.5 , Cnew.at<float>(2)};
				double cdir[3] = {(camcen[0]-pcen[0]) , (camcen[1]-pcen[0]) , (camcen[2]-pcen[2])};

				_VSP_(vtkConeSource , cam);
				cam->SetCenter(camcen[0] , -camcen[1]  ,camcen[2]);
				cam->SetDirection(cdir);
				cam->Update();

				_VSP_(vtkPolyDataMapper , CMapper_) ;
				CMapper_->SetInputConnection(cam->GetOutputPort());
				CMapper_->Update() ;

				_VSP_(vtkActor , CActor_);
				CActor_->SetMapper(CMapper_);
				CActor_->SetOrigin(0 , 0 , 0);
				CActor_->RotateX(180);

				CActor_->SetScale(0.5);
				if(nc==Indi) {
					CActor_->GetProperty()->SetColor(1.0  ,0.6 , 0.0);
				}
				else {
					CActor_->GetProperty()->SetColor(1.0  ,1.0 , 1.0);
				}

				Ren_->AddActor(CActor_);
			}
#endif


			_VSP_(vtkRenderWindow , RenWin_);
			RenWin_->AddRenderer(Ren_);
			RenWin_->SetSize(800 , 600);
			RenWin_->SetPosition(800 , 0);
			RenWin_->PointSmoothingOn();
			if(BSaveOffline){
				RenWin_->SetOffScreenRendering(1) ;
			}
			RenWin_->Render() ;

			if(BSaveOffline) {

				_VSP_(vtkWindowToImageFilter , WinImage);
				WinImage->SetInput(RenWin_);
				WinImage->SetInputBufferTypeToRGBA();
				WinImage->Update() ;

				_VSP_(vtkBMPWriter , Bmp);
				Bmp->SetInputConnection(WinImage->GetOutputPort());
				char chname[256];
				if(Indi<10) {
					sprintf(chname , "%s/00%d/%s\0" , sdir_Save.c_str() , Indi , sname_Save.c_str());
				}
				else if(Indi<100&&Indi>=10){
					sprintf(chname , "%s/0%d/%s\0" , sdir_Save.c_str() , Indi , sname_Save.c_str());
				}
				else if(Indi>=100){
					sprintf(chname , "%s/%d/%s\0" , sdir_Save.c_str() , Indi , sname_Save.c_str());
				}

				Bmp->SetFileName(std::string(chname).c_str()) ;		
				Bmp->Update();

			}
			else {

				RenWin_->SetWindowName("See Structure and Motion");

				_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;
				_VSP_(vtkRenderWindowInteractor , IRen_) ;
				IRen_->SetInteractorStyle(Trackball_);
				IRen_->SetRenderWindow(RenWin_);
				IRen_->Initialize() ;
				IRen_->Start();
			}

			return true ;
	}


	bool
		SceneHelper::
		deleteSpecificFile(
		const std::string& sdir_Save ,
		const std::string& sname , 
		const int KRefPCam){

			char chname[256] ;
			if(KRefPCam<10){
				sprintf(chname , "%s/00%d\0" , sdir_Save.c_str() , KRefPCam , sname.c_str()) ;
			}
			else if(KRefPCam>=10&&KRefPCam<100){
				sprintf(chname , "%s/0%d\0" , sdir_Save.c_str() , KRefPCam , sname.c_str()) ;
			}
			else if(KRefPCam<1000&&KRefPCam>=100) {
				sprintf(chname , "%s/%d\0" , sdir_Save.c_str() , KRefPCam , sname.c_str()) ;
			}
			std::string sname_file(chname);

			if(!stlplus::folder_exists(sname_file)) {
				stlplus::folder_create(sname_file) ;
			}

			sname_file=sname_file + "/" + sname ;

			if(stlplus::file_exists(sname_file)) {
				stlplus::file_delete(sname_file) ;
			}


			return true ;
	}


	bool
		SceneHelper::
		statisticPhotoConsistency(
		const cv::Mat& DepthMap ,
		const cv::Mat& DepthMask ,
		const std::vector<float>& DepthBox ,
		const std::map<int , PhotoCamera>& map_PCams , 
		const std::string& sdir_Save , 
		const std::string& sname ,
		const std::string& sdir_Image , 
		const int KRefPCam) {

			const cv::Size MapSize = DepthMap.size() ;

			const int NWidth = MapSize.width , NHeight = MapSize.height ;

			const std::vector<std::string> & vec_ImageNames = stlplus::folder_files(sdir_Image);
			cv::Mat limg = cv::imread(sdir_Image+"/"+vec_ImageNames[KRefPCam-1]);
			cv::Mat rimg = cv::imread(sdir_Image+"/"+vec_ImageNames[KRefPCam+1]);
			cv::Mat cimg = cv::imread(sdir_Image+"/"+vec_ImageNames[KRefPCam]);

			cv::cvtColor(limg , limg , cv::COLOR_BGR2GRAY) ;
			cv::cvtColor(rimg , rimg , cv::COLOR_BGR2GRAY) ;
			cv::cvtColor(cimg , cimg , cv::COLOR_BGR2GRAY) ;

			cv::normalize(limg , limg , 0 , 255 , cv::NORM_MINMAX);
			cv::normalize(rimg , rimg , 0 , 255 , cv::NORM_MINMAX);
			cv::normalize(cimg , cimg , 0 , 255 , cv::NORM_MINMAX);



			cv::Mat RDepthMap(MapSize , CV_32FC1) ;
			cv::normalize(DepthMap , RDepthMap, DepthBox[0] , DepthBox[1] , cv::NORM_MINMAX , CV_32FC1 , DepthMask);
			RDepthMap.setTo(0 , 255-DepthMask);

			std::vector<float> x3Points;
			x3Points.reserve(NWidth*NHeight*3) ;

			std::vector<unsigned char> x3Tex ;
			x3Tex.reserve(NWidth*NHeight);



			///step.1 : re-triangulation
			///
			std::map<int ,PhotoCamera>::const_iterator it_RefPCam = map_PCams.find(KRefPCam);

			const std::vector<float>& RefP = it_RefPCam->second._P_ ;
			const std::vector<float>& RefK = it_RefPCam->second._K_ ;
			const std::vector<float>& RefR = it_RefPCam->second._R_ ;
			const std::vector<float>& RefC = it_RefPCam->second._Cvec_;



#pragma omp parallel for
			for(int nh=0 ; nh<NHeight ; nh++) {

				const float *pdval = DepthMap.ptr<float>(nh);
				const unsigned char *pdmask = DepthMask.ptr<unsigned char>(nh);
				const unsigned char *ptex = cimg.ptr<unsigned char>(nh);

				for(int nw=0 ; nw<NWidth ; nw++ , pdmask++ ,pdval++ , ptex++) {
					if(pdmask[0]){
						const float ux = (nw-RefK[2])/RefK[0] , vx = (nh-RefK[5])/RefK[4] ;

						float rx = RefR[0]*ux + RefR[3]*vx + RefR[6] ;
						float ry = RefR[1]*ux + RefR[4]*vx + RefR[7] ;
						float rz = RefR[2]*ux + RefR[5]*vx + RefR[8] ;

						const float rn = std::sqrt(rx*rx+ry*ry+rz*rz) ;
						rx/=rn , ry/=rn , rz/=rn ;

						const float v3x = *pdval*rx + RefC[0] ;
						const float v3y = *pdval*ry + RefC[1] ;
						const float v3z = *pdval*rz + RefC[2] ;

#pragma omp critical
						{
							x3Points.push_back(v3x)  , x3Points.push_back(v3y) , x3Points.push_back(v3z) ;
							x3Tex.push_back(ptex[0]) ;
						}
					}
				}
			}

			std::vector<float> (x3Points).swap(x3Points);
			std::vector<unsigned char> (x3Tex).swap(x3Tex);



			///step.2: re-projective texture
			///
			const int NPoints = x3Points.size()/3 ;


			std::vector<std::vector<unsigned char>> vec_x3tex(2 , std::vector<unsigned char>(NPoints , -255));

			for(int ncam=KRefPCam-1 , nCnt=0 ; ncam<=KRefPCam+1 ; ncam++) {
				if(ncam==KRefPCam){
					continue; 
				}

				std::map<int ,PhotoCamera>::const_iterator it_ComPCam = map_PCams.find(ncam) ;
				const std::vector<float>& ComP = it_ComPCam->second._P_ ;

#pragma omp parallel for 
				for(int np=0 ;np<NPoints ; np++) {
					const float *const px3p = &x3Points[np*3] ;
					//const unsigned char *const px3t = &x3Tex[np] ;

					const float wx = ComP[8]*px3p[0]+ComP[9]*px3p[1]+ComP[10]*px3p[2]+ComP[11] ;
					const float ux = (ComP[0]*px3p[0]+ComP[1]*px3p[1]+ComP[2]*px3p[2]+ComP[3])/wx ;
					const float vx = (ComP[4]*px3p[0]+ComP[5]*px3p[1]+ComP[6]*px3p[2]+ComP[7])/wx ;

					if(ux<NWidth-2 && ux>1 && vx<NHeight-2 && vx>2) {

						cv::Point2f pt2(ux , vx);

						cv::Mat Tex ;
						cv::getRectSubPix(limg , cv::Size(1 ,1) , pt2 , Tex);

						vec_x3tex[nCnt][np] = Tex.at<unsigned char>(0) ;
					}				
				}

				nCnt++ ;
			}



			///step.3: statistic consistency
			///
			char chname[256] ;
			if(KRefPCam<10){
				sprintf(chname , "%s/00%d/%s\0" , sdir_Save.c_str() , KRefPCam , sname.c_str()) ;
			}
			else if(KRefPCam>=10&&KRefPCam<100){
				sprintf(chname , "%s/0%d/%s\0" , sdir_Save.c_str() , KRefPCam , sname.c_str()) ;
			}
			else if(KRefPCam<1000&&KRefPCam>=100) {
				sprintf(chname , "%s/%d/%s\0" , sdir_Save.c_str() , KRefPCam , sname.c_str()) ;
			}
			std::string sname_file(chname) ;

			if(!stlplus::file_exists(sname_file)) {
				stlplus::file_created(sname_file) ;
			}

			std::fstream fs;

			fs.open(chname , std::ios::out | std::ios::app);

			double mean_val(0) , stdvar_val(0) ;

#pragma omp parallel for 
			for(int np=0 ; np<NPoints ; np++) {

				const unsigned char *const px3t_r = &x3Tex[np] , 
					*const px3t_c_l = &vec_x3tex[0][np],
					*const px3t_c_r = &vec_x3tex[1][np] ;

				mean_val += ( (double)std::fabs(float(px3t_r[0]-px3t_c_l[0])) + (double)std::fabs(float(px3t_r[0]-px3t_c_r[0])) )/2 ;
			}

			mean_val /= NPoints ;

			fs<<mean_val<<"\t" ;

#pragma omp parallel for 
			for(int np=0 ; np<NPoints ; np++) {

				const unsigned char *const px3t_r = &x3Tex[np] , 
					*const px3t_c_l = &vec_x3tex[0][np],
					*const px3t_c_r = &vec_x3tex[1][np] ;

				double diff1 =  (std::fabs(float(px3t_r[0]-px3t_c_l[0])) - mean_val) ;
				double diff2 =  (std::fabs(float(px3t_r[0]-px3t_c_r[0])) - mean_val) ;

				diff1 *= diff1 ;
				diff2 *= diff2 ;

				stdvar_val = (diff1+diff2)/2 ;
			}

			stdvar_val /= NPoints ;

			stdvar_val = std::sqrt(stdvar_val) ;

			fs<<stdvar_val<<"\n" ;

			fs.close() ;

			fflush(stdout);
			fprintf(stdout , "\t\tmean photo error : %f\tstdvar photo error: %f\n" , mean_val , stdvar_val);
			fflush(stdout);

			return true ;
	}

}


