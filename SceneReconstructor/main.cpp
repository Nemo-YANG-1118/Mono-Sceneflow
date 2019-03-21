
#include <stlplus3/filesystemSimplified/file_system.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/video.hpp>

#include <iostream>
#include <utility>
#include <tuple>
#include <cstdlib>


#include "SceneConfiguration.h"
#include "SceneHelper.h"
#include "PhotoCamera.h"
#include "DepthFieldBuilders/DepthFieldGenerator.h"
#include "DepthFieldBuilders/DepthFieldOptimizer.h"
#include "DepthFieldBuilders/DepthFieldIntegrator.h"
#include "DepthFieldBuilders/DepthFieldHelper.h"
#include "DepthFieldBuilders/DepthFieldCommon/PoissonDistanceFieldReconstructor.h"


#pragma warning(disable:4129)


#undef _SEE_IMAGE_
#define _SEE_IMAGE_(img_ , sWinName_)\
	cv::namedWindow(sWinName_ , 1) ;\
	cv::imshow(sWinName_ , img_);\
	cv::waitKey(0);\
	cv::destroyWindow(sWinName_);


#undef _DEBUG_SEE_DEPTH_MAP_ORIGINAL_
#define _DEBUG_SEE_DEPTH_MAP_ORIGINAL_ 0

#undef _DEBUG_SEE_DEPTH_MAP_CURRENT_
#define _DEBUG_SEE_DEPTH_MAP_CURRENT_ 0

#undef _DEBUG_SEE_DEPTH_MAP_INTEG_
#define _DEBUG_SEE_DEPTH_MAP_INTEG_ 0

#undef _DEBUG_B_ENTIRE_DEPTH_MAP_
#define _DEBUG_B_ENTIRE_DEPTH_MAP_(_EntireMask , _UnEntireMask , BFullMap)\
	( (BFullMap) ? _EntireMask : _UnEntireMask )\

#undef _USE_POISSON_
#define _USE_POISSON_ 0


int 
main(int argc,char *argv[]) {


	using namespace stereoscene ;


	const int KBundleBegin = 16 ;
	const int KBundleEnd = 49 ;


	const bool BLiveScene = false ;

	const bool BMetric_L1 = false ;


	const bool BBoundingVolume = false ;
	const bool BUseMotionMask = true ;
	const bool BUseVisualMask = true ;


	const int NOptIters = BMetric_L1 ? 1 : 3  ;


	const double DFarBoundRatio = 3.0,
		DNearBoundRatio = 30 ;


	const bool 	BDebugOfflineSysthsize = false ,
		BDebugSaveScene = true ,
		BSynthTexture = true ,
		BSeeOnlyStructure = true ;


	const bool 	BDebugSfM = false ,
		BDebugSaveSfM = false ,
		BSeeOnlyPointCloud = true ;


	///for saving
	const bool BSeeOrSaveInitTrimesh = false ,
		BSeeOrSaveCurTrimesh = false,
		BSeeOrSaveIntegTrimesh =true,
		BSeeOrSavePoissonTrimesh = true,


		BDebugSaveReprojError = false,

		BDebugSaveVisMask = false,
		BDebugSaveCleanPoints = false,

		BDebugSaveTextureImage = true,

		BDebugSaveMeshPhoto = true,
		BDebugSaveDepthMap = true,
		BDebugEntireDepthMapSaved = true,

		BDebugSaveVideo = true ;



	const cv::Size ImageSize(800,600);
	//const cv::Size ImageSize(960, 540);
	//const cv::Size ImageSize(640 , 480);


	std::string sdir_ObservedImages = "D:/Datasets/DTU_Datasets/Scene_04_800_600/test",

		sdir_ObservedFolder = "../../StereoScene_Data/Observer",

		sdir_Save = "../../StereoScene_Data/Reconstructor/DTU_04_800_600",


		sext_PhotoMesh = ".bmp", //bmp or ply

		sname_PhotoConsist = "PhotoConsist.txt",

		sname_ReprojError = "Error.txt",


		sname_TexImage = "TexImage.bmp",
		sname_VisMask = "VisMask.bmp",

		sname_PointsCameras = "PointCloudAndCameras.bmp",

		sname_InitDepthMap = "Init_DepthMap.bmp",
		sname_CurDepthMap = "Cur_DepthMap.bmp",
		sname_IntegDepthMap = "Integ_DepthMap.bmp",
		sname_PoissonDepthMap = "Poisson_DepthMap.bmp",

		sname_InitUTexMesh = "Init_UTexMesh"+sext_PhotoMesh,
		sname_InitTexMesh = "Init_TexMesh"+sext_PhotoMesh,

		sname_CurUTexMesh = "Cur_UTexMesh"+sext_PhotoMesh,
		sname_CurTexMesh = "Cur_TexMesh"+sext_PhotoMesh,

		sname_IntegUTexMesh = "Integ_UTexMesh"+sext_PhotoMesh,
		sname_IntegTexMesh = "Integ_TexMesh"+sext_PhotoMesh,

		sname_PoissonUTexMesh = "Integ_UTexMesh"+sext_PhotoMesh,
		sname_PoissonTexMesh = "Integ_TexMesh"+sext_PhotoMesh,


		sname_DemoVideo = "DemoVideo.avi" ;



	const std::string shead  = (BMetric_L1) ? "L1_" : "L2_" ;

	sname_ReprojError = shead+sname_ReprojError ;

	sname_CurDepthMap = shead+sname_CurDepthMap ;
	sname_IntegDepthMap = shead+sname_IntegDepthMap ;
	sname_CurUTexMesh = shead+sname_CurUTexMesh ;
	sname_CurTexMesh = shead+sname_CurTexMesh ;
	sname_IntegUTexMesh = shead+sname_IntegUTexMesh ;
	sname_IntegTexMesh = shead+sname_IntegTexMesh ;
	sname_DemoVideo = shead+sname_DemoVideo ;



	const std::string &sname_InitMesh = BSynthTexture ? sname_InitTexMesh : sname_InitUTexMesh,
		&sname_CurMesh = BSynthTexture ? sname_CurTexMesh : sname_CurUTexMesh,
		&sname_PoissonMesh = BSynthTexture ? sname_PoissonTexMesh : sname_PoissonUTexMesh ;



	const std::vector<std::string>& vec_ObservedSubFolder = stlplus::folder_subdirectories(sdir_ObservedFolder) ;


	const int NBundles = vec_ObservedSubFolder.size()-2,
		NPads = _PER_BUNDLE_NFRAMES_/2 ;


	const int NThreshCameras = 0 + _PER_BUNDLE_NFRAMES_,

		NThreshPoints = NThreshCameras*50 ;


#if 1

	std::map<int,std::tr1::tuple<PhotoCamera,cv::Mat,cv::Mat>> map_DepthFields ;


	std::vector<std::string> vec_TexImageNames;
	if(BSynthTexture){
		const std::vector<std::string>& vec_filesname = stlplus::folder_files(sdir_ObservedImages);
		vec_TexImageNames.reserve(vec_filesname.size());
		for(int nf=0 ; nf<vec_filesname.size() ; nf++){
			const std::string& sext = stlplus::extension_part(vec_filesname[nf]) ;
			if(sext=="jpg" || sext=="JPG" || sext=="bmp" || sext=="BMP" || sext=="png"||sext=="PNG" ||
				sext=="tif" || sext=="TIF" ||sext=="tiff" ||sext=="TIFF"){
				vec_TexImageNames.push_back(sdir_ObservedImages + "/" + vec_filesname[nf]);
			}
		}
		std::vector<std::string>(vec_TexImageNames).swap(vec_TexImageNames);
	}


	///check saving folder
	if(!stlplus::folder_exists(sdir_Save)) {
		stlplus::folder_create(sdir_Save);
	}


	cv::Ptr<cv::BackgroundSubtractorMOG2> bgSub = cv::cuda::createBackgroundSubtractorMOG2();
	bgSub->setDetectShadows(false);
	bgSub->setNMixtures(2);


	cv::Mat LastFrame ;
	cv::cuda::GpuMat g_BgFrame,FgImage,g_FgMask;

	std::vector<std::vector<float>> vec_StructurePoints ;
	std::map<int,PhotoCamera> map_MotionPCams ;
	SceneHelper::loadStructureAndMotion(vec_StructurePoints,map_MotionPCams,sdir_ObservedFolder);


	for(int nb=(KBundleBegin>NPads?KBundleBegin:NPads) ; nb <(NBundles<KBundleEnd?NBundles:KBundleEnd) ; nb++) {


		//////////////////////////////////////////////////////////////////////////
		///--> Load observed data of current bundle
		//////////////////////////////////////////////////////////////////////////

		fflush(stdout) ;
		fprintf(stdout,"\n\n\n////////////////////////////////////////////////////\n") ;
		fprintf(stdout,"///\tScene No.%d\n",nb-NPads) ;
		fprintf(stdout,"///////////////////////////////////////////////////\n\n\n") ;
		fflush(stdout) ;


		std::vector<std::vector<float>> vec_x3Points ;
		std::map<int,PhotoCamera> map_PCams ;
		cv::Mat CurImage,CurFgMask ;


		if(!SceneHelper::loadImageSpecifiedSize(CurImage,sdir_ObservedImages,ImageSize,nb-NPads,false,BDebugSaveTextureImage,sdir_Save)) {
			continue;
		}

		if(CurImage.channels()==1) {
			cv::cvtColor(CurImage,CurImage,cv::COLOR_GRAY2RGB);
		}



		///foreground-background segmentation
		if(BUseMotionMask){
			if(g_BgFrame.empty()){
				g_BgFrame.upload(CurImage);
			}
			else {
				if(g_FgMask.empty()) {
					bgSub->apply(g_BgFrame,g_FgMask,0);
				}
				else {
					bgSub->getBackgroundImage(g_BgFrame);
					cv::cuda::GpuMat g_CurFrame(CurImage);
					bgSub->apply(g_CurFrame,g_FgMask,0);
					g_FgMask.download(CurFgMask);
					g_CurFrame.copyTo(g_BgFrame);

					cv::erode(CurFgMask,CurFgMask,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3))) ;
					cv::dilate(CurFgMask,CurFgMask,cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3)));

					CurFgMask*=255 ;
					SceneHelper::blackMapBorder(CurFgMask);

					CurFgMask.convertTo(CurFgMask,CV_8UC1);
					cv::Mat motMask;
					DepthFieldHelper::findConcaveMaskfromMaskMap(motMask,ImageSize,CurFgMask);
					motMask.copyTo(CurFgMask);
					SceneHelper::blackMapBorder(CurFgMask);


					//_SEE_IMAGE_(CurFgMask , "fore");
				}
			}
		}



		const int KthCurPCam = nb-NPads  ;

		if((BLiveScene)) {
			///for dynamic scene reconstruction
			if(!SceneHelper::loadx3Points(vec_x3Points,sdir_ObservedFolder+"/"+vec_ObservedSubFolder[nb],NThreshPoints) ||
				!SceneHelper::loadPCams(map_PCams,sdir_ObservedFolder+"/"+vec_ObservedSubFolder[nb],NThreshCameras)) {
				continue ;
			}
		}
		else {
			///for global static scene reconstruction
			if(nb>=NThreshCameras) {
				SceneHelper::findStructurefromMotion(vec_x3Points,map_PCams,KthCurPCam,NPads,ImageSize,vec_StructurePoints,map_MotionPCams) ;
			}
			else {
				continue ;
			}
		}

		const int NCams = map_PCams.size(),
			NPoints = vec_x3Points.size() ;


		///check sub-saving folder
		std::string sbasedir_Save ;
		{
			char chbasename[256] ;
			if(KthCurPCam<10) {
				sprintf(chbasename,"%s/00%d\0",sdir_Save.c_str(),KthCurPCam) ;
			}
			else if(KthCurPCam<100){
				sprintf(chbasename,"%s/0%d\0",sdir_Save.c_str(),KthCurPCam);
			}
			else if(KthCurPCam<1000) {
				sprintf(chbasename,"%s/%d\0",sdir_Save.c_str(),KthCurPCam) ;
			}

			sbasedir_Save.assign(chbasename);
		}



		fflush(stdout) ;
		fprintf(stdout," SceneReconstructor: load ( %d ) points and ( %d ) cameras\n",NPoints,NCams) ;
		fflush(stdout) ;




		std::map<int,PhotoCamera>::iterator it_CurPCam = map_PCams.find(KthCurPCam);

		cv::Mat ConcaveMask ;
		DepthFieldHelper::findConcaveMaskfromPointsVector(ConcaveMask,ImageSize,it_CurPCam->second._P_,vec_x3Points);



		///////////////////////////////////////////////////////////////////////////////
		///--> Section.1: Reconstruct the original depth map of current view
		///////////////////////////////////////////////////////////////////////////////


		cv::Mat CurDepthMap,CurDepthMask ;
		std::vector<float> CurDepthBox   ;


		const bool BCleaning = true,BGenerateFullMap = true ;


		DepthFieldGenerator(CurDepthMap,CurDepthMask,CurDepthBox,
			vec_x3Points,map_PCams) (
			BCleaning,KthCurPCam,ImageSize,BGenerateFullMap,KthCurPCam,BDebugSaveCleanPoints,sdir_Save) ;


		//_SEE_IMAGE_(CurDepthMask , "cur depth mask") ; 


		cv::Mat CurVisMask ;
		CurDepthMask.copyTo(CurVisMask);
		if(BUseVisualMask) {
			SceneHelper::generateVisualMask(CurVisMask,CurImage,KthCurPCam,BDebugSaveVisMask,sdir_Save,sname_VisMask) ;
			if(BUseMotionMask && !CurFgMask.empty()){
				CurVisMask&=CurFgMask ;
			}
		}
		CurVisMask &= ConcaveMask ;



		cv::Mat BoundMask;
		if(BBoundingVolume) {

			DepthFieldHelper::generateDepthBoundingMask(BoundMask,CurDepthMap,CurDepthMask,DFarBoundRatio,DNearBoundRatio);

			if(BUseVisualMask){
				BoundMask &= CurVisMask ;
			}
			else {
				BoundMask &= ConcaveMask ;
			}
		}
		else {
			BoundMask = BUseMotionMask ? CurVisMask: ConcaveMask ;
		}



		if(BDebugSaveDepthMap) {
			SceneHelper::saveDepthMap(KthCurPCam,false,BDebugEntireDepthMapSaved,sdir_Save,sname_InitDepthMap,CurDepthMap,
				_DEBUG_B_ENTIRE_DEPTH_MAP_(CurDepthMask,BoundMask,BDebugEntireDepthMapSaved));
		}


		if(BDebugSfM) {
			if(BSeeOnlyPointCloud) {
				SceneHelper::seeStructureAndMotion(vec_x3Points,map_PCams,BDebugSaveSfM,sdir_Save,sname_PointsCameras);
				continue ;
			}
			else {
				SceneHelper::seeStructureAndMotion(vec_x3Points,map_PCams,BDebugSaveSfM,sdir_Save,sname_PointsCameras) ;
			}
		}


#if _DEBUG_SEE_DEPTH_MAP_ORIGINAL_
		cv::Mat TmpDepthMap0 ;
		cv::normalize(CurDepthMap,TmpDepthMap0,0,1,cv::NORM_MINMAX,CV_32FC1,CurDepthMask) ;
		TmpDepthMap0.setTo(0,255-CurDepthMask);
		_SEE_IMAGE_(TmpDepthMap0,"Original Depth Map");

#endif

		if(BSeeOrSaveInitTrimesh &&(KthCurPCam>KBundleBegin)) {

			//cv::normalize(CurDepthMap , CurDepthMap , CurDepthBox[0] , CurDepthBox[1] , cv::NORM_MINMAX , CV_32FC1 , CurDepthMask);

			//DepthFieldGenerator::orthoBackprojection(CurDepthMap , BoundMask , CurImage ,  it_CurPCam->second , ImageSize , KthCurPCam, \
																//	BSeeOrSaveInitTrimesh , BSynthTexture , BDebugSaveMeshPhoto , sdir_Save , sname_InitMesh ) ;
		}



		/////////////////////////////////////////////////////////////////////////////
		///--> Section.2: Optimize the depth map on current bundle
		////////////////////////////////////////////////////////////////////////////


		cv::Mat CurErrorMap ;

		std::map<int,cv::Mat> map_uCurFlow,map_vCurFlow ; //optical flow in current view bundle
		cv::Mat CurRefImage ;


		DepthFieldOptimizer(CurDepthMap,CurDepthMask,CurErrorMap,CurDepthBox,
			map_PCams,map_uCurFlow,map_vCurFlow,CurRefImage) (
			sdir_ObservedImages,KthCurPCam,BMetric_L1,BDebugSaveReprojError,KthCurPCam,NOptIters) ;


		CurDepthMap.setTo(0,(255-CurDepthMask)) ;



		if(BBoundingVolume) {

			DepthFieldHelper::generateDepthBoundingMask(BoundMask,CurDepthMap,CurDepthMask,DFarBoundRatio,DNearBoundRatio);

			if(BUseVisualMask){
				BoundMask &= CurVisMask ;
			}
			else {
				BoundMask &= ConcaveMask ;
			}
		}
		else {
			BoundMask = BUseMotionMask ? CurVisMask : ConcaveMask ;
		}


		///for paper
		if(BDebugSaveReprojError) {

			SceneHelper::saveReprojectiveErrorMap(\
				KthCurPCam,CurErrorMap,sdir_Save,sname_ReprojError) ;

			//continue;
		}



		if(BDebugSaveDepthMap) {
			SceneHelper::saveDepthMap(KthCurPCam,false,BDebugEntireDepthMapSaved,sdir_Save,sname_CurDepthMap,CurDepthMap,
				_DEBUG_B_ENTIRE_DEPTH_MAP_(CurDepthMask,BoundMask,BDebugEntireDepthMapSaved)) ;

		}



#if _DEBUG_SEE_DEPTH_MAP_CURRENT_ 
		cv::Mat TmpDepthMap1 ;
		cv::normalize(CurDepthMap,TmpDepthMap1,0,1,cv::NORM_MINMAX,CV_32FC1,CurDepthMask) ;
		TmpDepthMap1.setTo(0,255-CurDepthMask);
		_SEE_IMAGE_(TmpDepthMap1,"Current Depth Map");

#endif



		if((BSeeOrSaveCurTrimesh) && (KthCurPCam>KBundleBegin)){

			//DepthFieldGenerator::orthoBackprojection(CurDepthMap , BoundMask , CurImage ,  it_CurPCam->second , ImageSize , KthCurPCam ,
			//	BSeeOrSaveCurTrimesh , BSynthTexture , BDebugSaveMeshPhoto , sdir_Save , sname_CurMesh ) ;


			std::vector<float> x3Points ;


			///first time need to be delete old file
			SceneHelper::deleteSpecificFile(sdir_Save,sname_PoissonDepthMap,KthCurPCam);

			SceneHelper::statisticPhotoConsistency(CurDepthMap,ConcaveMask,CurDepthBox,map_PCams,\
				sdir_Save,sname_PhotoConsist,sdir_ObservedImages,KthCurPCam) ;



			std::vector<unsigned char> x3Tex ;

			DepthFieldHelper::triangulateDepthMap(x3Points,it_CurPCam->second,CurDepthMap,BoundMask,CurDepthBox,BSynthTexture,CurImage,x3Tex);;


			cv::Mat UDepthMap;
			cv::normalize(CurDepthMap,UDepthMap,0,255.0,cv::NORM_MINMAX,CV_32FC1,CurDepthMask);
			UDepthMap.setTo(0,255-CurDepthMask);
			UDepthMap.convertTo(UDepthMap,CV_8UC1);
			cv::flip(UDepthMap,UDepthMap,0);

			//_SEE_IMAGE_(UDepthMap, "udepth");


			DepthFieldHelper::synthesizeScene(\
				BDebugOfflineSysthsize,BSeeOnlyStructure,BDebugSaveScene,\
				sdir_Save,sname_CurMesh,\
				x3Points,map_PCams,KthCurPCam,NPads,UDepthMap,BSynthTexture,vec_TexImageNames,x3Tex);
		}



#if _USE_POISSON_ 


		///for paper by comparing with poisson
		///
		cv::Mat PoissonDepth(ImageSize,CV_32FC1) ;

		//std::string sname_SavePoisson = sbasedir_Save+"/"+sname_Poisson ;

		PoissonDistanceFieldReconstructor()(PoissonDepth,CurDepthMap,CurDepthMask) ;

		//fprintf(stdout , "\t\tpoisson done...\n") ;


		PoissonDepth.setTo(1.0,255-CurDepthMask);


		if(BDebugSaveDepthMap) {
			SceneHelper::saveDepthMap(KthCurPCam,false,BDebugEntireDepthMapSaved,sdir_Save,sname_PoissonDepthMap,PoissonDepth,
				_DEBUG_B_ENTIRE_DEPTH_MAP_(CurDepthMask,BoundMask,BDebugEntireDepthMapSaved));
		}



		if( BSeeOrSavePoissonTrimesh && KthCurPCam>KBundleBegin ) {

			std::vector<float> x3Points ;


			SceneHelper::statisticPhotoConsistency(PoissonDepth,ConcaveMask,CurDepthBox,map_PCams,\
				sdir_Save,sname_PhotoConsist,sdir_ObservedImages,KthCurPCam) ;



			std::vector<unsigned char> x3Tex ;

			DepthFieldHelper::triangulateDepthMap(x3Points,it_CurPCam->second,PoissonDepth,BoundMask,CurDepthBox,BSynthTexture,CurImage,x3Tex);;


			cv::Mat UDepthMap;
			cv::normalize(PoissonDepth,UDepthMap,0,255.0,cv::NORM_MINMAX,CV_32FC1,CurDepthMask);
			UDepthMap.setTo(0,255-CurDepthMask);
			UDepthMap.convertTo(UDepthMap,CV_8UC1);
			cv::flip(UDepthMap,UDepthMap,0);

			//_SEE_IMAGE_(UDepthMap, "udepth");

			DepthFieldHelper::synthesizeScene(\
				BDebugOfflineSysthsize,BSeeOnlyStructure,BDebugSaveScene,\
				sdir_Save,sname_PoissonMesh,\
				x3Points,map_PCams,KthCurPCam,NPads,UDepthMap,BSynthTexture,vec_TexImageNames,x3Tex);

		}

		//continue ;
#endif





		///////////////////////////////////////////////////////////////////////////
		///--> Section.3: Integrate former and current depth map together
		///////////////////////////////////////////////////////////////////////////


		if(!map_DepthFields.empty()) {


			///default former index means using last keyframe as compared with current keyframe
			DepthFieldIntegrator(map_PCams,CurDepthMap,CurDepthMask,CurDepthBox,
				map_uCurFlow,map_vCurFlow) (
				map_DepthFields,CurImage,KthCurPCam);



			if(BBoundingVolume) {
				DepthFieldHelper::generateDepthBoundingMask(BoundMask,CurDepthMap,CurDepthMask,DFarBoundRatio,DNearBoundRatio);


				if(BUseVisualMask){
					BoundMask &= CurVisMask ;
				}
				else {
					BoundMask &= ConcaveMask ;
				}
			}
			else {
				BoundMask = BUseMotionMask ? CurVisMask : ConcaveMask ;
			}



			if(BDebugSaveDepthMap) {
				SceneHelper::saveDepthMap(KthCurPCam,false,BDebugEntireDepthMapSaved,sdir_Save,sname_IntegDepthMap,CurDepthMap,
					_DEBUG_B_ENTIRE_DEPTH_MAP_(CurDepthMask,BoundMask,BDebugEntireDepthMapSaved));
			}

#if _DEBUG_SEE_DEPTH_MAP_INTEG_ 
			cv::Mat TmpDepthMap2 ;
			cv::normalize(CurDepthMap,TmpDepthMap2,0,1,cv::NORM_MINMAX,CV_32FC1,CurDepthMask) ;
			TmpDepthMap2.setTo(0,255-CurDepthMask);
			_SEE_IMAGE_(TmpDepthMap2,"Integrated Depth Map");

#endif

			if((BSeeOrSaveIntegTrimesh) && (KthCurPCam>KBundleBegin)){

				//DepthFieldGenerator::orthoBackprojection(CurDepthMap , BoundMask , CurImage ,  it_CurPCam->second , ImageSize , KthCurPCam ,
				//	BSeeOrSaveIntegTrimesh , BSynthTexture , BDebugSaveMeshPhoto , sdir_Save , (BSynthTexture)?sname_IntegTexMesh:sname_IntegUTexMesh ) ;

				//DepthFieldGenerator::orthoBackprojection(CurDepthMap , BoundMask , CurImage ,  it_CurPCam->second , ImageSize , KthCurPCam ,
				//	BSeeOrSaveIntegTrimesh , !BSynthTexture,  BDebugSaveMeshPhoto , sdir_Save , (BSynthTexture)?sname_IntegUTexMesh:sname_IntegTexMesh) ;


				std::vector<float> x3Points ;


				SceneHelper::statisticPhotoConsistency(CurDepthMap,ConcaveMask,CurDepthBox,map_PCams,\
					sdir_Save,sname_PhotoConsist,sdir_ObservedImages,KthCurPCam) ;



				std::vector<unsigned char> x3Tex ;

				DepthFieldHelper::triangulateDepthMap(x3Points,it_CurPCam->second,CurDepthMap,BoundMask,CurDepthBox,BSynthTexture,CurImage,x3Tex);


				cv::Mat UDepthMap;
				cv::normalize(CurDepthMap,UDepthMap,0,255.0,cv::NORM_MINMAX,CV_32FC1,CurDepthMask);
				UDepthMap.setTo(0,255-CurDepthMask);
				UDepthMap.convertTo(UDepthMap,CV_8UC1);
				cv::flip(UDepthMap,UDepthMap,0);

				//_SEE_IMAGE_(BoundMask , "Final Mask");

				///un-textured
				DepthFieldHelper::synthesizeScene(\
					BDebugOfflineSysthsize,BSeeOnlyStructure,BDebugSaveScene,\
					sdir_Save,sname_IntegUTexMesh,\
					x3Points,map_PCams,KthCurPCam,NPads,UDepthMap,false,vec_TexImageNames,x3Tex);

				///textured
				if(BSynthTexture) {
					DepthFieldHelper::synthesizeScene(\
						BDebugOfflineSysthsize,BSeeOnlyStructure,BDebugSaveScene,\
						sdir_Save,sname_IntegTexMesh,\
						x3Points,map_PCams,KthCurPCam,NPads,UDepthMap,true,vec_TexImageNames,x3Tex);
				}
			}

		}



		///save current data
		if(it_CurPCam!=map_PCams.end()) {
			map_DepthFields.insert(std::make_pair(KthCurPCam,
				std::tr1::make_tuple(it_CurPCam->second,CurDepthMap,CurImage)));
		}

	}


	fflush(stdout) ;
	fprintf(stdout,"\n\n\n////////////////////////////////////////////////////\n") ;
	fprintf(stdout,"///\tWork Done with the Whole Sequence from Observing\n") ;
	fprintf(stdout,"///////////////////////////////////////////////////\n\n\n") ;
	fflush(stdout) ;

#endif


#if 0

	if(BDebugSaveVideo) {
		SceneHelper::saveImagesAndMapsVideo(ImageSize,sname_DemoVideo,sdir_Save,
			sname_TexImage ,sname_PointsCameras,sname_CurDepthMap,sname_IntegUTexMesh,sname_IntegTexMesh);
		//sdir_ObservedImages );
	}

#endif





	return EXIT_SUCCESS ;
}