
#include <stlplus3/filesystemSimplified/file_system.hpp>

#include <VTK/vtkSmartPointer.h>
#include <VTK/vtkFloatArray.h>
#include <VTK/vtkIdTypeArray.h>
#include <VTK/vtkUnsignedCharArray.h>
#include <VTK/vtkPoints.h>
#include <VTK/vtkPointData.h>
#include <VTK/vtkPolyData.h>
#include <VTK/vtkCellArray.h>
#include <VTK/vtkPolyDataMapper.h>
#include <VTK/vtkActor.h>
#include <VTK/vtkRenderer.h>
#include <VTK/vtkRenderWindow.h>
#include <VTK/vtkRenderWindowInteractor.h>
#include <VTK/vtkInteractorStyleTrackballCamera.h>
#include <VTK/vtkCamera.h>
#include <VTK/vtkProperty.h>
#include <VTK/vtkWindowToImageFilter.h>
#include <VTK/vtkBMPWriter.h>
#include <VTK/vtkWindowedSincPolyDataFilter.h>
#include <VTK/vtkCleanPolyData.h>
#include <VTK/vtkPolyDataNormals.h>
#include <VTK/vtkPlaneSource.h>
#include <VTK/vtkProperty.h>
#include <VTK/vtkConeSource.h>
#include <VTK/vtkTextureMapToPlane.h>
#include <VTK/vtkImageImport.h>
#include <VTK/vtkTIFFReader.h>
#include <VTK/vtkBMPReader.h>
#include <VTK/vtkPNGReader.h>
#include <VTK/vtkJPEGReader.h>
#include <VTK/vtkPLYWriter.h>
#include <VTK/vtkTIFFWriter.h>
#include <VTK/vtkLine.h>
#include <VTK/vtkTriangleFilter.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <list>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>


#include "DepthFieldHelper.h"
#include "DepthFieldCommon/TrimeshGenerator.h"

#include "../SceneConfiguration.h"

#ifdef _VSP_
#undef _VSP_
#endif
#define _VSP_(ClassName_ , ObjName_);\
	vtkSmartPointer<ClassName_> ObjName_ = vtkSmartPointer<ClassName_>::New() ;



#ifdef _DISP_POLYMAPPER_
#undef _DISP_POLYMAPPER_
#endif
#define _DISP_POLYMAPPER_(Mapper_ , BInverse ,sname_)\
	_VSP_(vtkActor , Actor_) ;\
	Actor_->SetMapper(Mapper_) ;\
	if(BInverse) {Actor_->SetScale(1.0 , -1.0 , -1.0); }\
	_VSP_(vtkRenderer , Ren_) ;\
	Ren_->AddActor(Actor_) ;\
	_VSP_(vtkRenderWindow , RenWin_) ;\
	RenWin_->AddRenderer(Ren_);\
	RenWin_->SetSize(800 , 600);\
	RenWin_->Render() ;\
	RenWin_->SetWindowName(sname_.c_str()) ;\
	_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;\
	_VSP_(vtkRenderWindowInteractor , IRen_) ;\
	IRen_->SetInteractorStyle(Trackball_);\
	IRen_->SetRenderWindow(RenWin_);\
	IRen_->Initialize() ;\
	IRen_->Start();\



#ifdef _DISP_ACTOR_
#undef _DISP_ACTOR_
#endif
#define _DISP_ACTOR_(Actor_ , sname_)\
	_VSP_(vtkRenderer , Ren_) ;\
	Ren_->AddActor(Actor_) ;\
	_VSP_(vtkRenderWindow , RenWin_) ;\
	RenWin_->AddRenderer(Ren_);\
	RenWin_->SetSize(800 , 600);\
	RenWin_->Render() ;\
	RenWin_->SetWindowName(sname_.c_str()) ;\
	_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;\
	_VSP_(vtkRenderWindowInteractor , IRen_) ;\
	IRen_->SetInteractorStyle(Trackball_);\
	IRen_->SetRenderWindow(RenWin_);\
	IRen_->Initialize() ;\
	IRen_->Start();\


#ifdef _DISP_RENDERER_
#undef _DISP_RENDERER_
#endif
#define _DISP_RENDERER_(Ren_ , sname_)\
	_VSP_(vtkRenderWindow , RenWin_) ;\
	RenWin_->AddRenderer(Ren_);\
	RenWin_->SetSize(800 , 600);\
	RenWin_->Render() ;\
	RenWin_->SetWindowName(sname_.c_str()) ;\
	_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;\
	_VSP_(vtkRenderWindowInteractor , IRen_) ;\
	IRen_->SetInteractorStyle(Trackball_);\
	IRen_->SetRenderWindow(RenWin_);\
	IRen_->Initialize() ;\
	IRen_->Start();\


#ifndef _CHECK_FOLDER_
#undef _CHECK_FOLDER_
#endif
#define _CHECK_FOLDER_(sdir_ , chFunctionName_)\
	if(!stlplus::folder_exists(sdir_) || stlplus::folder_empty(sdir_)){\
	fflush(stderr) ;\
	fprintf(stderr, "Error in <DepthFieldHelper/%s>" , chFunctionName_);\
	fflush(stderr);\
	exit(0);}\



namespace stereoscene{


	bool
		DepthFieldHelper::
		synthesizeScene(
		const bool BOfflineSynthesize , 
		const bool BOnlyStructure , 
		const bool BSaveScene,
		const std::string& sdir_Save ,
		const std::string& sname_Save ,
		std::vector<float>& x3P ,
		std::map<int , PhotoCamera> &map_PCams , 
		const int kthRef , 
		const int NPad ,
		const cv::Mat& DepthMap ,
		const bool BSynthTexture ,
		const std::vector<std::string> &vec_sTexName ,
		std::vector<unsigned char>& x3T ) {


			std::vector<float> x3N ;
			std::vector<cind> x3C ;

			TrimeshGenerator::triangulateDelaunayMesh(x3C , x3P , 3 , false , x3N , true);


			_VSP_(vtkFloatArray , FPoints);
			FPoints->SetNumberOfComponents(3) ;
			FPoints->SetNumberOfTuples(x3P.size()/3);
			FPoints->SetArray(x3P.data() , x3P.size() , 1);

			_VSP_(vtkPoints, Points3) ;
			Points3->SetData(FPoints);

			_VSP_(vtkPolyData, Polys);
			Polys->SetPoints(Points3);


			if(!x3N.empty()) {
				_VSP_(vtkFloatArray,FNormals);
				FNormals->SetNumberOfComponents(3) ;
				FNormals->SetArray(x3N.data(),x3N.size(),1);
				Polys->GetPointData()->SetNormals(FNormals);
			}


			if(BSynthTexture && !x3T.empty()) {
				_VSP_(vtkUnsignedCharArray , RGBTex3);
				RGBTex3->SetNumberOfComponents(3);
				RGBTex3->SetNumberOfTuples(x3T.size()/3);
				RGBTex3->SetArray(x3T.data() , x3T.size() ,1);

				Polys->GetPointData()->SetScalars(RGBTex3);
			}


#if 0
			std::vector<cind> x3L ;
			extractValidLinesFromCells(x3L,x3C,x3P,5.0f) ;

			_VSP_(vtkIdTypeArray , IndArray);
			IndArray->SetNumberOfComponents(2);
			IndArray->SetArray(x3L.data() , x3L.size() , 1);

			_VSP_(vtkCellArray , Lines2);
			Lines2->SetCells(x3L.size()/3 , IndArray);

			//Polys->SetPolys(TriCells);
			Polys->SetLines(Lines2);
			Polys->BuildLinks() ;

#else 
			std::vector<cind> x3Tri ;
			extractValidTrianglesFromCells(x3Tri , x3C , x3P , 5.0f);

			_VSP_(vtkIdTypeArray , IndArray);
			IndArray->SetNumberOfComponents(3);
			IndArray->SetArray(x3Tri.data() , x3Tri.size() , 1);
			
			_VSP_(vtkCellArray , Triangles3);
			Triangles3->SetCells(x3Tri.size()/4 , IndArray);

			Polys->SetPolys(Triangles3);
			//Polys->BuildCells() ;
#endif



			_VSP_(vtkTriangleFilter , PTriangulator) ;
			PTriangulator->SetInputData(Polys);
			PTriangulator->Update() ;


			_VSP_(vtkCleanPolyData,PCleaner);
			//PCleaner->SetInputData(Polys);
			PCleaner->SetInputConnection(PTriangulator->GetOutputPort());
			PCleaner->SetPieceInvariant(0) ;
			PCleaner->Update();


			_VSP_(vtkPolyDataMapper , PMapper);
			//PMapper->SetInputConnection(PTriangulator->GetOutputPort());
			PMapper->SetInputConnection(PCleaner->GetOutputPort());
			//PMapper->SetInputData(Polys);
			PMapper->Update();


			const std::string& sext = stlplus::extension_part(sname_Save) ;
			std::string sname_savePhoto ;

			if(BSaveScene && BOfflineSynthesize) {

				char chname[256] ;
				if(kthRef<10){
					sprintf(chname , "%s/00%d/%s\0" , sdir_Save.c_str() , kthRef , sname_Save.c_str());
				}
				else if(kthRef<100 && kthRef>=10) {
					sprintf(chname , "%s/0%d/%s\0" , sdir_Save.c_str() , kthRef , sname_Save.c_str());
				}
				else if(kthRef<1000 && kthRef>=100) {
					sprintf(chname , "%s/%d/%s\0" , sdir_Save.c_str() , kthRef , sname_Save.c_str());
				}

				sname_savePhoto.assign(chname);

				if(sext=="ply") {

					_VSP_(vtkPLYWriter , Ply);
					Ply->SetInputData(PMapper->GetInput());
					Ply->SetFileName(chname);
					Ply->Write() ;
				}
			}

			//std::cout<<sname_savePhoto<<"\n" ;

			bool bsext( false ) ;
			if( BOfflineSynthesize &&
				(sext=="bmp" || sext=="png" || sext=="tif" || sext=="jpg")) {
					bsext = true ;
					//std::cout<<sext<<"\n";
			}



			_VSP_(vtkActor, PActor);
			PActor->SetMapper(PMapper);
			if(!BOnlyStructure) {
				const double *pcen3 = PActor->GetCenter() ;
				PActor->RotateY(3) ;
			}
			PActor->SetPosition(0 , 0 , -1);
		

			//PActor->SetScale(1.0 , 1.0 , 1.0);
			//PActor->GetProperty()->SetBackfaceCulling(1);


			double* pcen = PActor->GetCenter();


			double pbox[6]={0};
			PActor->GetBounds(pbox) ;


			_VSP_(vtkRenderer , PRen);
			PRen->AddActor(PActor) ;

			if(BSynthTexture) {
				PRen->SetBackground(0.32,0.32,0.32) ;
				PRen->SetBackground2(0,0,0);
				PRen->SetGradientBackground(1);
			}
		

			//float frange = std::sqrt(pbox[3]*pbox[3]+1) ;


			vtkCamera* RenCam = PRen->GetActiveCamera();

			if(BOnlyStructure) {

				//RenCam->SetParallelProjection(1);
				RenCam->SetPosition( 0 , 0 , 0 );
				RenCam->SetFocalPoint( 0 , 0 , -1 );
				//RenCam->SetParallelScale( (pbox[3]-pbox[2])/1.2 );
				RenCam->SetClippingRange(0.0001 , 100);
			}
			else {

				RenCam->SetParallelProjection(1);
				RenCam->SetPosition( 0 , std::fabs(pbox[3])*1.2  , std::fabs(pbox[4])*0.1 );
				RenCam->SetFocalPoint( 0 , 0 ,  -std::fabs(pbox[5])*1.0 );
				RenCam->SetParallelScale( (pbox[3]-pbox[2])/1.2 );
				RenCam->SetClippingRange(0.0001 , 100);

			}


			if(!BOnlyStructure) {


				const double *const pcen = PActor->GetCenter() ;


				//std::map<int ,PhotoCamera> map_NewPCams;
				//const int Indi = kthRef , Indj = 0;
				//transformPCamsI2J(map_NewPCams ,Indi , Indj , map_PCams);

				const std::map<int ,PhotoCamera>::iterator it_iPCam = map_PCams.find(kthRef);
				PhotoCamera& iPCam = it_iPCam->second ;
				const cv::Mat Ri = cv::Mat(3 , 3 , CV_32FC1 , iPCam._R_.data()) ,
					Ti = cv::Mat(3 , 1 , CV_32FC1 , iPCam._Tvec_.data()) ;

				cv::Mat Rj = cv::Mat(3,3,CV_32FC1) ;
				cv::setIdentity(Rj);
				cv::Mat Cj = cv::Mat::zeros(3,1, CV_32FC1);
				Cj.at<float>(2) = std::fabs(pbox[4])*0.2   ;
				cv::Mat Tj = -Rj*Cj;


				/// i is defined as reference camera and j is the camera need to be working on :
				///
				/// Rij = Rj * RiT 
				///	Tij = Tj - Rij * Ti
				///
				const cv::Mat Rij = Rj*(Ri.t()) ;
				const cv::Mat Tij = Tj - Rij*Ti ;


				for(int nc=0 ; nc<=kthRef+NPad ; nc++) {

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
					//std::cout<<pcen[1]<<std::endl ;
					double camcen[3] = {Cnew.at<float>(0) , Cnew.at<float>(1)-3.0 , Cnew.at<float>(2)};
					double cdir[3] = {(camcen[0]-pcen[0]) , (camcen[1]-pcen[1]) , -(camcen[2]-pcen[2])};

					_VSP_(vtkConeSource , cam);
					cam->SetCenter(camcen[0] , -camcen[1]  ,camcen[2]);
				
					cam->SetDirection(cdir);
					cam->Update();

					_VSP_(vtkPolyDataMapper , cammapper);
					cammapper->SetInputConnection(cam->GetOutputPort());
					cammapper->Update();

					_VSP_(vtkActor , camactor);
					camactor->SetMapper(cammapper);
					//camactor->SetOrigin(0 , 0 , 0);
					//camactor->RotateY(180);

					if(nc!=kthRef){
						camactor->GetProperty()->SetColor(1.0 , 1.0 , 1.0);
						camactor->SetScale(0.5);
					}
					else {
						camactor->GetProperty()->SetColor(1.0 , 0.6 , 0);
						camactor->SetScale(0.5);

#if 0
						if(BSynthTexture) {

							_VSP_(vtkTexture, texture ) ; 

							if(BSynthTexture && vec_sTexName.size()>kthRef && stlplus::file_exists(vec_sTexName[kthRef]) ) {

								const std::string& sext = stlplus::extension_part(vec_sTexName[kthRef]);

								//_SEE_IMAGE_(cv::imread(vec_sTexName[kthRef]) , "tex");

								if(sext=="bmp"){
									_VSP_(vtkBMPReader , bmp) ;
									bmp->SetFileName(vec_sTexName[kthRef].c_str());
									//bmp->Update() ;
									texture->SetInputConnection(bmp->GetOutputPort());
								}
								else if(sext=="jpg") {
									_VSP_(vtkJPEGReader , jpg) ;
									jpg->SetFileName(vec_sTexName[kthRef].c_str());
									//jpg->Update() ;
									texture->SetInputConnection(jpg->GetOutputPort());
								}
								else if(sext=="png") {				
									_VSP_(vtkPNGReader , png) ;
									png->SetFileName(vec_sTexName[kthRef].c_str());
									//png->Update() ;
									texture->SetInputConnection(png->GetOutputPort());
								}
								else if(sext=="tiff"){
									_VSP_(vtkTIFFReader , tiff) ;
									tiff->SetFileName(vec_sTexName[kthRef].c_str());
									//tiff->Update() ;
									texture->SetInputConnection(tiff->GetOutputPort());
								}
							}
							else {

								if(!DepthMap.empty()) {

									_VSP_(vtkImageImport , imgport);
									imgport->SetDataScalarTypeToUnsignedChar();
									imgport->SetImportVoidPointer((void*)DepthMap.ptr<unsigned char>(0) , 1);
									imgport->SetDataExtent(0 , DepthMap.cols-1 , 0 , DepthMap.rows-1, 0 ,0);
									imgport->SetWholeExtent(0 , DepthMap.cols-1 , 0 , DepthMap.rows-1 , 0 , 0);
									imgport->SetDataOrigin(0, 0, 0);
									imgport->SetNumberOfScalarComponents(1);
									imgport->SetDataExtentToWholeExtent();
									imgport->SetDataSpacing(1 , 1 , 1);
									//imgport->Update() ;


									texture->SetInputConnection(imgport->GetOutputPort());
								}

							}

							texture->EdgeClampOn() ;
							texture->Update();


							// Create a plane
							double planecen[3] = {camcen[0] , camcen[1]-0.6 , camcen[2]} ;

							_VSP_(vtkPlaneSource , splane) ;
							splane->SetCenter(planecen[0] , planecen[1] ,planecen[2] );
							splane->SetNormal(0.0, -1.0, -0.0);
							splane->SetXResolution(4);
							splane->SetYResolution(3);
							splane->Update();


							// Apply the texture
							_VSP_(vtkTextureMapToPlane , texplane) ;
							texplane->SetInputConnection(splane->GetOutputPort());
							texplane->Update() ;

							_VSP_(vtkPolyDataMapper , texmapper);
							texmapper->SetInputConnection(texplane->GetOutputPort());
							texmapper->Update();

							_VSP_(vtkActor , texplaneActor);
							texplaneActor->SetMapper(texmapper);
							texplaneActor->SetTexture(texture);
							texplaneActor->SetScale(5.0);
							texplaneActor->SetOrigin(planecen[0] , planecen[1] , planecen[2]) ;
							texplaneActor->RotateY(180);
							texplaneActor->RotateZ(180);
							//texplaneActor->SetOrigin(0 , 0 , 0);
							//texplaneActor->RotateX(90);

							PRen->AddActor(texplaneActor);

						}
#endif
					}

					camactor->SetOrigin(0,0,0);
					camactor->RotateX(180);


					PRen->AddActor(camactor);

				}

			}


			_VSP_(vtkRenderWindow , PRenWin);
			PRenWin->AddRenderer(PRen);
			PRenWin->SetSize(800 , 600);
			PRenWin->SetPointSmoothing(1);
			//PRenWin->SetPolygonSmoothing(1);
			PRenWin->SetLineSmoothing(1);
			if(BOfflineSynthesize) {
				PRenWin->SetOffScreenRendering(1) ;
			}

			PRenWin->Render();


			if(BOfflineSynthesize && bsext){
				_VSP_(vtkWindowToImageFilter , WinImage) ;
				WinImage->SetInput(PRenWin);
				WinImage->SetInputBufferTypeToRGBA() ;
				WinImage->Update() ;

				//std::string &sbase = stlplus::basename_part(sname_savePhoto) ;
				//std::cout<<sbase<<"\n" ;

				_VSP_(vtkBMPWriter , Bmp) ;
				Bmp->SetInputConnection(WinImage->GetOutputPort());
				Bmp->SetFileName(sname_savePhoto.c_str());
				//Bmp->Se
				Bmp->Update() ;

			}
			else {

				PRenWin->SetWindowName("Triangulated Depth") ;

				_VSP_(vtkInteractorStyleTrackballCamera , Tball);
				_VSP_(vtkRenderWindowInteractor ,  IRen);
				IRen->SetRenderWindow(PRenWin);
				IRen->SetInteractorStyle(Tball) ;
				IRen->Initialize() ;
				IRen->Start() ;
			}

			return true ;
	}



	bool
		DepthFieldHelper::
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
		DepthFieldHelper::
		triangulateDepthMap(
		std::vector<float>& x3P,
		const PhotoCamera& PCams ,
		const cv::Mat& DepthMap , 
		const cv::Mat& DepthMask,
		const std::vector<float>& DepthBox,
		const bool BGetTexture ,
		const cv::Mat& TexImage ,
		std::vector<unsigned char>& x3T ,
		const int BorderBound) {


			const int NWidth = DepthMap.cols , 
				NHeight = DepthMap.rows ;

			const int WBorderGap = BorderBound + _BORDER_NGAP_ ;
			const int HBorderGap = int(BorderBound*0.8)+_BORDER_NGAP_ ;

			cv::Mat DMap ;
			cv::normalize(DepthMap , DMap , DepthBox[0], DepthBox[1] ,cv::NORM_MINMAX , CV_32FC1 , DepthMask);
			DMap.setTo(0 , 255-DepthMask);


			const std::vector<float>& Kvec = PCams._K_ ;

			x3P.clear() ;
			x3P.reserve(NWidth*NHeight*3);

			if(BGetTexture){
				x3T.clear();
				x3T.reserve(NWidth*NHeight*3);
			}

#pragma omp parallel for schedule(dynamic, 1)
			for(int nh=HBorderGap ; nh<NHeight-_BORDER_NGAP_-1 ; nh++) {

				const float *pdval=DMap.ptr<float>(nh)+WBorderGap ;
				const unsigned char* pdmask = DepthMask.ptr<unsigned char>(nh)+WBorderGap ;

				const cv::Vec3b* pdtex = NULL ;
				if(BGetTexture){
					///must to be rgb image
					pdtex = TexImage.ptr<cv::Vec3b>(nh) ;
				}

				for(int nw=WBorderGap ; nw<NWidth-WBorderGap ; nw++ , pdval++ ,pdmask++) {
					if(*pdmask) {

						float rx = (nw-Kvec[2])/Kvec[4] , ry = (nh-Kvec[5])/Kvec[4];

						const float rn = std::sqrt(rx*rx+ry*ry+1.0) ;

						float rz = *pdval/rn ;

						rx*=rz ;
						ry*=-rz ;
						rz*= -1 ; ///reverse the orientation

#pragma omp critical
						{
							x3P.push_back(rx),
								x3P.push_back(ry),
								x3P.push_back(rz);

							if(BGetTexture) {
								x3T.push_back(pdtex[nw].val[2]), 
									x3T.push_back(pdtex[nw].val[1]), 
									x3T.push_back(pdtex[nw].val[0]);
							}
						}

					}
				}
			}
			std::vector<float> (x3P).swap(x3P) ;


			return true ;
	}

	bool
		DepthFieldHelper::
		extractValidTrianglesFromCells(
		std::vector<cind>& x3Tri,
		const std::vector<cind>& x3C,
		const std::vector<float>& x3P,
		const float FThreshCell) {


		///statistic triangles
		///
		const int NCells = x3C.size()/4,
			NPoints = x3P.size() / 3;

		std::vector<double> vec_EdgeLength(NCells*3);

#pragma omp parallel for schedule(dynamic , 1)
		for(int nc=0 ; nc<NCells ; nc++) {

			const cind* px3c = &x3C[nc*4] ;

			const float *px3p0 = &x3P[px3c[1]*3];
			const float *px3p1 = &x3P[px3c[2]*3];
			const float *px3p2 = &x3P[px3c[3]*3];

			///0->1 , 1->2 , 2->0
			vec_EdgeLength[nc*3] = std::sqrt((px3p0[0]-px3p1[0])*(px3p0[0]-px3p1[0]) + (px3p0[1]-px3p1[1])*(px3p0[1]-px3p1[1]) + (px3p0[2]-px3p1[2])*(px3p0[2]-px3p1[2])) ;
			vec_EdgeLength[nc*3+1]= std::sqrt((px3p1[0]-px3p2[0])*(px3p1[0]-px3p2[0]) + (px3p1[1]-px3p2[1])*(px3p1[1]-px3p2[1]) + (px3p1[2]-px3p2[2])*(px3p1[2]-px3p2[2])) ;
			vec_EdgeLength[nc*3+2]= std::sqrt((px3p2[0]-px3p0[0])*(px3p2[0]-px3p0[0]) + (px3p2[1]-px3p0[1])*(px3p2[1]-px3p0[1]) + (px3p2[2]-px3p0[2])*(px3p2[2]-px3p0[2])) ;
		}

		double DMeanEdge = std::accumulate(vec_EdgeLength.begin(),vec_EdgeLength.end(),0.0)/vec_EdgeLength.size() ;

		double DSvarEdge = 0;

		std::for_each(vec_EdgeLength.begin(),
			vec_EdgeLength.end(),
			[&DMeanEdge,&DSvarEdge](const double &len){
			DSvarEdge +=  (len-DMeanEdge)*(len-DMeanEdge) ;
		});
		DSvarEdge/=vec_EdgeLength.size();
		DSvarEdge = std::sqrt(DSvarEdge);


		x3Tri.clear() ;
		x3Tri.reserve(NCells*4);

#pragma omp parallel for schedule(dynamic, 1)
		for(int nc=0 ; nc<NCells ; nc++) {

			const cind* const pCell = &x3C[nc*4] ;
			const cind ind0 = pCell[1],ind1 = pCell[2],ind2 = pCell[3] ;

			const double *const pLenEdge = &vec_EdgeLength[nc*3];

			bool sw0 = (pLenEdge[0]<DMeanEdge+FThreshCell*DSvarEdge) ? true : false;
			bool sw1 = (pLenEdge[1]<DMeanEdge+FThreshCell*DSvarEdge) ? true : false;
			bool sw2 = (pLenEdge[2]<DMeanEdge+FThreshCell*DSvarEdge) ? true : false;

			if(sw0 && sw1 && sw2) {
#pragma omp critical
				{
					x3Tri.push_back(3),
						x3Tri.push_back(pCell[1]) ,
						x3Tri.push_back(pCell[2]) ,
						x3Tri.push_back(pCell[3]) ;
				}
			}
		}

		std::vector<cind> (x3Tri).swap(x3Tri);


		return true ;
	}


	bool
		DepthFieldHelper::
		extractValidLinesFromCells(
		std::vector<cind>& x3L ,
		const std::vector<cind>& x3C,
		const std::vector<float>& x3P,
		const float FThreshCell) {


		///statistic triangles
		///
		const int NCells = x3C.size()/4,
			NPoints = x3P.size() / 3;

		std::vector<double> vec_EdgeLength(NCells*3 );

#pragma omp parallel for schedule(dynamic , 1)
		for(int nc=0 ; nc<NCells ; nc++) {

			const cind* px3c = &x3C[nc*4] ;

			const float *px3p0 = &x3P[px3c[1]*3];
			const float *px3p1 = &x3P[px3c[2]*3];
			const float *px3p2 = &x3P[px3c[3]*3];

			///0->1 , 1->2 , 2->0
			vec_EdgeLength[nc*3] = std::sqrt((px3p0[0]-px3p1[0])*(px3p0[0]-px3p1[0]) + (px3p0[1]-px3p1[1])*(px3p0[1]-px3p1[1]) + (px3p0[2]-px3p1[2])*(px3p0[2]-px3p1[2])) ;
			vec_EdgeLength[nc*3+1]= std::sqrt((px3p1[0]-px3p2[0])*(px3p1[0]-px3p2[0]) + (px3p1[1]-px3p2[1])*(px3p1[1]-px3p2[1]) + (px3p1[2]-px3p2[2])*(px3p1[2]-px3p2[2])) ;
			vec_EdgeLength[nc*3+2]= std::sqrt((px3p2[0]-px3p0[0])*(px3p2[0]-px3p0[0]) + (px3p2[1]-px3p0[1])*(px3p2[1]-px3p0[1]) + (px3p2[2]-px3p0[2])*(px3p2[2]-px3p0[2])) ;
		}

		double DMeanEdge = std::accumulate(vec_EdgeLength.begin(),vec_EdgeLength.end(),0.0)/vec_EdgeLength.size() ;

		double DSvarEdge = 0;

		std::for_each(vec_EdgeLength.begin(),
			vec_EdgeLength.end(),
			[&DMeanEdge , &DSvarEdge](const double &len){
			DSvarEdge +=  (len-DMeanEdge)*(len-DMeanEdge) ;
		});
		DSvarEdge/=vec_EdgeLength.size();
		DSvarEdge = std::sqrt(DSvarEdge);


		///extract valid lines 
		///
		x3L.clear() ;
		x3L.reserve(NCells*3);

#pragma omp parallel for schedule(dynamic, 1)
		for (int nc=0 ; nc<NCells ; nc++) {
			
				const cind* const pCell = &x3C[nc*4] ;
				const cind ind0 = pCell[1] , ind1 = pCell[2] , ind2 = pCell[3] ;

				const double *const pLenEdge = &vec_EdgeLength[nc*3];

				bool sw0 = (pLenEdge[0]<DMeanEdge+FThreshCell*DSvarEdge) ? true : false;
				bool sw1 = (pLenEdge[1]<DMeanEdge+FThreshCell*DSvarEdge) ? true : false;
				bool sw2 = (pLenEdge[2]<DMeanEdge+FThreshCell*DSvarEdge) ? true : false;

				if(sw0 && sw1 && sw2) {
#pragma omp critical
					{
						x3L.push_back(2),x3L.push_back(ind0),x3L.push_back(ind1),
							x3L.push_back(2),x3L.push_back(ind1),x3L.push_back(ind2),
							x3L.push_back(2),x3L.push_back(ind2),x3L.push_back(ind0);
					}
				}

		}

		std::vector<cind> (x3L).swap(x3L);


		return true;
	}


}


namespace stereoscene{


	bool
		DepthFieldHelper::
		seeTrimeshTextured(
		float *const _px3Points ,
		const int NPoints , 
		cind *const _px3TriCells , 
		const int NTriCells ,
		unsigned char *const _px3Texture , 
		const int NChannels , 
		float *const _px3Normals ,
		const bool BInverse , 
		const std::string& _sname_Window,
		const double DScale , 
		const bool BSaveMeshPhoto ,
		const std::string& _sdir ,
		const std::string& _sname_Save)  {


			if(BSaveMeshPhoto) {
				if(!stlplus::folder_exists(_sdir)) {
					stlplus::folder_create(_sdir);
				}
			}


			_VSP_(vtkPolyData , Polys) ;


			///Points' Coordinates
			_VSP_(vtkFloatArray , FPointsArray) ;
			FPointsArray->SetNumberOfComponents(3);
			FPointsArray->SetArray(_px3Points , NPoints*3 , 1);
			_VSP_(vtkPoints , PointsCoord) ;
			PointsCoord->SetData(FPointsArray);
			Polys->SetPoints(PointsCoord);


			///TriCells' Edges List
			///	each group int[4] := { npts , c[n].v0Ind , c[n].v1Ind , c[n].v2Ind }
			_VSP_(vtkIdTypeArray , IIndArray);
			IIndArray->SetArray(_px3TriCells , NTriCells*4 , 1);
			_VSP_(vtkCellArray , TCells_);
			TCells_->SetCells(NTriCells , IIndArray);
			Polys->SetPolys(TCells_);


			///Points' Normals (if it's been got)
			if(_px3Normals!=NULL) {
				_VSP_(vtkFloatArray , FNormalsArray);
				FNormalsArray->SetNumberOfComponents(3);
				FNormalsArray->SetArray(_px3Normals , NPoints*3 ,1);
				Polys->GetPointData()->SetNormals(FNormalsArray) ;
			}


			///Points' Textures
			_VSP_(vtkUnsignedCharArray , UScalarArray) ;
			UScalarArray->SetNumberOfComponents(NChannels);
			UScalarArray->SetArray(_px3Texture , NPoints*NChannels , 1);
			UScalarArray->SetName("Texture");
			Polys->GetPointData()->SetScalars(UScalarArray);


			_VSP_(vtkCleanPolyData , PClear);
			PClear->SetInputData(Polys);
			PClear->Update() ;

			_VSP_(vtkPolyDataMapper , PlyMapper);
			//PlyMapper->SetInputData(Polys);
			PlyMapper->SetInputConnection(PClear->GetOutputPort());
			PlyMapper->Update() ;


			_VSP_(vtkActor , PlyActor);
			PlyActor->SetMapper(PlyMapper);
			PlyActor->SetScale(DScale);
			double *pcen = PlyActor->GetCenter();
			if(BInverse){
				PlyActor->SetOrigin(0, 0 , 0);
				PlyActor->RotateX(180);
			} 



			_VSP_(vtkRenderer , PlyRen) ;
			PlyRen->AddActor(PlyActor) ;
			PlyRen->SetBackground(0.0 ,0.0 ,0.0);


			double *pbound = PlyActor->GetBounds() ;
			vtkCamera* pPCam = PlyRen->GetActiveCamera();
			pPCam->SetParallelProjection(1);
			pPCam->SetPosition(0 , 0 , pPCam->GetDistance());
			pPCam->SetFocalPoint(0 , 0 , 0) ;
			pPCam->SetParallelScale((pbound[3]-pbound[2])/2.0);

			_VSP_(vtkRenderWindow , PlyRenWin) ;
			PlyRenWin->AddRenderer(PlyRen);
			PlyRenWin->SetSize(800 , 600);


			if(BSaveMeshPhoto){
				PlyRenWin->OffScreenRenderingOn();
			}

			PlyRenWin->Render() ;

			if(!_sname_Window.empty()) {
				PlyRenWin->SetWindowName(_sname_Window.c_str()) ;
			} else {
				PlyRenWin->SetWindowName("Textured Trimesh");
			}



			if(!BSaveMeshPhoto ) {

				_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;
				_VSP_(vtkRenderWindowInteractor , IRen_) ;
				IRen_->SetInteractorStyle(Trackball_);
				IRen_->SetRenderWindow(PlyRenWin);
				IRen_->Initialize() ;
				IRen_->Start();

			} else {

				_VSP_(vtkWindowToImageFilter , WinImage);
				WinImage->SetInput(PlyRenWin);
				WinImage->SetInputBufferTypeToRGBA();
				WinImage->Update() ;

				_VSP_(vtkBMPWriter , BMPSaver);
				BMPSaver->SetInputConnection(WinImage->GetOutputPort());
				BMPSaver->SetFileName(std::string(_sdir+"/"+_sname_Save).c_str());
				BMPSaver->Update();
			}


			return true;
	}



	bool 
		DepthFieldHelper::
		seeTrimesh(
		float *const _px3Points , 
		const int NPoints , 
		cind *const _px3TriCells , 
		const int NTriCells , 
		float *const _px3Normals ,
		const bool BInverse , 
		const std::string& _sname_Window ,
		const double DScale ,
		const bool BSaveMeshPhoto,
		const std::string& _sdir ,
		const std::string& _sname_Save) {


			if(BSaveMeshPhoto) {
				if(!stlplus::folder_exists(_sdir)) {
					stlplus::folder_create(_sdir);
				}
			}


			_VSP_(vtkPolyData , Polys);


			///Points
			_VSP_(vtkFloatArray , FPointsArray) ;
			FPointsArray->SetNumberOfComponents(3);
			FPointsArray->SetArray(_px3Points , NPoints*3 , 1);

			_VSP_(vtkPoints , Points) ;
			Points->SetData(FPointsArray);
			Polys->SetPoints(Points) ;



			///Cells. Note this: 
			///	IdTypeArray: {NumberOfVerticesInEachCell , T[0].v0 , T[0].v1 , T[0].v2 , ... , NumberOfVerticesInEachCell, T[n-1].v0 , T[n-1].v1 , T[n-1].v2}
			///
			_VSP_(vtkIdTypeArray , ICellsArray) ;
			ICellsArray->SetArray(_px3TriCells , NTriCells*4 , 1);

			_VSP_(vtkCellArray , TriCells) ;
			TriCells->SetCells(NTriCells , ICellsArray);
			Polys->SetPolys(TriCells) ;



			///Normals
			if(_px3Normals!=NULL) {
				_VSP_(vtkFloatArray, FPNormalsArray) ;
				FPNormalsArray->SetNumberOfComponents(3);
				FPNormalsArray->SetArray(_px3Normals , NPoints*3 , 1);
				Polys->GetPointData()->SetNormals(FPNormalsArray);
			}


			_VSP_(vtkWindowedSincPolyDataFilter , PSmoother);
			PSmoother->SetInputData(Polys);
			PSmoother->SetBoundarySmoothing(0);
			PSmoother->SetNormalizeCoordinates(1);
			PSmoother->SetNumberOfIterations(20);
			PSmoother->Update();

			_VSP_(vtkCleanPolyData , PClear);
			PClear->SetInputConnection(PSmoother->GetOutputPort());
			PClear->Update() ;

			_VSP_(vtkPolyDataNormals , PNorm);
			PNorm->SetInputConnection(PSmoother->GetOutputPort());
			PNorm->SetFeatureAngle(30);
			PNorm->SetAutoOrientNormals(1);
			PNorm->SetConsistency(0);
			PNorm->SetSplitting(1);
			PNorm->Update();

			/*		_VSP_(vtkButterflySubdivisionFilter , PSub);
			PSub->SetInputConnection(PSmoother->GetOutputPort());
			PSub->SetNumberOfSubdivisions(1);
			PSub->Update();*/


			_VSP_(vtkPolyDataMapper , PlyMapper) ;
			//PlyMapper->SetInputData(Polys);
			PlyMapper->SetInputConnection(PNorm->GetOutputPort());
			PlyMapper->Update() ;

			_VSP_(vtkActor , PlyActor);
			PlyActor->SetMapper(PlyMapper);
			PlyActor->SetScale(DScale);
			double *pcen = PlyActor->GetCenter();
			if(BInverse){
				PlyActor->SetOrigin(0, 0 , 0);
				PlyActor->RotateX(180);
			} 



			_VSP_(vtkRenderer , PlyRen) ;
			PlyRen->AddActor(PlyActor) ;
			PlyRen->SetBackground(0.0 ,0.0 ,0.0);



			double *pbound = PlyActor->GetBounds() ;
			vtkCamera* pPCam = PlyRen->GetActiveCamera();
			pPCam->SetParallelProjection(1);
			pPCam->SetPosition(0 , 0 , 0+pPCam->GetDistance());
			pPCam->SetFocalPoint(0 , 0 , 0) ;
			pPCam->SetParallelScale((pbound[3]-pbound[2])/2.0);


			_VSP_(vtkRenderWindow , PlyRenWin) ;
			PlyRenWin->AddRenderer(PlyRen);
			PlyRenWin->SetSize(800 , 600);
			PlyRenWin->SetPolygonSmoothing(1);
			if(BSaveMeshPhoto) {
				PlyRenWin->OffScreenRenderingOn();
			}

			PlyRenWin->Render() ;

			if(!_sname_Window.empty()) {
				PlyRenWin->SetWindowName(_sname_Window.c_str()) ;
			} else {
				PlyRenWin->SetWindowName("Trimesh");
			}


			if(!BSaveMeshPhoto) {

				_VSP_(vtkInteractorStyleTrackballCamera , Trackball_) ;
				_VSP_(vtkRenderWindowInteractor , IRen_) ;
				IRen_->SetInteractorStyle(Trackball_);
				IRen_->SetRenderWindow(PlyRenWin);
				IRen_->Initialize() ;
				IRen_->Start();

			} else {

				_VSP_(vtkWindowToImageFilter , WinImage);
				WinImage->SetInput(PlyRenWin);
				WinImage->SetInputBufferTypeToRGBA();
				WinImage->Update() ;

				_VSP_(vtkBMPWriter , BMPSaver);
				BMPSaver->SetInputConnection(WinImage->GetOutputPort());
				BMPSaver->SetFileName(std::string(_sdir+"/"+_sname_Save).c_str());
				BMPSaver->Update();
			}

			return true ;
	}


}



namespace stereoscene{


	bool
		DepthFieldHelper::
		savePoints(
		const std::vector<std::vector<float>>& _vec_x3Points ,
		const std::string &_sdir , 
		const std::string &_sname) {

			if(!stlplus::folder_exists(_sdir)){
				stlplus::folder_create(_sdir);
			}


			const int NPoints = _vec_x3Points.size();

			std::fstream fs(_sdir+"/"+_sname , std::ios::out);
			fs<<NPoints<<"\n";

			std::for_each(_vec_x3Points.begin() , _vec_x3Points.end(), 
				[&fs](const std::vector<float>& _x3){
					fs<<_x3[0]<<"\t"<<_x3[1]<<"\t"<<_x3[2]<<"\n" ;
			});

			fs.close();

			return true ;
	}



	bool 
		DepthFieldHelper::
		loadSpecificImage(
		cv::Mat& _DstImage , 
		const std::string& sdir_Image , 
		const cv::Size& ImageSize ,
		const int Kth ,
		const bool BGray) {

			_CHECK_FOLDER_(sdir_Image , "loadSpecificImage") ;

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

						if(ImageSize.height!=_DstImage.rows || ImageSize.width!=_DstImage.cols){
							cv::resize(_DstImage , _DstImage , ImageSize , 0 ,0 , cv::INTER_CUBIC) ;
						}

						break; 
					}

					imgCnt++ ;
				}

			}


			return true ;
	}


}

#include "DepthFieldCommon/ConcaveHullExtractor.h"


namespace stereoscene{


	bool
		DepthFieldHelper::
		findConcaveMaskfromPointsVector(
		cv::Mat& ConcaveMask ,
		const cv::Size& MapSize ,
		const std::vector<float>& Pvec,
		const std::vector<std::vector<float> >& vec_x3Points) {


			const int NPoints3 = vec_x3Points.size()/3 ;

			std::vector<std::vector<cv::Point>> vec_cchull(1) ;
			ConcaveHullExtractor::extractConcaveHullfromPointsVector3(vec_cchull[0] , MapSize ,Pvec , vec_x3Points);

			ConcaveMask.create(MapSize, CV_8UC1);

			cv::Mat colormask = cv::Mat(MapSize , CV_8UC3);
			cv::drawContours(colormask , vec_cchull , -1 , cv::Scalar(255 , 255 ,255) , cv::FILLED);
			cv::cvtColor(colormask , ConcaveMask , cv::COLOR_BGR2GRAY);
			ConcaveMask.convertTo(ConcaveMask , CV_8UC1 ,255.0);

			cv::dilate(ConcaveMask , ConcaveMask , cv::getStructuringElement(cv::MORPH_ELLIPSE ,cv::Size(3,3)));

			cv::findContours(ConcaveMask , vec_cchull , cv::RETR_EXTERNAL , cv::CHAIN_APPROX_NONE);
			ConcaveMask.col(0).setTo(0);
			ConcaveMask.col(ConcaveMask.cols-1).setTo(0);
			ConcaveMask.row(0).setTo(0);
			ConcaveMask.row(ConcaveMask.rows-1).setTo(0);


			colormask.create(MapSize , CV_8UC3);
			cv::drawContours(colormask , vec_cchull , -1 , cv::Scalar(255 ,255 ,255) , cv::FILLED);
			cv::cvtColor(colormask , ConcaveMask , cv::COLOR_BGR2GRAY);
			ConcaveMask.convertTo(ConcaveMask , CV_8UC1 ,255.0);
			ConcaveMask.colRange(0,2).setTo(0);
			ConcaveMask.rowRange(0,2).setTo(0);
			ConcaveMask.colRange(ConcaveMask.cols-2 , ConcaveMask.cols).setTo(0);
			ConcaveMask.rowRange(ConcaveMask.rows-2 , ConcaveMask.rows).setTo(0);



			return true ;
	}



	bool
		DepthFieldHelper::
		findConcaveMaskfromPointsArray(
		cv::Mat& ConcaveMask ,
		const cv::Size& MapSize ,
		const std::vector<float>& Pvec,
		const std::vector<float>& x3P) {

			const int NPoints3 = x3P.size()/3 ;

			std::vector<std::vector<cv::Point>> vec_cchull(1) ;
			ConcaveHullExtractor::extractConcaveHullfromPointsArray3(vec_cchull[0] , MapSize ,Pvec , x3P);

			ConcaveMask.create(MapSize, CV_8UC1);

			cv::Mat colormask = cv::Mat(MapSize , CV_8UC3);
			cv::drawContours(colormask , vec_cchull , -1 , cv::Scalar(255 , 255 ,255) , cv::FILLED);
			cv::cvtColor(colormask , ConcaveMask , cv::COLOR_BGR2GRAY);
			ConcaveMask.convertTo(ConcaveMask , CV_8UC1 ,255.0);

			cv::dilate(ConcaveMask , ConcaveMask , cv::getStructuringElement(cv::MORPH_ELLIPSE , cv::Size(3,3)));

			cv::findContours(ConcaveMask , vec_cchull , cv::RETR_EXTERNAL , cv::CHAIN_APPROX_NONE);
			ConcaveMask.col(0).setTo(0);
			ConcaveMask.col(ConcaveMask.cols-1).setTo(0);
			ConcaveMask.row(0).setTo(0);
			ConcaveMask.row(ConcaveMask.rows-1).setTo(0);


			colormask.create(MapSize , CV_8UC3);
			cv::drawContours(colormask , vec_cchull , -1 , cv::Scalar(255 ,255 ,255) , cv::FILLED);
			cv::cvtColor(colormask , ConcaveMask , cv::COLOR_BGR2GRAY);
			ConcaveMask.convertTo(ConcaveMask , CV_8UC1 ,255.0);
			ConcaveMask.colRange(0,2).setTo(0);
			ConcaveMask.rowRange(0,2).setTo(0);
			ConcaveMask.colRange(ConcaveMask.cols-2 , ConcaveMask.cols).setTo(0);
			ConcaveMask.rowRange(ConcaveMask.rows-2 , ConcaveMask.rows).setTo(0);


			return true ;
	}



	bool
		DepthFieldHelper::
		findConcaveMaskfromMaskMap(
		cv::Mat& ConcaveMask ,
		const cv::Size& MapSize ,
		const cv::Mat& MotionMap){

			const int NHeight = MapSize.height , NWidth = MapSize.width ;


			std::vector<int> x2P;
			x2P.reserve(NHeight*NWidth) ;


#pragma omp parallel for schedule(dynamic, 1)
			for(int nh=0; nh<NHeight; nh++) {
				const unsigned char* pmot = MotionMap.ptr<unsigned char>(nh);
				for(int nw=0 ; nw<NWidth ; nw++ , pmot++) {
					if(*pmot){
#pragma omp critical
						{
							x2P.push_back(nw), x2P.push_back(nh);
						}
					}
				}
			}
			std::vector<int>(x2P).swap(x2P);


			std::vector<std::vector<cv::Point>> vec_cchull(1) ;
			ConcaveHullExtractor::extractConcaveHullfromPointsArray2(vec_cchull[0] , x2P);

			ConcaveMask.create(MapSize, CV_8UC1);

			cv::Mat colormask = cv::Mat(MapSize , CV_8UC3);
			cv::drawContours(colormask , vec_cchull , -1 , cv::Scalar(255 , 255 ,255) , cv::FILLED);
			cv::cvtColor(colormask , ConcaveMask , cv::COLOR_BGR2GRAY);
			ConcaveMask.convertTo(ConcaveMask , CV_8UC1 ,255.0);

			cv::dilate(ConcaveMask , ConcaveMask , cv::getStructuringElement(cv::MORPH_ELLIPSE , cv::Size(3,3)));

			cv::findContours(ConcaveMask , vec_cchull , cv::RETR_EXTERNAL , cv::CHAIN_APPROX_NONE);
			ConcaveMask.col(0).setTo(0);
			ConcaveMask.col(ConcaveMask.cols-1).setTo(0);
			ConcaveMask.row(0).setTo(0);
			ConcaveMask.row(ConcaveMask.rows-1).setTo(0);


			colormask.create(MapSize , CV_8UC3);
			cv::drawContours(colormask , vec_cchull , -1 , cv::Scalar(255 ,255 ,255) , cv::FILLED);
			cv::cvtColor(colormask , ConcaveMask , cv::COLOR_BGR2GRAY);
			ConcaveMask.convertTo(ConcaveMask , CV_8UC1 ,255.0);
			ConcaveMask.colRange(0,2).setTo(0);
			ConcaveMask.rowRange(0,2).setTo(0);
			ConcaveMask.colRange(ConcaveMask.cols-2 , ConcaveMask.cols).setTo(0);
			ConcaveMask.rowRange(ConcaveMask.rows-2 , ConcaveMask.rows).setTo(0);


			return true ;
	}



	bool 
		DepthFieldHelper::
		generateDepthBoundingMask(
		cv::Mat& _BoundMask ,
		const cv::Mat& _DepthMap ,
		const cv::Mat& _DepthMask , 
		const double DFarRatio , 
		const double DNearRatio ) {

			const int NWidth = _DepthMap.cols ,
				NHeight = _DepthMap.rows ;

			_DepthMask.copyTo(_BoundMask);


			cv::Mat meandval ,stdvdval ;
			cv::meanStdDev(_DepthMap , meandval , stdvdval , _DepthMask) ;

			double mindval(0) , maxdval(0);
			cv::minMaxIdx(_DepthMap , &mindval , &maxdval , NULL , NULL , _DepthMask) ;

			const double NearDval = std::max(meandval.at<double>(0)-stdvdval.at<double>(0)*DNearRatio , mindval) ;
			const double FarDval = std::min(meandval.at<double>(0)+stdvdval.at<double>(0)*DFarRatio , maxdval);


			const unsigned char* const pdmask = _DepthMask.ptr<unsigned char>(0);
			const float *const pdval = _DepthMap.ptr<float>(0) ;
			unsigned char *const pbdmask = _BoundMask.ptr<unsigned char>(0) ;

#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NWidth*NHeight ; np++) {
				if(pdmask[np]){
					if(pdval[np]>FarDval || pdval[np]<NearDval){
						pbdmask[np] = unsigned char(0) ;
					}
				}
			}


			return true;
	}



	bool
		generaOrthoArrayBoundingBox(
		std::vector<float>& x3OrthoBox ,
		const PhotoCamera & PCam , 
		const std::vector<float>& sx3P) {


			const int NPoints = sx3P.size()/3 ;

			x3OrthoBox.resize(6 , 0);
			x3OrthoBox[0] = 1e+6 , x3OrthoBox[1] = -1e+6 ;
			x3OrthoBox[2] = 1e+6 , x3OrthoBox[3] = -1e+6 ;
			x3OrthoBox[4] = 1e+6 , x3OrthoBox[5] = -1e+6 ;


			const std::vector<float>& Pvec = PCam._P_ ;
			const std::vector<float>& Kvec = PCam._K_ ;
			const std::vector<float>& Cvec = PCam._Cvec_ ;

#pragma omp parallel for schedule(dynamic, 1)
			for(int np=0 ; np<NPoints ; np++) {

				const float *const psx3 = &sx3P[np*3] ;

				const float wx = Pvec[8]*psx3[0]+Pvec[9]*psx3[1]+Pvec[10]*psx3[2]+Pvec[11] ;
				const float ux = ((Pvec[0]*psx3[0]+Pvec[1]*psx3[1]+Pvec[2]*psx3[2]+Pvec[3])/wx-Kvec[2])/Kvec[0]  ;
				const float vx = ((Pvec[4]*psx3[0]+Pvec[5]*psx3[1]+Pvec[6]*psx3[2]+Pvec[7])/wx-Kvec[5])/Kvec[4]  ;

				const float dval = std::sqrt((psx3[0]-Cvec[0])*(psx3[0]-Cvec[0])+(psx3[1]-Cvec[1])*(psx3[1]-Cvec[1])+(psx3[2]-Cvec[2])*(psx3[2]-Cvec[2])) ;

				const float rn = std::sqrt(ux*ux+vx*vx+1.0) ;

				const float rx = dval*ux/rn ;
				const float ry = dval*vx/rn ;
				const float rz = dval/rn;

#pragma omp critical
				{
					x3OrthoBox[0] = x3OrthoBox[0]<rx ? x3OrthoBox[0] : rx ;
					x3OrthoBox[1] = x3OrthoBox[1]>rx ? x3OrthoBox[1] : rx ;
					x3OrthoBox[2] = x3OrthoBox[2]<ry ? x3OrthoBox[2] : ry ;
					x3OrthoBox[3] = x3OrthoBox[3]>ry ? x3OrthoBox[3] : ry ;
					x3OrthoBox[4] = x3OrthoBox[4]<rz ? x3OrthoBox[4] : rz ;
					x3OrthoBox[5] = x3OrthoBox[5]>rz ? x3OrthoBox[5] : rz ;
				} 
			}



			return true ;
	}



	bool
		DepthFieldHelper::
		generateOrthoBoundingBoxfromCleanedSource(
		std::vector<float>& x3OrthoBox ,
		std::vector<float>& x2OrthoBox , 
		const PhotoCamera& PCam , 
		const std::vector<std::vector<float>>& vec_sx3) {

			const int NPoints = vec_sx3.size();

			x3OrthoBox.resize(6 , 0);
			x3OrthoBox[0] = 1e+6 , x3OrthoBox[1] = -1e+6 ;
			x3OrthoBox[2] = 1e+6 , x3OrthoBox[3] = -1e+6 ;
			x3OrthoBox[4] = 1e+6 , x3OrthoBox[5] = -1e+6 ;


			x2OrthoBox.resize(4 , 0);
			x2OrthoBox[0] = 1e+5 , x2OrthoBox[1] = -1e+5 ;
			x2OrthoBox[2] = 1e+5 , x2OrthoBox[3] = -1e+5 ;


			const std::vector<float>& Pvec = PCam._P_ ;
			const std::vector<float>& Kvec = PCam._K_ ;
			const std::vector<float>& Cvec = PCam._Cvec_ ;

#pragma omp parallel for schedule(dynamic, 1)
			for(int np=0 ; np<NPoints ; np++) {

				const float *const psx3 = &vec_sx3[np][0] ;

				const float wx = Pvec[8]*psx3[0]+Pvec[9]*psx3[1]+Pvec[10]*psx3[2]+Pvec[11] ;
				const float ux = (Pvec[0]*psx3[0]+Pvec[1]*psx3[1]+Pvec[2]*psx3[2]+Pvec[3])/wx ;
				const float vx = (Pvec[4]*psx3[0]+Pvec[5]*psx3[1]+Pvec[6]*psx3[2]+Pvec[7])/wx ;


				const float dval = std::sqrt((psx3[0]-Cvec[0])*(psx3[0]-Cvec[0])+(psx3[1]-Cvec[1])*(psx3[1]-Cvec[1])+(psx3[2]-Cvec[2])*(psx3[2]-Cvec[2])) ;

				float rx = (ux-Kvec[2])/Kvec[0] , ry = (vx-Kvec[5])/Kvec[4] ;

				const float rn = std::sqrt(rx*rx+ry*ry+1.0) ;

				rx = dval*rx/rn ;
				ry = dval*ry/rn ;
				float rz = dval/rn;

#pragma omp critical
				{
					x3OrthoBox[0] = x3OrthoBox[0]<rx ? x3OrthoBox[0] : rx ;
					x3OrthoBox[1] = x3OrthoBox[1]>rx ? x3OrthoBox[1] : rx ;
					x3OrthoBox[2] = x3OrthoBox[2]<ry ? x3OrthoBox[2] : ry ;
					x3OrthoBox[3] = x3OrthoBox[3]>ry ? x3OrthoBox[3] : ry ;
					x3OrthoBox[4] = x3OrthoBox[4]<rz ? x3OrthoBox[4] : rz ;
					x3OrthoBox[5] = x3OrthoBox[5]>rz ? x3OrthoBox[5] : rz ;

					x2OrthoBox[0] = x2OrthoBox[0]<ux ? x2OrthoBox[0] : ux ;
					x2OrthoBox[1] = x3OrthoBox[1]>ux ? x3OrthoBox[1] : ux ;
					x2OrthoBox[2] = x2OrthoBox[0]<vx ? x2OrthoBox[2] : vx ;
					x2OrthoBox[3] = x3OrthoBox[1]>vx ? x3OrthoBox[3] : vx ;
				} 
			}


			return true ;
	}


	bool
		DepthFieldHelper::
		filterOrthoPointsArray(
		std::vector<float>& dx3P ,
		const std::vector<float>& sx3P ,
		const std::vector<float>& x3OrthoBox ,
		const PhotoCamera& PCam ,
		const double DZNearRatio , 
		const double DZFarRatio ) {

			const int NPoints = sx3P.size()/3 ;

			std::vector<float> nx3P ;
			nx3P.reserve(NPoints*3);

			for(int np=0 ; np<NPoints ; np++){

				const float *const px3p = &sx3P[np*3] ;

				if( px3p[0]>x3OrthoBox[0] && px3p[0]<x3OrthoBox[1] &&
					px3p[1]>x3OrthoBox[2] && px3p[1]<x3OrthoBox[3] &&
					px3p[2]>DZNearRatio*x3OrthoBox[4] && px3p[2]<DZFarRatio*x3OrthoBox[5] ) {

#pragma omp critical
						{
							nx3P.push_back(px3p[0]) , nx3P.push_back(px3p[1]) , nx3P.push_back(px3p[2]) ;
						}
				}		
			}

			std::vector<float>(nx3P).swap(dx3P);


			return true ;
	}


	bool
		DepthFieldHelper::
		filterOrthoDepthMap(
		cv::Mat& nDepthMap ,
		cv::Mat& nDepthMask ,
		const PhotoCamera& PCam ,
		const cv::Mat& sDepthMap ,
		const cv::Mat& sDepthMask ,
		const std::vector<float>& x3OrthoBox ,
		const std::vector<float>& x2OrthoBox , 
		const double DZNearRatio , 
		const double DZFarRatio ) {

			const int NWidth = sDepthMap.cols , NHeight = sDepthMap.rows ;


			std::vector<float> filteredPoints2 , filteredDepth ;
			filteredPoints2.reserve(NWidth*NHeight*3) ;
			filteredDepth.reserve(NWidth*NHeight);

			const std::vector<float>& Pvec = PCam._P_ ;
			const std::vector<float>& Kvec = PCam._K_ ;
			const std::vector<float>& Cvec = PCam._Cvec_ ;		

			cv::Mat UDepthMap = cv::Mat(NHeight , NWidth , CV_32FC1);
			UDepthMap.setTo(0);
			cv::normalize(sDepthMap , UDepthMap , 0 , 255.0 , cv::NORM_MINMAX , CV_32FC1 , sDepthMask) ;
			UDepthMap.convertTo(UDepthMap , CV_8UC1);




#pragma omp parallel for schedule(dynamic , 1)
			for(int nh=0 ; nh<NHeight ; nh++) {

				const float *psdval = sDepthMap.ptr<float>(nh);
				const unsigned char *psdmask = sDepthMask.ptr<unsigned char>(nh) ,
					*pudep = UDepthMap.ptr<unsigned char>(nh);


				for(int nw=0 ; nw<NWidth ; nw++ , psdval++ , psdmask++ , pudep++) {
					if(psdmask[0]) {

						const float ux = (nw-Kvec[2])/Kvec[0] , vx = (nh-Kvec[5])/Kvec[4] ;

						const float rn = std::sqrt(ux*ux+vx*vx+1.0) ;

						const float rx = *psdval*ux/rn , ry = *psdval*vx/rn , rz = *psdval/rn ;

						if( rx>x3OrthoBox[0] && rx<x3OrthoBox[1] &&
							ry>x3OrthoBox[2] && ry<x3OrthoBox[3] &&
							rz>DZNearRatio*x3OrthoBox[4] && rz<DZFarRatio*x3OrthoBox[5] ) 

#pragma omp critical
						{
							filteredPoints2.push_back(nw) , filteredPoints2.push_back(nh) , filteredPoints2.push_back(0);

						}

					}

				}
			}



			return true ;
	}


}