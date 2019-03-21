
#ifndef DEPTH_MAPPER_HPP_
#define DEPTH_MAPPER_HPP_

#include <VTK/vtkSmartPointer.h>
#include <VTK/vtkFloatArray.h>
#include <VTK/vtkUnsignedCharArray.h>
#include <VTK/vtkIdTypeArray.h>
#include <VTK/vtkPoints.h>
#include <VTK/vtkPointData.h>
#include <VTK/vtkCellArray.h>
#include <VTK/vtkPolyData.h>
#include <VTK/vtkPolyDataMapper.h>
#include <VTK/vtkActor.h>
#include <VTK/vtkRenderer.h>
#include <VTK/vtkRenderWindow.h>
#include <VTK/vtkRenderWindowInteractor.h>
#include <VTK/vtkInteractorStyleTrackballCamera.h>
#include <VTK/vtkCamera.h>
#include <VTK/vtkWindowToImageFilter.h>
#include <VTK/vtkImageExport.h>
#include <VTK/vtkStructuredGrid.h>

#include <VTK/vtkWindowedSincPolyDataFilter.h>


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include <vector>
#include <algorithm>
#include <utility>
#include <string>


#include "../../SceneConfiguration.h"

#include "TrimeshGenerator.h"


#ifdef _VSP_
#undef _VSP_
#endif
#define _VSP_(ClassName_ , ObjName_)\
	vtkSmartPointer<ClassName_> ObjName_ = vtkSmartPointer<ClassName_>::New() ;


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




namespace stereoscene{


	struct DepthMapper{

	public:

		DepthMapper(
			std::vector<float>& _x2hPoints ,
			std::vector<cind>& _TriCells ,
			std::vector<float>& _x2Depth)
			: _x2hPoints_(_x2hPoints) , 
			_TriCells_(_TriCells) ,
			_x2Depth_(_x2Depth) {}


		bool
			operator() (
			cv::Mat& _DepthMap ,
			cv::Mat& _DepthMask ,
			std::vector<float>& _DepthBox ,
			const PhotoCamera& PCam ,
			const cv::Size& MapSize ,
			const bool BGetDepthBox , 
			const int NMinD=3 , 
			const int NMaxD=255) {


				///step.1: generate 2d-triangular mesh
				///
				const bool BInverse = false ;
				const int NPointsDim = 3 ;

				TrimeshGenerator(_x2hPoints_ , _TriCells_ ) (BInverse , NPointsDim) ;


				///step.2: map the depth field
				///
				if(BGetDepthBox) {

					std::pair<std::vector<float>::iterator , std::vector<float>::iterator> pa_iterMinMax = std::minmax_element(_x2Depth_.begin() , _x2Depth_.end()) ;

					_DepthBox.resize(2, 0);
					_DepthBox[0] = *pa_iterMinMax.first , _DepthBox[1] = *pa_iterMinMax.second ;
				}

				const float &&FRangeD = _DepthBox[1]-_DepthBox[0] ;
			
				const int NPoints = _x2hPoints_.size()/NPointsDim ;



				///pre-normalize the depth into [NMinD , NMaxD] so that binary mask can be built convenient by using thresholding
				const int NRange = NMaxD-NMinD ;

				///only used for uchar field data, like images loaded from folders
				std::vector<unsigned char> _x2UCharDepth(NPoints);

#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0 ; np<NPoints ; np++) {
					_x2UCharDepth[np] = unsigned char( (NMinD+NRange*(_x2Depth_[np]-_DepthBox[0])/FRangeD) < NMaxD ? (1+NRange*(_x2Depth_[np]-_DepthBox[0])/FRangeD) : NMaxD ) ;
				}


				shadeImageDepthMap(
					_DepthMap , 
					_x2hPoints_ , 
					_TriCells_ ,
					_x2UCharDepth ,
					MapSize ,
					NMinD ,
					NMaxD ,
					true );


				if(NMinD){

					///build the mask for depth map and remapping depth map as specific data type ,
					///for using this, NMinD must be setting bigger than 0
					cv::threshold(_DepthMap , _DepthMask , NMinD-2 , 255 , cv::THRESH_BINARY);

				} else {

					_DepthMask.create(MapSize , CV_8UC1);
					memset(_DepthMask.ptr<unsigned char>(0) , 255 , sizeof(unsigned char)*MapSize.width*MapSize.height);

#pragma omp parallel for schedule(static ,1)
					for(int nr=0 ; nr<_BORDER_NGAP_ ; nr++) {
						_DepthMask.row(nr).setTo(0) ;
						_DepthMask.row(MapSize.height-1-nr).setTo(0);
						_DepthMask.col(nr).setTo(0);
						_DepthMask.col(MapSize.width-1-nr).setTo(0);
					}
				}


				_DepthMap.setTo(NMaxD , 255-_DepthMask) ;
				_DepthMap.convertTo(_DepthMap , CV_32FC1) ; ///should be transformed into float type

				//cv::normalize(_DepthMap , _DepthMap , _DepthBox[0] , _DepthBox[1] , cv::NORM_MINMAX , CV_32FC1 , _DepthMask);
				//_DepthMap.setTo(_DepthBox[1] , 255-_DepthMask);

				cv::normalize(_DepthMap , _DepthMap , 0 , 1.0 , cv::NORM_MINMAX);

				//_SEE_IMAGE_(_DepthMap , "depth map");

				return true ;
		}



	private:

		std::vector<float>& _x2hPoints_ ;

		std::vector<cind>& _TriCells_ ;

		std::vector<float>& _x2Depth_ ;



	public:

		static bool
			shadeImageDepthMap(
			cv::Mat& _DepthMap ,
			std::vector<float>& _x3Points ,
			std::vector<cind>& _x3TriCells ,
			std::vector<unsigned char>& _x3UcharDepth , 
			const cv::Size& MapSize ,
			const int NMinD ,
			const int NMaxD ,
			const bool BOffLine = true) {


				const int NPoints = _x3Points.size()/3 ,
					NTriCells = _x3TriCells.size()/4 ;


				_VSP_(vtkPolyData , Polys_) ;


				///Points
				_VSP_(vtkFloatArray , FPointsArray);
				FPointsArray->SetNumberOfComponents(3);
				FPointsArray->SetArray(_x3Points.data() , _x3Points.size() , 1);

				_VSP_(vtkPoints , Points_) ;
				Points_->SetData(FPointsArray);
				Polys_->SetPoints(Points_) ;


				///Depth
				_VSP_(vtkUnsignedCharArray, Depth_) ;					
				Depth_->SetNumberOfComponents(1);
				Depth_->SetArray(_x3UcharDepth.data() , _x3UcharDepth.size() , 1);			
				Depth_->SetName("Depth");

				Polys_->GetPointData()->SetScalars(Depth_);


				///TriCells
				_VSP_(vtkIdTypeArray , IIndArray) ;
				IIndArray->SetArray(_x3TriCells.data() , _x3TriCells.size() , 1);

				_VSP_(vtkCellArray , TriCells_) ;
				TriCells_->SetCells(NTriCells , IIndArray);
				Polys_->SetPolys(TriCells_);


				_VSP_(vtkPolyDataMapper , Mapper_) ;
				Mapper_->SetInputData(Polys_);
				Mapper_->Update();


				_VSP_(vtkActor , Actor_);
				Actor_->SetMapper(Mapper_);


				_VSP_(vtkRenderer ,Ren_);
				Ren_->AddActor(Actor_);
				Ren_->SetBackground(0.0, 0.0, 0.0);

				vtkCamera *p_Camera = Ren_->GetActiveCamera();
				p_Camera->SetParallelProjection(1);
				p_Camera->SetPosition(MapSize.width/2 , MapSize.height/2 , p_Camera->GetDistance()+0);
				p_Camera->SetFocalPoint(MapSize.width/2 , MapSize.height/2 , 0);
				p_Camera->SetParallelScale(MapSize.height/2);


				int NWinW = (MapSize.width) <1280 ? (MapSize.width) : (MapSize.width/2) , 
					NWinH = (MapSize.height) <1024 ? (MapSize.height) : (MapSize.height/2) ;


				_VSP_(vtkRenderWindow , RenWin_);
				RenWin_->AddRenderer(Ren_);
				RenWin_->SetPosition(0 , 0);
				RenWin_->SetSize(NWinW , NWinH) ;

				if(BOffLine){
					RenWin_->OffScreenRenderingOn() ;
				}

				RenWin_->Render();

				if(!BOffLine) {
					RenWin_->SetWindowName("Depth Map") ;

					_VSP_(vtkInteractorStyleTrackballCamera , Tball);
					_VSP_(vtkRenderWindowInteractor , IRen);
					IRen->SetInteractorStyle(Tball) ;
					IRen->SetRenderWindow(RenWin_);
					IRen->Initialize() ;
					IRen->Start() ;
				}


				_VSP_(vtkWindowToImageFilter , Win2Image_) ;
				Win2Image_->SetInput(RenWin_);
				Win2Image_->SetMagnification(1);
				Win2Image_->SetInputBufferTypeToRGBA() ;
				Win2Image_->Update();


				cv::Mat WinImage(NWinH , NWinW , CV_8UC4) ;
				_VSP_(vtkImageExport , ImageExporter_);
				ImageExporter_->SetInputConnection(Win2Image_->GetOutputPort());
				ImageExporter_->ImageLowerLeftOn() ; ///graphic coordinates 
				ImageExporter_->Update() ;
				ImageExporter_->Export(WinImage.ptr<unsigned char>(0));


				cv::cvtColor(WinImage , _DepthMap , cv::COLOR_RGBA2GRAY);
				if(NWinW != MapSize.width || NWinH != MapSize.height) {
					cv::resize(_DepthMap , _DepthMap , MapSize);
				}



				return 0 ;
		}



		static bool
			shadePolygonDepthMap(
			cv::Mat& _DepthMap ,
			std::vector<float>& _x3Points ,
			std::vector<cind>& _x3TriCells ,
			const cv::Size& MapSize ,
			const bool BOffLine = true) {


				const int NPoints = _x3Points.size()/3 ,
					NTriCells = _x3TriCells.size()/4 ;


				_VSP_(vtkPolyData , Polys_) ;


				///Points
				_VSP_(vtkFloatArray , FPointsArray);
				FPointsArray->SetNumberOfComponents(3);
				FPointsArray->SetArray(_x3Points.data() , _x3Points.size() , 1);

				_VSP_(vtkPoints , Points_) ;
				Points_->SetData(FPointsArray);
				Polys_->SetPoints(Points_) ;



				///TriCells
				_VSP_(vtkIdTypeArray , IIndArray) ;
				IIndArray->SetArray(_x3TriCells.data() , _x3TriCells.size() , 1);

				_VSP_(vtkCellArray , TriCells_) ;
				TriCells_->SetCells(NTriCells , IIndArray);
				Polys_->SetPolys(TriCells_);


				_VSP_(vtkPolyDataMapper , Mapper_) ;
				Mapper_->SetInputData(Polys_);
				Mapper_->Update();


				_VSP_(vtkActor , Actor_);
				Actor_->SetMapper(Mapper_);
				Actor_->SetOrigin(0 , 0 , 0);
				Actor_->RotateX(180);
				
				const double *pcen = Actor_->GetCenter() ;
				double pbox[6]={0} ;
				Actor_->GetBounds(pbox) ;

				_VSP_(vtkRenderer ,Ren_);
				Ren_->AddActor(Actor_);


				vtkCamera *p_Camera = Ren_->GetActiveCamera();
				p_Camera->SetParallelProjection(1);
				p_Camera->SetPosition(pcen[0] , pcen[1] , 1);
				p_Camera->SetFocalPoint(pcen[0] , pcen[1] , 0);
				p_Camera->SetParallelScale((pbox[3]-pbox[2])/2);


				int NWinW = (MapSize.width) <1280 ? (MapSize.width) : (MapSize.width/2) , 
					NWinH = (MapSize.height) <1024 ? (MapSize.height) : (MapSize.height/2) ;


				_VSP_(vtkRenderWindow , RenWin_);
				RenWin_->AddRenderer(Ren_);
				RenWin_->SetPosition(0 , 0);
				RenWin_->SetSize(NWinW , NWinH) ;
				//RenWin_->SetPolygonSmoothing(1);


				if(BOffLine){
					RenWin_->OffScreenRenderingOn() ;
				}

				RenWin_->Render();


				_VSP_(vtkWindowToImageFilter , Win2Image_) ;
				Win2Image_->SetInput(RenWin_);
				Win2Image_->SetMagnification(1);
				Win2Image_->SetInputBufferTypeToZBuffer() ;
				Win2Image_->Update();


				cv::Mat WinImage(NWinH , NWinW , CV_32FC1) ;
				_VSP_(vtkImageExport , ImageExporter_);
				ImageExporter_->SetInputConnection(Win2Image_->GetOutputPort());
				ImageExporter_->ImageLowerLeftOff() ; ///graphic coordinates 
				ImageExporter_->Update() ;
				ImageExporter_->Export(WinImage.ptr<float>(0));


				//cv::cvtColor(WinImage , _DepthMap , cv::COLOR_RGBA2GRAY);
				if(NWinW != MapSize.width || NWinH != MapSize.height) {
					cv::resize(WinImage , _DepthMap , MapSize);
				}
				else {
					WinImage.convertTo(_DepthMap , CV_32FC1);
				}


				if(!BOffLine) {
					RenWin_->SetWindowName("Depth Map") ;

					_VSP_(vtkInteractorStyleTrackballCamera , Tball);
					_VSP_(vtkRenderWindowInteractor , IRen);
					IRen->SetInteractorStyle(Tball) ;
					IRen->SetRenderWindow(RenWin_);
					IRen->Initialize() ;
					IRen->Start() ;
				}

				return 0 ;
		}


	};




}


#endif