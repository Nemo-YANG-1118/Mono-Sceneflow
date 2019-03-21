#include <stlplus3/filesystemSimplified/file_system.hpp>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/surface/poisson.h>
#include <pcl/surface/concave_hull.h>

#include <VTK/vtkSmartPointer.h>
#include <VTK/vtkPoints.h>
#include <VTK/vtkPolyData.h>
#include <VTK/vtkWindowedSincPolyDataFilter.h>
#include <VTK/vtkCleanPolyData.h>
#include <VTK/vtkDecimatePro.h>
#include <VTK/vtkCellArray.h>
#include <VTK/vtkPointData.h>
#include <VTK/vtkPolyDataMapper.h>
#include <VTK/vtkIdList.h>
#include <VTK/vtkWindowToImageFilter.h>
#include <VTK/vtkUnsignedCharArray.h>
#include <VTK/vtkImageExport.h>

#include <VTK/vtkPLYWriter.h>
#include <VTK/vtkPLYReader.h>
#include <VTK/vtkIdTypeArray.h>
//#include <VTK/vtkIdType.h>
#include <VTK/vtkIdList.h>



#include <VTK/vtkSmartPointer.h>
#include <VTK/vtkFloatArray.h>
#include <VTK/vtkPointData.h>
#include <VTK/vtkPolyData.h>
#include <VTK/vtkCellArray.h>
#include <VTK/vtkIdTypeArray.h>
#include <VTK/vtkPolyDataMapper.h>
#include <VTK/vtkActor.h>
#include <VTK/vtkRenderer.h>
#include <VTK/vtkRenderWindow.h>
#include <VTK/vtkRenderWindowInteractor.h>
#include <VTK/vtkInteractorStyleTrackballCamera.h>
#include <VTK/vtkVertexGlyphFilter.h>
#include <VTK/vtkWindowToImageFilter.h>
#include <VTK/vtkImageExport.h>
#include <VTK/vtkCamera.h>
#include <VTK/vtkTIFFWriter.h>
#include <VTK/vtkProperty.h>
#include <VTK/vtkDelaunay2D.h>
#include <VTK/vtkButterflySubdivisionFilter.h>
#include <VTK/vtkWindowedSincPolyDataFilter.h>
#include <VTK/vtkCleanPolyData.h>
#include <VTK/vtkPolyDataNormals.h>



#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <fstream>
#include <iterator>
#include <utility>

#include <iostream>
#include <algorithm>


#include <omp.h>


#include "PoissonDistanceFieldReconstructor.h"

#include "TrimeshGenerator.h"
#include "DepthMapper.hpp"


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


#undef _VSP_
#define _VSP_(TypeName_ , VarName_);\
	vtkSmartPointer<TypeName_> VarName_ = vtkSmartPointer<TypeName_>::New();\

#undef _PSPCLD_
#define _PSPCLD_(TypeName_ , PVarName_);\
	pcl::PointCloud<TypeName_>::Ptr PVarName_(new pcl::PointCloud<TypeName_> ) ;\


#undef _DISP_VEC_
#define _DISP_VEC_(vecname_begin , vecname_end , tname)\
	std::copy(vecname_begin , vecname_end, (std::ostream_iterator<tname>(std::cout , "\t"))) ;\
	std::cout<<"\n" ;\


#undef _DEBUG_SEE_AFTER_PREPROCESS_
#define _DEBUG_SEE_AFTER_PREPROCESS_ 1


namespace stereoscene{


	static bool
		renderPoisson(
		cv::Mat& DepthMap ,
		const bool BOffline ,
		vtkSmartPointer<vtkPolyData>& Polys , 
		const cv::Size& MapSize ) ;


	bool 
		PoissonDistanceFieldReconstructor::
		reconstructPoisson(
		cv::Mat& DepthMap ,
		const std::vector<float>& sx3Points ,
		const cv::Mat& DepthMask , 
		const std::string& sname_save) {


			const bool BInverse = false ;

			const cv::Size MapSize = DepthMask.size();

			const int NPoints3 = sx3Points.size()/3 ;

			cv::Mat SrcPoints3 = cv::Mat(NPoints3 , 3 , CV_32FC1) ;
			memcpy(SrcPoints3.ptr<float>(0) , &sx3Points[0] , sizeof(float)*3*NPoints3);

			double MinMax3[6]={0};
			cv::minMaxIdx(SrcPoints3.col(0) , &MinMax3[0] , &MinMax3[1] , NULL , NULL );
			cv::minMaxIdx((BInverse?(-1):(1))*SrcPoints3.col(1) , &MinMax3[2] , &MinMax3[3] , NULL , NULL );
			cv::minMaxIdx((BInverse?(-1):(1))*SrcPoints3.col(2) , &MinMax3[4] , &MinMax3[5] , NULL , NULL );

			const double Bound3[3] = {\
				(MinMax3[1]-MinMax3[0])/2 ,
				(MinMax3[3]-MinMax3[2])/2 ,
				(MinMax3[5]-MinMax3[4])/2 };



			//_DISP_VEC_(MinMax3 , MinMax3+6 , double);


			_PSPCLD_(pcl::PointXYZ , x3Cld) ;
			x3Cld->resize(NPoints3);

			if(BInverse) {

				const float *const psx3 = &sx3Points[0];

#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0 ; np<NPoints3 ; np++) {
					x3Cld->points[np].data[0] = psx3[np*3] ;
					x3Cld->points[np].data[1] = -psx3[np*3+1] ;
					x3Cld->points[np].data[2] = -psx3[np*3+2] ;
				}
			}
			else {

				const float *const psx3 = &sx3Points[0];

#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0 ; np<NPoints3 ; np++) {
					memcpy(&x3Cld->points[np].data[0] , &psx3[np*3], sizeof(float)*3) ;
				}
			}


			///step.1: calculate normals of points cloud
			///
			pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree (new pcl::search::KdTree<pcl::PointXYZ> );


			_PSPCLD_(pcl::Normal , n3Cld) ;

			pcl::NormalEstimationOMP<pcl::PointXYZ , pcl::Normal> NormalEst ;
			NormalEst.setInputCloud(x3Cld);
			NormalEst.setSearchMethod(kdtree) ;
			NormalEst.setKSearch(12); //6 as a ring
			NormalEst.compute(*n3Cld);


			///concatenate n3cld and x3cld 
			_PSPCLD_(pcl::PointNormal , x3nCld) ;
			x3nCld->resize(NPoints3);

#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints3 ; np++) {
				memcpy(&x3nCld->points[np].data[0] , &x3Cld->points[np].data[0] , sizeof(float)*3) ;
				x3nCld->points[np].normal[0] = -n3Cld->points[np].normal[0];
				x3nCld->points[np].normal[1] = -n3Cld->points[np].normal[1];
				x3nCld->points[np].normal[2] = -n3Cld->points[np].normal[2];		
			}




			///step.2: reconstruct poisson surface
			///
			const double DScale = 1.06 ;

			pcl::PolygonMesh PolyMesh ;

			pcl::Poisson<pcl::PointNormal> PoissonRec ;
			PoissonRec.setDepth(9) ;
			PoissonRec.setManifold(1);
			PoissonRec.setSamplesPerNode(2.0);
			PoissonRec.setScale(DScale);
			PoissonRec.setInputCloud(x3nCld) ;
			PoissonRec.reconstruct(PolyMesh);



			_VSP_(vtkPolyData , Polys);
			pcl::io::mesh2vtk(PolyMesh , Polys);



			cv::Mat ZBuff = cv::Mat::zeros(MapSize , CV_32FC1 ) ,
				DMAP = cv::Mat::zeros(MapSize , CV_32FC1) ;


			const bool BOffline = true ;


			renderPoisson(ZBuff , BOffline , Polys , MapSize ) ;


			cv::normalize(ZBuff , DMAP , 0 , 1.0 , cv::NORM_MINMAX , CV_32FC1);

			if(DepthMap.empty()) {
				DepthMap.create(MapSize , CV_32FC1);
			}


			cv::bilateralFilter(DMAP , DepthMap , 5 , 2.0, 20);

			//DepthMap.setTo(0 , 255-DepthMask);


			if(!sname_save.empty()) {

				_VSP_(vtkPLYWriter , Ply);
				Ply->SetInputData(Polys);
				Ply->SetFileName(sname_save.c_str()) ;
				Ply->Update();
			}


			return true ;
	}



	bool
		PoissonDistanceFieldReconstructor::
		convertDepthMapToDistanceField(
		std::vector<float>& dx3Points ,
		const cv::Mat& sDepthMap , 
		const cv::Mat& sDepthMask) {

			const int NWidth=sDepthMap.cols , NHeight = sDepthMap.rows;

			dx3Points.reserve(NWidth*NHeight*3);

			cv::Mat dDepthMap ;
			cv::bilateralFilter(sDepthMap , dDepthMap ,  5 , 2.0 , 20);

			cv::normalize(dDepthMap , dDepthMap, 0 , 255.0 ,  cv::NORM_MINMAX , CV_32FC1 , sDepthMask);
			dDepthMap.setTo(255.0 , 255-sDepthMask);


			const int  wcen = (NWidth)/2 ,
				hcen = (NHeight)/2 ;

			for(int nh=0 ; nh<NHeight ; nh++) {

				const unsigned char* pmask=sDepthMask.ptr<unsigned char>(nh);

				const float *pdval = dDepthMap.ptr<float>(nh) ;

				for(int nw=0 ; nw<NWidth ; nw++ , pmask++ , pdval++){
					if(*pmask){
						dx3Points.push_back( nw ) ;
						dx3Points.push_back( NHeight-nh ) ;
						dx3Points.push_back( 255-*pdval ) ;
					}
				}
			}



			return true ;
	}


	bool
		PoissonDistanceFieldReconstructor::
		orthoTriangulateDepthMap(
		std::vector<float>& dx3Points ,
		const PhotoCamera& PCam ,
		const cv::Mat& DepthMap ,
		const cv::Mat& DepthMask ) {


			const int NWidth = DepthMap.cols ,
				NHeight = DepthMap.rows ;


			dx3Points.reserve(NWidth*NHeight*3);


			const std::vector<float> &Kvec = PCam._K_;

#pragma omp parallel for schedule(dynamic , 1)
			for(int nh=0; nh<NHeight ; nh++) {

				const float* const pdval = DepthMap.ptr<float>(nh) ;
				const unsigned char* const pdmask = DepthMask.ptr<unsigned char>(nh) ;

				for(int nw=0;  nw<NWidth ; nw++ ) {
					if(pdmask[nw]) {

						const float ux = (nw-Kvec[2])/Kvec[0] , vx = (nh-Kvec[5])/Kvec[4] ;

						const float rn = std::sqrt(ux*ux+vx*vx+1.0) ;


						dx3Points.push_back(pdval[nw]*ux/rn) ;
						dx3Points.push_back(pdval[nw]*vx/rn) ;
						dx3Points.push_back(pdval[nw]/rn) ;
					}
				}
			}

			std::vector<float>(dx3Points).swap(dx3Points) ;


			return true ;
	}


	bool
		renderPoisson(
		cv::Mat& DepthMap ,
		const bool BOffline ,
		vtkSmartPointer<vtkPolyData>& Polys , 
		const cv::Size& MapSize ) {


			_VSP_(vtkPolyData, Polys1) ;
			Polys1->ShallowCopy(Polys);

			const int NPoints = Polys->GetNumberOfPoints();


			//std::vector<unsigned char> x3Tex(NPoints*3);

			//#pragma omp parallel for
			//		for(int np=0 ; np<NPoints ; np++){
			//			double pt3[3] ;
			//			Polys->GetPoint(np , pt3);
			//
			//			x3Tex[np*3] = (255-pt3[2]);
			//			x3Tex[np*3+1] = x3Tex[np*3] ;
			//			x3Tex[np*3+2]=x3Tex[np*3+1];
			//		}
			//


			//_VSP_(vtkUnsignedCharArray  , UTex) ;
			//UTex->SetNumberOfComponents(3) ;
			//UTex->SetArray(x3Tex.data() , x3Tex.size() , 1);

			//Polys1->GetPointData()->SetScalars(UTex);


			fprintf(stdout , "\n\t\trendering poisson...\n") ;



			_VSP_(vtkPolyDataMapper , PMapper2);
			PMapper2->SetInputData(Polys1);
			PMapper2->Update() ;


			_VSP_(vtkActor , PActor) ;
			PActor->SetMapper(PMapper2);



			_VSP_(vtkRenderer , PRen);
			PRen->AddActor(PActor);
			PActor->SetPosition(0 , 0 , 0);
			PActor->SetOrigin(0 , 0 , 0);
			PActor->SetScale(1.0/MapSize.width , 1.0/MapSize.width , 1.0/MapSize.width);
			//


			double *pbox = PActor->GetBounds();
			double *pcen = PActor->GetCenter() ;

			vtkCamera *pCamera = PRen->GetActiveCamera() ;
			pCamera->SetParallelProjection(1);
			pCamera->SetPosition(pcen[0] , pcen[1] , 1) ;
			pCamera->SetFocalPoint(pcen[0] , pcen[1] , 0) ;
			pCamera->SetParallelScale( (MapSize.height)/(2.0*MapSize.width)) ;
			//pCamera->SetParallelScale(300) ;
			//pCamera->SetClippingRange(0.01, 500.0)


			_VSP_(vtkRenderWindow ,PRenWin);
			PRenWin->AddRenderer(PRen);
			PRenWin->SetSize(MapSize.width , MapSize.height);
			PRenWin->SetPosition(0 , 0);
			PRenWin->SetPolygonSmoothing(1) ;
			PRenWin->SetLineSmoothing(1);

			if(BOffline) {
				PRenWin->OffScreenRenderingOn();
			}

			PRenWin->Render() ;


			_VSP_(vtkWindowToImageFilter , WinImage);
			WinImage->SetInput(PRenWin);
			WinImage->SetInputBufferTypeToZBuffer();
			WinImage->SetMagnification(1.0);
			WinImage->Update() ;


			cv::Mat WinMap = cv::Mat(MapSize , CV_32FC1) ;
			_VSP_(vtkImageExport , ImageExporter);
			ImageExporter->SetInputConnection(WinImage->GetOutputPort());
			ImageExporter->ImageLowerLeftOff() ; ///graphic coordinates 
			ImageExporter->SetExportVoidPointer(WinMap.ptr<float>(0));
			ImageExporter->Update() ;
			ImageExporter->Export(WinMap.ptr<float>(0));

			if(WinMap.cols != MapSize.width || WinMap.rows!=MapSize.height) {
				cv::resize(WinMap , DepthMap , MapSize);
			}
			else {
				WinMap.convertTo(DepthMap , CV_32FC1);
			}


			if(!BOffline){

				_VSP_(vtkInteractorStyleTrackballCamera , Tball);
				_VSP_(vtkRenderWindowInteractor , IRen);
				IRen->SetInteractorStyle(Tball) ;
				IRen->SetRenderWindow(PRenWin);
				IRen->Initialize() ;
				IRen->Start() ;

			}


			return true ;
	} ;


}




#undef _SEE_POLYMAPPER_
#define _SEE_POLYMAPPER_(PMapper)\
	_VSP_(vtkActor , PActor);\
	PActor->SetMapper(PMapper);\
	_VSP_(vtkRenderer , PRen);\
	PRen->AddActor(PActor);\
	_VSP_(vtkRenderWindow , PRenWin);\
	PRenWin->AddRenderer(PRen);\
	PRenWin->SetSize(800 , 600);\
	PRenWin->Render();\
	_VSP_(vtkInteractorStyleTrackballCamera , TBall);\
	_VSP_(vtkRenderWindowInteractor , IRen);\
	IRen->SetInteractorStyle(TBall);\
	IRen->SetRenderWindow(PRenWin);\
	IRen->Initialize();\
	IRen->Start() ;\



namespace stereoscene{



	bool
		PoissonDistanceFieldReconstructor::
		preprocessTriMesh(
		std::vector<float>& dx3P,
		const bool BInverse ,
		const std::vector<float>& sx3P ) {


			const int NPoints3 = sx3P.size()/3 ;

			_VSP_(vtkPoints , Points3);
			Points3->SetNumberOfPoints(NPoints3);
			Points3->SetDataTypeToFloat() ;

			if(BInverse) {
#pragma omp parallel for schedule(dynamic,1)
				for(int np3=0 ; np3<NPoints3 ; np3++) {
					const float *const psx3 = &sx3P[np3*3];
					Points3->SetPoint(np3 , psx3[0] , -psx3[1] , -psx3[2]);
				}
			}
			else {
				memcpy(Points3->GetVoidPointer(0), &sx3P[0] , sizeof(float)*3*NPoints3);
			}


			_VSP_(vtkPolyData , Polys);
			Polys->SetPoints(Points3);


			double pcen[3]={0};
			Polys->GetCenter(pcen);


			_VSP_(vtkDelaunay2D , dt) ;
			dt->SetInputData(Polys);
			dt->Update();


			_VSP_(vtkWindowedSincPolyDataFilter ,PSmoother);
			PSmoother->SetInputConnection(dt->GetOutputPort()) ;
			PSmoother->SetNumberOfIterations(20);
			PSmoother->SetBoundarySmoothing(1);
			PSmoother->SetNormalizeCoordinates(1);
			PSmoother->Update();


			_VSP_(vtkPolyDataNormals , PNorm);
			PNorm->SetInputConnection(PSmoother->GetOutputPort());
			PNorm->AutoOrientNormalsOn() ;
			PNorm->SetSplitting(1);
			PNorm->SetConsistency(0);	
			PNorm->Update();


			//_VSP_(vtkDecimatePro, PDec);
			//PDec->SetInputConnection(PNorm->GetOutputPort());
			//PDec->SetTargetReduction(0.50) ; //delete 80%
			//PDec->Update();

			/*		_VSP_(vtkButterflySubdivisionFilter , PSub) ;
			PSub->SetInputConnection(PDec->GetOutputPort());
			PSub->SetNumberOfSubdivisions(1);
			PSub->Update();*/

			_VSP_(vtkPolyDataMapper , PMapper);
			PMapper->SetInputConnection(PNorm->GetOutputPort());
			PMapper->Update();


			_VSP_(vtkPoints , NewPoints3);
			NewPoints3->ShallowCopy(PMapper->GetInput()->GetPoints()) ;
			NewPoints3->SetDataTypeToFloat();

			const int NNewPoints3 = NewPoints3->GetNumberOfPoints();

			dx3P.resize(NNewPoints3*3);

			if(BInverse) {
#pragma omp parallel for schedule(dynamic,1)
				for(int np3=0 ; np3<NPoints3 ; np3++) {					
					double pt3[3] = {0};
					NewPoints3->GetPoint(np3 , pt3);

					float * pdx3 = &dx3P[np3*3] ;
					pdx3[0] = pt3[0] , pdx3[1] = -pt3[1], pdx3[2] = -pt3[2];
				}
			}
			else {
				memcpy(&dx3P[0] , NewPoints3->GetVoidPointer(0) , sizeof(float)*3*NNewPoints3);
			}

#if 1
			if(_DEBUG_SEE_AFTER_PREPROCESS_){

				//double pcen[3]={0};
				double pbox[6]={0};
				Polys->GetBounds(pbox);

				_VSP_(vtkActor , PActor);
				PActor->SetMapper(PMapper);

				if(BInverse){
					//PActor->SetOrigin(pcen[0], pcen[1] , pcen[2]);
					//PActor->RotateX(180);
				}


				_VSP_(vtkRenderer , PRen);
				PRen->AddActor(PActor);

				_VSP_(vtkCamera ,pCamera);
				pCamera = PRen->GetActiveCamera();
				pCamera->SetParallelProjection(1);
				pCamera->SetPosition(0 , 0 , pCamera->GetDistance());
				pCamera->SetFocalPoint(0, 0 ,0);
				pCamera->SetParallelScale((pbox[3]-pbox[2])/2);


				_VSP_(vtkRenderWindow , PRenWin);
				PRenWin->AddRenderer(PRen);
				PRenWin->SetSize(800 , 600);
				PRenWin->Render();


				_VSP_(vtkInteractorStyleTrackballCamera , TBall);
				_VSP_(vtkRenderWindowInteractor , IRen);
				IRen->SetInteractorStyle(TBall);
				IRen->SetRenderWindow(PRenWin);
				IRen->Initialize();
				IRen->Start() ;

			}
#endif

			return true ;
	}


}