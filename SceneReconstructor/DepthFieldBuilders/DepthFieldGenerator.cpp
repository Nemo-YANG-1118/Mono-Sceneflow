
#include <iostream>
#include <iterator>
#include <list>
#include <cmath>
#include <algorithm>

#include <omp.h>

#include "../SceneConfiguration.h"
#include "DepthFieldGenerator.h"
#include "DepthFieldHelper.h"


#ifdef _DEBUG_SEE_CLEANER_COMPARED_
#undef _DEBUG_SEE_CLEANER_COMPARED_
#endif
#define _DEBUG_SEE_CLEANER_COMPARED_ 0


#ifdef _DEBUG_SEE_DEPTH_MAP_
#undef _DEBUG_SEE_DEPTH_MAP_
#endif
#define _DEBUG_SEE_DEPTH_MAP_ 0


namespace stereoscene{


	bool 
		DepthFieldGenerator::
		operator()(
		const bool BCleaning  ,
		const int KthRefPCam ,
		const cv::Size& MapSize ,
		const bool BGenerateFullMap ,
		const int KScene,
		const bool BSaveCleanPoints ,
		const std::string& _sdir_Save ) {


			if(KScene) {
				fflush(stdout) ;
				fprintf(stdout , "\n DepthFieldGenerator: About to generate the depth field of scene.%d\n" , KScene) ;
				fflush(stdout) ;
			}


			std::map<int ,PhotoCamera>::const_iterator it_PCam=_map_PCams_.find(KthRefPCam);

			if(BCleaning) {

				///step.1: clean the original points cloud
				///
				fflush(stdout) ;
				fprintf(stdout , " --> step.1 : cleaning the points cloud by removing ") ;
				fflush(stdout) ;



				orthoTriangulatePoints(it_PCam->second);


#if _DEBUG_SEE_CLEANER_COMPARED_
				DepthFieldHelper::seePoints(_vec_x3Points_ , true  , std::string("Before Clean the Points")) ;
#endif


				const int NCleaned = cleanPointsCloud() ;


				fflush(stdout) ;
				fprintf(stdout , "( %d ) outliers\n" , NCleaned) ;
				fflush(stdout) ;

#if _DEBUG_SEE_CLEANER_COMPARED_

				DepthFieldHelper::seePoints(_vec_x3Points_ , true , std::string("After Clean the Points")) ;
#endif


				unorthoTriangulatePoints(it_PCam->second);

				if(BSaveCleanPoints) {

					char chname[256] = {0};
					if(KScene<10) {
						sprintf(chname , "%s/00%d\0" , _sdir_Save.c_str() , KScene);
					} else if(KScene<100&&KScene>=10) {
						sprintf(chname , "%s/0%d\0" , _sdir_Save.c_str() , KScene);
					} else if(KScene<1000&&KScene>=100) {
						sprintf(chname , "%s/%d\0" , _sdir_Save.c_str() , KScene);
					}


					std::string sdir_Save(chname) ;

					DepthFieldHelper::savePoints(_vec_x3Points_ , sdir_Save ,std::string("CleanPoints.txt"));
				}
			}




			///step.2: generate the depth map of the scene
			///
			fflush(stdout) ;
			fprintf(stdout , " --> step.2 : generating the depth field by depth mapping with Points Cloud\n") ;
			fflush(stdout) ;

			generateDepthField(KthRefPCam , BGenerateFullMap , MapSize) ;



			return true ;
	}


}



#include "DepthFieldCommon/PointsCleaner.hpp"


namespace stereoscene{


	int
		DepthFieldGenerator::
		cleanPointsCloud() {


			const int NOrgPoints = _vec_x3Points_.size();


			/// Parameters Setting
			///
			const int NSearchNeighbors = 10 ;


			///live distance = [2.0 , 3.0]
			///global distance= [6.0  , 6.5]
			const double DThreshPlaneDistance = 7.0 , 
				DThreshStdVar = 1.0 ;



			PointsCleaner<float>(true , NSearchNeighbors , DThreshPlaneDistance , DThreshStdVar) (
				_vec_x3Points_ ,
				_vec_x3Points_);



			return (NOrgPoints - _vec_x3Points_.size()) ;
	}


	bool
		DepthFieldGenerator::
		orthoTriangulatePoints(
		const PhotoCamera& PCam) {

			const int NPoints = _vec_x3Points_.size() ;

			const std::vector<float> &Kvec = PCam._K_ ;
			const std::vector<float> &Pvec = PCam._P_ ;
			const std::vector<float> &Cvec = PCam._Cvec_ ;

#pragma omp parallel for schedule(dynamic, 1)
			for(int np3=0; np3<NPoints ; np3++){

				float *const px3 = &_vec_x3Points_[np3][0] ;

				const float wx = Pvec[8]*px3[0]+Pvec[9]*px3[1]+Pvec[10]*px3[2]+Pvec[11];
				const float ux = ((Pvec[0]*px3[0]+Pvec[1]*px3[1]+Pvec[2]*px3[2]+Pvec[3])/wx - Kvec[2])/Kvec[0] ;
				const float vx = ((Pvec[4]*px3[0]+Pvec[5]*px3[1]+Pvec[6]*px3[2]+Pvec[7])/wx - Kvec[5])/Kvec[4] ;

				const float dval = std::sqrt((px3[0]-Cvec[0])*(px3[0]-Cvec[0])+(px3[1]-Cvec[1])*(px3[1]-Cvec[1])+(px3[2]-Cvec[2])*(px3[2]-Cvec[2]));

				const float rn = std::sqrt(ux*ux+vx*vx+1.0);


				px3[0]= dval*ux/rn ;
				px3[1]= dval*vx/rn ;
				px3[2]= dval/rn ;
			}


			return true ;
	}


	bool
		DepthFieldGenerator::
		unorthoTriangulatePoints(
		const PhotoCamera& PCam) {

			const int NPoints = _vec_x3Points_.size() ;

			const std::vector<float> &Pvec = PCam._P_ ;
			const std::vector<float> &Cvec = PCam._Cvec_ ;
			const std::vector<float> &Rvec = PCam._R_ ;

#pragma omp parallel for schedule(dynamic, 1)
			for(int np3=0; np3<NPoints ; np3++){

				float *const px3 = &_vec_x3Points_[np3][0] ;

				const float dval = std::sqrt(px3[0]*px3[0]+px3[1]*px3[1]+px3[2]*px3[2]);

				px3[0]/=dval , px3[1]/=dval , px3[2]/=dval ;

				const float rx = px3[0]*Rvec[0]+px3[1]*Rvec[3]+px3[2]*Rvec[6] ;
				const float ry = px3[0]*Rvec[1]+px3[1]*Rvec[4]+px3[2]*Rvec[7] ;
				const float rz = px3[0]*Rvec[2]+px3[1]*Rvec[5]+px3[2]*Rvec[8] ;

				const float rn = std::sqrt(rx*rx+ry*ry+rz*rz);


				px3[0]= dval*rx/rn + Cvec[0] ;
				px3[1]= dval*ry/rn + Cvec[1] ;
				px3[2]= dval*rz/rn + Cvec[2] ;
			}


			return true;
	}

}



#include <Fade2D/Fade_2D.h>
#include <stlplus3/filesystemSimplified/file_system.hpp>

#include "DepthFieldCommon/DepthMapper.hpp"


namespace stereoscene{



	bool
		DepthFieldGenerator::
		orthoBackprojection(
		const cv::Mat& _DepthMap , 
		const cv::Mat& _DepthMask , 
		const cv::Mat& _TexImage , 
		const PhotoCamera& CurPCam ,
		const cv::Size& MapSize ,
		const int KthCurPCam , 
		const bool BVisualized ,
		const bool BUseTexture , 
		const bool BSaveMeshPhoto,
		const std::string& _sdir ,
		const std::string& _sname_Save) {


			const int NWidth = MapSize.width , 
				NHeight = MapSize.height ,
				NPoints = NWidth*NHeight ;


			const int NWMeshBorder = 9 + _BORDER_NGAP_ ,
				NHMeshBorder = std::floor(float(NWMeshBorder*NWidth)/NHeight) ;

			cv::Mat FDepthMap ;
			cv::normalize(_DepthMap , FDepthMap , 0.0 , 1.0 , cv::NORM_MINMAX);

			///step.1: back projection
			///
			std::vector<float> x3Points ;
			x3Points.reserve(NPoints*3);
			//std::map<float , std::pair<float , float>> map_zx3Points ;


			std::vector<unsigned char> x3Tex;
			x3Tex.reserve(NPoints*3);


			const std::vector<float>& CurK = CurPCam._K_ ;


#pragma omp parallel for schedule(dynamic, 1)
			for(int nh=NHMeshBorder; nh<NHeight-NHMeshBorder; nh++) {

				const unsigned char* pMask = _DepthMask.ptr<unsigned char>(nh)+NWMeshBorder;
				const float *pDepth = _DepthMap.ptr<float>(nh)+NWMeshBorder;

				if(BUseTexture) {

					const cv::Vec3b* pbgrTex = _TexImage.ptr<cv::Vec3b>(nh)+NWMeshBorder;

					for(int nw=NWMeshBorder; nw<NWidth-NWMeshBorder; nw++ , pMask++ , pDepth++, pbgrTex++) {

						if(*pMask) {

							const float &&x2u = (nw-CurK[2])/CurK[0] , &&x2v = (nh-CurK[5])/CurK[4] ;

							const float &&rayn = std::sqrt(x2u*x2u+x2v*x2v+1.0);

#pragma omp critical
							{
								//map_zx3Points.insert(std::make_pair(*pDepth/rayn , std::make_pair(*pDepth*x2u/rayn , *pDepth*x2v/rayn)));
								x3Points.push_back(*pDepth*x2u/rayn) , x3Points.push_back(*pDepth*x2v/rayn) , x3Points.push_back(*pDepth/rayn) ;

								x3Tex.push_back(pbgrTex->val[2]) , x3Tex.push_back(pbgrTex->val[1]) , x3Tex.push_back(pbgrTex->val[0]);
							}
						}
					}
				} else {

					for(int nw=NWMeshBorder; nw<NWidth-NWMeshBorder; nw++ , pMask++ , pDepth++) {

						if(*pMask) {

							const float &&x2u = (nw-CurK[2])/CurK[0] , &&x2v = (nh-CurK[5])/CurK[4] ;

							const float &&rayn = std::sqrt(x2u*x2u+x2v*x2v+1.0);

#pragma omp critical
							{
								//map_zx3Points.insert(std::make_pair(*pDepth/rayn , std::make_pair(*pDepth*x2u/rayn , *pDepth*x2v/rayn)));
								x3Points.push_back(*pDepth*x2u/rayn) , x3Points.push_back(*pDepth*x2v/rayn) , x3Points.push_back(*pDepth/rayn) ;
							}
						}
					}
				}

			}
			std::vector<float> (x3Points).swap(x3Points);
			std::vector<unsigned char> (x3Tex).swap(x3Tex);



			///step.2: generate trimesh
			///
			std::vector<cind> x3TCells ;
			std::vector<float> x3Normals ;

			TrimeshGenerator::triangulateDelaunayMesh(x3TCells , x3Points, 3 , false, x3Normals , true);


			///step.3: saving and visualization
			///
			char chSave[256] = {0};
			if(KthCurPCam<10){
				sprintf(chSave , "%s/00%d" , _sdir.c_str() ,KthCurPCam );
			} else if(KthCurPCam>=10 && KthCurPCam<100) {
				sprintf(chSave , "%s/0%d" ,_sdir.c_str(), KthCurPCam);
			} else if(KthCurPCam>=100 && KthCurPCam<1000) {
				sprintf(chSave , "%s/%d" , _sdir.c_str() , KthCurPCam);
			}

			std::string sdir_Save(chSave) ; 



			if(!BUseTexture) {

				DepthFieldHelper::seeTrimesh(x3Points.data() , x3Points.size()/3 ,
					x3TCells.data() , x3TCells.size()/4 , 
					x3Normals.data() , true , stlplus::basename_part(_sname_Save) , 3.0,
					BSaveMeshPhoto ,sdir_Save , _sname_Save);

			} else {

				DepthFieldHelper::seeTrimeshTextured(x3Points.data() , x3Points.size()/3 ,
					x3TCells.data() , x3TCells.size()/4 , 
					x3Tex.data() , 3 , 
					x3Normals.data(), true , stlplus::basename_part(_sname_Save) , 3.0 ,
					BSaveMeshPhoto , sdir_Save , _sname_Save) ;
			}



			return true ;
	}



	bool
		DepthFieldGenerator::
		calculateDepthBox(
		std::vector<float>& DepthBox ,
		const cv::Size& MapSize ,
		const PhotoCamera& PCam ) {


			const int NPoints3 = _vec_x3Points_.size() ;


			const std::vector<float>& Pvec = PCam._P_ ,
				&Kvec = PCam._K_ ,
				&Cvec = PCam._Cvec_ ;


			///step.1: orthogonal triangulation
			///
			std::vector<float> x3P(NPoints3*3 , 0 ) , x3D(NPoints3 , 0);

#pragma omp parallel for schedule(dynamic, 1)
			for(int np3=0 ; np3<NPoints3 ; np3++) {
				const std::vector<float>& x3Points = _vec_x3Points_[np3] ;


				const float dval = std::sqrt( (x3Points[0]-Cvec[0])*(x3Points[0]-Cvec[0]) + 
					(x3Points[1]-Cvec[1])*(x3Points[1]-Cvec[1]) +
					(x3Points[2]-Cvec[2])*(x3Points[2]-Cvec[2]) ) ;

				x3D[np3] = dval ;

				const float wx = Pvec[8]*x3Points[0]+Pvec[9]*x3Points[1]+Pvec[10]*x3Points[2]+Pvec[11] ;
				float rx = ( (Pvec[0]*x3Points[0]+Pvec[1]*x3Points[1]+Pvec[2]*x3Points[2]+Pvec[3])/wx - Kvec[2] )/Kvec[0] ;
				float ry = ( (Pvec[4]*x3Points[0]+Pvec[5]*x3Points[1]+Pvec[6]*x3Points[2]+Pvec[7])/wx - Kvec[5] )/Kvec[4] ;

				const float rn = std::sqrt(rx*rx+ry*ry+1.0);

				float *const px3p = &x3P[np3*3] ;
				px3p[0] = dval*rx/rn ;
				px3p[1] = dval*ry/rn ;

				//x3D[np3] = px3p[2] = dval/rn ;

			}


			///step.2: shade polygon-depth map to estimate occlusion
			///
			std::vector<cind> x3C ;
			TrimeshGenerator::triangulateDelaunayMesh(x3C , x3P , 3);

			cv::Mat OrthoDepthMap , OrthoDepthMask ;
			DepthMapper::shadePolygonDepthMap(OrthoDepthMap , x3P , x3C , MapSize , true );

			cv::threshold(OrthoDepthMap , OrthoDepthMask , 0.998 , 255 , cv::THRESH_BINARY_INV);
			OrthoDepthMask.convertTo(OrthoDepthMask , CV_8UC1) ;

			double dmax_val(0) , dmin_val(0);
			cv::minMaxIdx(OrthoDepthMap , &dmin_val , &dmax_val , NULL , NULL  , OrthoDepthMask);
		
			const float depthmax = *std::max_element(x3D.begin() , x3D.end());
			const float depthmin = *std::min_element(x3D.begin() , x3D.end());

			const float dratio = depthmin/dmin_val ;

			_DepthBox_.resize(2 , 0);
			_DepthBox_[0] = depthmin ;
			_DepthBox_[1] = std::min(float(dmax_val*dratio) , depthmax ) ;
			_DepthBox_[1] = depthmax ;

			std::cout<<depthmin<<"\t"<<depthmax<<"\n";
			std::cout<<_DepthBox_[0]<<"\t"<<_DepthBox_[1]<<"\n\n" ;

			return true ;
	}



	bool 
		DepthFieldGenerator::
		generateDepthField(
		const int KthRefPCam ,
		const bool BGenerateFullMap ,
		const cv::Size& MapSize) {


			///depth map should be generated on reference-camera's plane
			std::map<int , PhotoCamera>::const_iterator it_RefPCam = _map_PCams_.find(KthRefPCam) ;
			if(it_RefPCam==_map_PCams_.end()) {

				fflush(stderr) ;
				fprintf(stderr , "\nError in <DepthFieldGenerator/quadrangulateDepthField>: ref cam doesn't exist\n") ;
				fflush(stderr) ;

				exit(0);
			} 

			const PhotoCamera& RefPCam = it_RefPCam->second ;


			const int NCams = _map_PCams_.size() ,
				NPad = _PER_BUNDLE_NFRAMES_/2 ,
				KRef = it_RefPCam->first ;


			///step.1: reproject all points onto reference-image plane to remove over-border points 
			///		and calculate inliers' depth
			///
			const float NWidthBorder = MapSize.width - _BORDER_NGAP_ , NHeightBorder = MapSize.height - _BORDER_NGAP_ ;
			const int NPoints = _vec_x3Points_.size() ;

			std::vector<float> vec_x2hPoints ;
			vec_x2hPoints.reserve(NPoints*3);

			std::vector<float> vec_x2Depth ;
			vec_x2Depth.reserve(NPoints) ;


			const std::vector<float> &P_ = RefPCam._P_ , 
				&Cvec_ = RefPCam._Cvec_ ;


#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints ; np++) {

				const std::vector<float>& x3_ = _vec_x3Points_[np] ;

				const float &&wx2 = P_[8]*x3_[0] + P_[9]*x3_[1] + P_[10]*x3_[2]+P_[11] ;


				///setting homo-plane as (x, y ,0)
				const float &&x2u = (P_[0]*x3_[0] + P_[1]*x3_[1] + P_[2]*x3_[2]+P_[3])/wx2 ,
					&&x2v = (P_[4]*x3_[0] + P_[5]*x3_[1] + P_[6]*x3_[2]+P_[7])/wx2 ;


				if(x2u>_BORDER_NGAP_ && x2u< NWidthBorder && x2v>_BORDER_NGAP_ && x2v<NHeightBorder) {

					const float &&depval = std::sqrt( (x3_[0]-Cvec_[0])*(x3_[0]-Cvec_[0]) + 
						(x3_[1]-Cvec_[1])*(x3_[1]-Cvec_[1]) + 
						(x3_[2]-Cvec_[2])*(x3_[2]-Cvec_[2])) ;


#pragma omp critical
					{
						vec_x2Depth.push_back(depval) , 
							vec_x2hPoints.push_back(x2u), vec_x2hPoints.push_back(x2v) , vec_x2hPoints.push_back(0) ;
					}
				}
			}


			const float FMaxDepth = *std::max_element(vec_x2Depth.begin() , vec_x2Depth.end() ) ;


			if(BGenerateFullMap) {

				///add 4 corners
				vec_x2hPoints.push_back(_BORDER_NGAP_) , vec_x2hPoints.push_back(_BORDER_NGAP_) , vec_x2hPoints.push_back(0) , vec_x2Depth.push_back(FMaxDepth);
				vec_x2hPoints.push_back(NWidthBorder) , vec_x2hPoints.push_back(_BORDER_NGAP_) , vec_x2hPoints.push_back(0) , vec_x2Depth.push_back(FMaxDepth);
				vec_x2hPoints.push_back(_BORDER_NGAP_) , vec_x2hPoints.push_back(NHeightBorder) , vec_x2hPoints.push_back(0) , vec_x2Depth.push_back(FMaxDepth);
				vec_x2hPoints.push_back(NWidthBorder) , vec_x2hPoints.push_back(NHeightBorder) , vec_x2hPoints.push_back(0) , vec_x2Depth.push_back(FMaxDepth);
			}



			///step.2: generate the depth field
			///
			std::vector<float> (vec_x2hPoints).swap(vec_x2hPoints) ;
			std::vector<float> (vec_x2Depth).swap(vec_x2Depth) ;


			DepthMapper(vec_x2hPoints , _vec_TriCells_ , vec_x2Depth ) (
				_DepthMap_ , _DepthMask_ , _DepthBox_ ,  RefPCam , MapSize , true , 3 , 255);


			if(BGenerateFullMap) {
				_DepthMask_.setTo(unsigned char(255));

				_DepthMask_.colRange(0 , _BORDER_NGAP_).setTo((unsigned char)0);
				_DepthMask_.colRange(MapSize.width-_BORDER_NGAP_ , MapSize.width).setTo((unsigned char)0);
				_DepthMask_.rowRange(0 , _BORDER_NGAP_).setTo((unsigned char)0);
				_DepthMask_.rowRange(MapSize.height-_BORDER_NGAP_ , MapSize.height).setTo((unsigned char)0);
			}

			_DepthMask_.convertTo(_DepthMask_ , CV_8UC1);

#if _DEBUG_SEE_DEPTH_MAP_ 			
			_SEE_IMAGE_(_DepthMap_ , "Depth Map");
#endif

			return true;
	}



	bool
		DepthFieldGenerator::
		generateMapTrimesh(
		std::vector<int>& _TriCells ,
		const cv::Size& _MapSize ,
		const cv::Mat& _MapMask) {

			const int &NWidth = _MapSize.width , 
				&NHeight = _MapSize.height ;



			std::vector<GEOM_FADE2D::Point2> vec_Points ;
			vec_Points.reserve(NWidth*NHeight);


			int x2Cnt(0);
			std::for_each(
				_MapMask.ptr<unsigned char>(0),
				_MapMask.ptr<unsigned char>(0)+NWidth*NHeight , 
				[&x2Cnt , &NWidth , &NHeight , &vec_Points](
				const unsigned char& _mark){
					if(_mark) {
						vec_Points.push_back(GEOM_FADE2D::Point2(x2Cnt%NWidth , x2Cnt/NWidth)) , 
							(vec_Points.end()-1)->setCustomIndex(vec_Points.size()-1);
					}
					x2Cnt++ ;
			});

			std::vector<GEOM_FADE2D::Point2> (vec_Points).swap(vec_Points) ;
			std::vector<GEOM_FADE2D::Point2*> vec_PointsHandle ;

			GEOM_FADE2D::Fade_2D DT2 ;
			DT2.insert(vec_Points, vec_PointsHandle);

			std::vector<GEOM_FADE2D::Triangle2*> vec_pTriangles ;
			DT2.getTrianglePointers(vec_pTriangles);

			//std::cout<<vec_PointsHandle.size()<<std::endl ;

			const int NTriCells = vec_pTriangles.size() ;
			_TriCells.resize(NTriCells*4);

#pragma omp parallel for schedule(dynamic , 1)
			for(int nt=0 ; nt<NTriCells ; nt++) {
				int *const pxTriCells = &_TriCells[nt*4];
				pxTriCells[0] = 3 ,
					pxTriCells[1] = vec_pTriangles[nt]->getCorner(0)->getCustomIndex() ,
					pxTriCells[2] = vec_pTriangles[nt]->getCorner(1)->getCustomIndex() ,
					pxTriCells[3] = vec_pTriangles[nt]->getCorner(2)->getCustomIndex() ;
			}


			return true ;
	}



	bool
		DepthFieldGenerator::
		generateVerticesTrimesh(
		std::vector<cind> &_x3TriCells ,
		std::vector<float> &_x3Normals , 
		const std::vector<float>& _x3Points ,
		const bool BComputeNormals,
		const bool BInverse) {


			TrimeshGenerator::triangulateDelaunayMesh(_x3TriCells , _x3Points , 3 , BInverse , _x3Normals , true);



			return true ;
	}


}