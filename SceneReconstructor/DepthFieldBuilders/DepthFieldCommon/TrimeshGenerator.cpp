
#include <Fade2D/Fade_2D.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <utility>
#include <iterator>
#include <numeric>


#include <omp.h>

#include "TrimeshGenerator.h"


#undef sqrt


#include <cmath>


namespace stereoscene{


	static bool
		generateConvexHullConstrainedEdges(
		std::vector<GEOM_FADE2D::Segment2>& _vec_hullEdges,
		const std::vector<GEOM_FADE2D::Point2>& _vec_Points) {


		const int NPoints = _vec_Points.size() ;

		std::vector<cv::Point2f> vec_x2Points(NPoints) ;

#pragma omp parallel for schedule(dynamic , 1)
		for(int np=0 ; np<NPoints ; np++) {
			vec_x2Points[np].x = _vec_Points[np].x(),vec_x2Points[np].y = _vec_Points[np].y() ;
		}



		///step.2: calculate convex hull to generate a refined zone
		///
		std::vector<int> hullInd ;
		cv::convexHull(vec_x2Points,hullInd,false,false);

		const int NHullEdges = hullInd.size() ;
		std::vector<GEOM_FADE2D::Segment2> vec_hullEdges ;
		_vec_hullEdges.reserve(NHullEdges) ;

		for(std::vector<int>::const_iterator it_x0Ind=hullInd.begin(),it_x1Ind = it_x0Ind+1 ;
			it_x1Ind!=hullInd.end() ; ++it_x0Ind,++it_x1Ind) {
			_vec_hullEdges.push_back(GEOM_FADE2D::Segment2(_vec_Points[*it_x0Ind],_vec_Points[*it_x1Ind])) ;
		}

		_vec_hullEdges.push_back(GEOM_FADE2D::Segment2(_vec_Points[hullInd[NHullEdges-1]],_vec_Points[hullInd[0]])) ;


		return true ;
	}



	static bool
		calculatePointsNormals(
		std::vector<float>& _x3Normals ,
		const std::vector<float>& _x3Points , 
		const std::vector<GEOM_FADE2D::Point2*>& _vec_PointsHandle) {

			const int NPoints = _vec_PointsHandle.size() ;

			_x3Normals.resize(NPoints*3 , 0);

#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints ; np++) {

				const GEOM_FADE2D::Triangle2 *const pTriBegin =  _vec_PointsHandle[np]->getIncidentTriangle() ,
					*pTri = pTriBegin;

				float *const px3Normals = &_x3Normals[np*3] ;

				float &vtnx = px3Normals[0] , &vtny = px3Normals[1] , &vtnz = px3Normals[2] ;

				int triCnt(0);
				do{
					const int &&xInd0 = pTri->getCorner(0)->getCustomIndex()*3 ,
						&&xInd1 = pTri->getCorner(1)->getCustomIndex()*3 ,
						&&xInd2 = pTri->getCorner(2)->getCustomIndex()*3 ;


					const float *const px0 = &_x3Points[xInd0] , 
						*const px1 = &_x3Points[xInd1] ,
						*const px2 = &_x3Points[xInd2] ;


					///tri-cells' normal: tn := (v1-v0)x(v2-v0)/||(v1-v0)x(v2-v0)||
					///		cross product:  x ¡Á y := { x[1]*y[2]-x[2]*y[1] , x[2]y[0]-x[0]y[2] , x[0]*y[1]-x[1]*y[0] }
					///
					const float edge10[3] = {px1[0]-px0[0] , px1[1]-px0[1] , px1[2]-px0[2]} ,
						edge20[3] = {px2[0]-px0[0] , px2[1]-px0[1] , px2[2]-px0[2]} ;

					const float &&tnx = edge10[1]*edge20[2]-edge10[2]*edge20[1] , 
						&&tny = edge10[2]*edge20[0]-edge10[0]*edge20[2] ,
						&&tnz = edge10[0]*edge20[1]-edge10[1]*edge20[0] ,
						&&tnn = std::sqrt(tnx*tnx + tny*tny + tnz*tnz) ;


					vtnx += tnx/tnn , vtny += tny/tnn , vtnz += tnz/tnn ;			

					triCnt++ ;

				}while(pTri!=pTriBegin) ;

				vtnx/=triCnt , vtny/=triCnt , vtnz/=triCnt ;

				const float &&vtnn = std::sqrt(vtnx*vtnx + vtny*vtny + vtnz*vtnz);

				vtnx /= vtnn  ,vtny /= vtnn , vtnz /= vtnn ;

			}


			return true ;
	}



	static bool
		calculatePointsNormals(
		std::vector<float>& _x3Normals ,
		const std::vector<const float*>& _x3Points , 
		const std::vector<GEOM_FADE2D::Point2*>& _vec_PointsHandle) {

			const int NPoints = _vec_PointsHandle.size() ;

			_x3Normals.resize(NPoints*3 , 0);

#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints ; np++) {

				const GEOM_FADE2D::Triangle2 *const pTriBegin =  _vec_PointsHandle[np]->getIncidentTriangle() ,
					*pTri = pTriBegin;

				float *const px3Normals = &_x3Normals[np*3] ;

				float &vtnx = px3Normals[0] , &vtny = px3Normals[1] , &vtnz = px3Normals[2] ;

				int triCnt(0);
				do{
					const int &&xInd0 = pTri->getCorner(0)->getCustomIndex()*3 ,
						&&xInd1 = pTri->getCorner(1)->getCustomIndex()*3 ,
						&&xInd2 = pTri->getCorner(2)->getCustomIndex()*3 ;


					const float *const px0 = _x3Points[xInd0] , 
						*const px1 = _x3Points[xInd1] ,
						*const px2 = _x3Points[xInd2] ;


					///tri-cells' normal: tn := (v1-v0)x(v2-v0)/||(v1-v0)x(v2-v0)||
					///		cross product:  x ¡Á y := { x[1]*y[2]-x[2]*y[1] , x[2]y[0]-x[0]y[2] , x[0]*y[1]-x[1]*y[0] }
					///
					const float edge10[3] = {px1[0]-px0[0] , px1[1]-px0[1] , px1[2]-px0[2]} ,
						edge20[3] = {px2[0]-px0[0] , px2[1]-px0[1] , px2[2]-px0[2]} ;

					const float &&tnx = edge10[1]*edge20[2]-edge10[2]*edge20[1] , 
						&&tny = edge10[2]*edge20[0]-edge10[0]*edge20[2] ,
						&&tnz = edge10[0]*edge20[1]-edge10[1]*edge20[0] ,
						&&tnn = static_cast<float>(std::sqrt(tnx*tnx + tny*tny + tnz*tnz)) ;

					vtnx += tnx/tnn , vtny += tny/tnn , vtnz += tnz/tnn ;			

					triCnt++ ;

				}while(pTri!=pTriBegin) ;

				vtnx/=triCnt , vtny/=triCnt , vtnz/=triCnt ;

				const float &&vtnn = std::sqrt(vtnx*vtnx + vtny*vtny + vtnz*vtnz);

				vtnx/=vtnn , vtny/=vtnn , vtnz/=vtnn ;
			}



			return true ;
	}



	bool 
		TrimeshGenerator::
		triangulateDelaunayMesh(
		std::vector<cind>& _xTriCells ,
		const std::vector<float>& _xPoints, 
		const int Nxdim , 
		const bool BInverse , 
		std::vector<float>& _x3Normals ,
		const bool BComputePointsNormals ) {

			const int NPoints = _xPoints.size()/Nxdim ;

			std::vector<GEOM_FADE2D::Point2> vec_Points(NPoints) ;

			if(BInverse) {
#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0 ; np<NPoints ; np++) {
					vec_Points[np].change(_xPoints[np*Nxdim] , -_xPoints[np*Nxdim+1]) ,
						vec_Points[np].setCustomIndex(np);
				}
			} else{
#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0 ; np<NPoints ; np++) {
					vec_Points[np].change(_xPoints[np*Nxdim] , _xPoints[np*Nxdim+1]) ,
						vec_Points[np].setCustomIndex(np);
				}
			}

			GEOM_FADE2D::Fade_2D DT2 ;

			std::vector<GEOM_FADE2D::Point2*> vec_PointsHandle ; ///can access associated triangles

			if(!BComputePointsNormals){
				DT2.insert(vec_Points);
			}else if(BComputePointsNormals && Nxdim==3) {
				DT2.insert(vec_Points , vec_PointsHandle);
			} else {
				fflush(stdout);
				fprintf(stdout , "\nError: the dimension of source points must be 3\n");
				fflush(stdout);

				exit(0);
			}


			std::vector<GEOM_FADE2D::Triangle2 *> vec_pTriangles ;
			DT2.getTrianglePointers(vec_pTriangles);

			int NTriCells = vec_pTriangles.size() ;

			NTriCells = vec_pTriangles.size();

			_xTriCells.resize(NTriCells*4);

#pragma omp parallel for schedule(dynamic , 1)
			for(int nt=0 ;nt<NTriCells ; nt++) {

				cind *const pxTriCells = &_xTriCells[nt*4] ;
				const GEOM_FADE2D::Triangle2 *const pTriangle = vec_pTriangles[nt] ;

				pxTriCells[0] = 3 ,
					pxTriCells[1] = pTriangle->getCorner(0)->getCustomIndex() ,
					pxTriCells[2] = pTriangle->getCorner(1)->getCustomIndex() ,
					pxTriCells[3] = pTriangle->getCorner(2)->getCustomIndex() ;
			}



			if(BComputePointsNormals && Nxdim==3) {

				calculatePointsNormals(_x3Normals ,_xPoints , vec_PointsHandle) ;
			}


			return true ;
	}



	bool
		TrimeshGenerator::
		triangulateDelaunayMesh(
		std::vector<cind>& _xTriCells ,
		const std::vector<const float*>& _xPoints, 
		const int Nxdim , 
		const bool BInverse ,
		std::vector<float>& _x3Normals ,
		const bool BComputePointsNormals) {

			const int NPoints = _xPoints.size()/Nxdim ;

			std::vector<GEOM_FADE2D::Point2> vec_Points(NPoints) ;

#pragma omp parallel for schedule(dynamic , 1)
			for(int np=0 ; np<NPoints ; np++) {
				vec_Points[np].change(*_xPoints[np*Nxdim] , *_xPoints[np*Nxdim+1]) ,
					vec_Points[np].setCustomIndex(np);
			}


			GEOM_FADE2D::Fade_2D DT2 ;

			std::vector<GEOM_FADE2D::Point2*> vec_PointsHandle ; ///can access associated triangles

			if(!BComputePointsNormals){
				DT2.insert(vec_Points);
			}else if(BComputePointsNormals && Nxdim==3) {
				DT2.insert(vec_Points , vec_PointsHandle);
			} else {
				fflush(stdout);
				fprintf(stdout , "\nError: the dimension of source points must be 3\n");
				fflush(stdout);

				exit(0);
			}


			std::vector<GEOM_FADE2D::Triangle2 *> vec_pTriangles ;
			DT2.getTrianglePointers(vec_pTriangles);

			const int NTriCells = vec_pTriangles.size() ;
			_xTriCells.resize(NTriCells*4);


#pragma omp parallel for schedule(dynamic , 1)
			for(int nt=0 ;nt<NTriCells ; nt++) {

				cind *const pxTriCells = &_xTriCells[nt*4] ;
				const GEOM_FADE2D::Triangle2 *const pTriangle = vec_pTriangles[nt] ;

				pxTriCells[0] = 3 ,
					pxTriCells[1] = cind(pTriangle->getCorner(0)->getCustomIndex()) ,
					pxTriCells[2] = cind(pTriangle->getCorner(1)->getCustomIndex()) ,
					pxTriCells[3] = cind(pTriangle->getCorner(2)->getCustomIndex()) ;
			}



			if(BComputePointsNormals && Nxdim==3) {

				calculatePointsNormals(_x3Normals , _xPoints , vec_PointsHandle) ;
			}


			return true ;
	}



	bool
		TrimeshGenerator::
		generateConstrainedTrimesh(
		const int Nxdim ,
		const bool BInverse ,
		const bool BConstrained) {


			const int NPoints = _x2Points_.size()/Nxdim ;

			std::vector<GEOM_FADE2D::Point2> vec_Points(NPoints , GEOM_FADE2D::Point2());


			if(!BInverse)  {
#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0 ; np<NPoints ; np++) {
					vec_Points[np].change(_x2Points_[np*Nxdim] , _x2Points_[np*Nxdim+1]) , 
						vec_Points[np].setCustomIndex(np);
				}
			} else {
#pragma omp parallel for schedule(dynamic , 1)
				for(int np=0 ; np<NPoints ; np++) {
					vec_Points[np].change(_x2Points_[np*Nxdim] , -_x2Points_[np*Nxdim+1]) , 
						vec_Points[np].setCustomIndex(np);
				}
			}


			GEOM_FADE2D::Fade_2D DT2(NPoints) ;
			DT2.insert(vec_Points);

			if(BConstrained) {

				std::vector<GEOM_FADE2D::Segment2> vec_hullEdges ;
				generateConvexHullConstrainedEdges(vec_hullEdges , vec_Points) ;


				GEOM_FADE2D::ConstraintGraph2* pConstraint = 
					DT2.createConstraint(vec_hullEdges , GEOM_FADE2D::CIS_IGNORE_DELAUNAY);

				GEOM_FADE2D::Zone2 *pRZone = DT2.createZone(pConstraint , GEOM_FADE2D::ZL_INSIDE) ;
				DT2.applyConstraintsAndZones();

			}


			///generate triangular mesh
			std::vector<GEOM_FADE2D::Triangle2*> vec_pTriangles ;
			DT2.getTrianglePointers(vec_pTriangles) ;



			const int NTriCells = vec_pTriangles.size() ;
			_x2TriCells_.resize(NTriCells*4) ;

#pragma omp parallel for schedule(dynamic ,1)
			for(int nt=0 ; nt<NTriCells ; nt++) {

				cind *const pTriCell = &_x2TriCells_[nt*4] ;
				const GEOM_FADE2D::Triangle2* const pTriangle = vec_pTriangles[nt];

				pTriCell[0] = 3 ,
					pTriCell[1] = cind(pTriangle->getCorner(0)->getCustomIndex()) ,
					pTriCell[2] = cind(pTriangle->getCorner(1)->getCustomIndex()) ,
					pTriCell[3] = cind(pTriangle->getCorner(2)->getCustomIndex()) ;
			}


			return true ;
	}


}