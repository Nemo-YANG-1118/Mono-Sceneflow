
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/algorithm.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>

#include <iostream>


#include <omp.h>

#include "ConcaveHullExtractor.h"


typedef CGAL::Exact_predicates_inexact_constructions_kernel CG_Kernel ;

typedef CG_Kernel::FT CG_FLOAT ;
typedef CG_Kernel::Point_2  CG_Point2d  ;
typedef CG_Kernel::Segment_2  CG_Segment2d ;

typedef CGAL::Alpha_shape_vertex_base_2<CG_Kernel> Vertb2;
typedef CGAL::Alpha_shape_face_base_2<CG_Kernel>  Faceb2;
typedef CGAL::Triangulation_data_structure_2<Vertb2,Faceb2> Trids2;
typedef CGAL::Delaunay_triangulation_2<CG_Kernel,Trids2> DTriangulation_2;

typedef CGAL::Alpha_shape_2<DTriangulation_2>  Alpha_shape_2;
typedef Alpha_shape_2::Alpha_shape_edges_iterator Alpha_shape_edges_iterator;



namespace stereoscene{

	template <class OutputIterator>
	void
		alpha_edges( const Alpha_shape_2&  A,
		OutputIterator out) {

			//A.find_optimal_alpha(1);

			for(Alpha_shape_edges_iterator it =  A.alpha_shape_edges_begin();
				it != A.alpha_shape_edges_end(); ++it){
					*out++ = A.segment(*it);
			}
	}


	static bool
		findAlphaEdges(
		std::vector<cv::Point>& ccHull ,
		std::vector<CG_Point2d>& vec_Points2) {

			const int NPoints2=vec_Points2.size() ;

			Alpha_shape_2 AS2(vec_Points2.begin(), vec_Points2.end(),
				CG_FLOAT(NPoints2/100),
				Alpha_shape_2::GENERAL);


			std::vector<CG_Segment2d> vec_segments;
			vec_segments.reserve(NPoints2/10);

			
			alpha_edges( AS2, std::back_inserter(vec_segments));
			//AS2.find_optimal_alpha(1);
			//AS2.set_al
			
			std::vector<CG_Segment2d>(vec_segments).swap(vec_segments);


			//concaveHull.push_back(cv::Point(vec_segments[0].point(0).x() , vec_segments[0].point(0).y() ));

			ccHull.clear();
			ccHull.reserve(vec_segments.size()*2);

			std::for_each(vec_segments.begin() , vec_segments.end() , 
				[&ccHull](const CG_Segment2d& _seg){
					ccHull.push_back(cv::Point(_seg.point(0).x() , _seg.point(0).y() ));
					ccHull.push_back(cv::Point(_seg.point(1).x() , _seg.point(1).y() ));
			});
			
			std::vector<cv::Point>(ccHull).swap(ccHull);

			//std::cout<<"concave "<<concaveHull.size()<<std::endl;

			return true ;
	}


	bool
		ConcaveHullExtractor::
		extractConcaveHullfromPointsVector3(
		std::vector<cv::Point>& ccHull ,
		const cv::Size& HullSize ,
		const std::vector<float>& Pvec , 
		const std::vector<std::vector<float> >& vec_x3P) {

			const int NPoints = vec_x3P.size();
			std::vector<CG_Point2d> vec_Points2;
			vec_Points2.reserve(NPoints);


#pragma omp parallel for schedule(dynamic,1)
			for(int np=0 ; np<NPoints ; np++) {

				const float *px3p = &vec_x3P[np][0] ;

				const float wx = Pvec[8]*px3p[0]+Pvec[9]*px3p[1]+Pvec[10]*px3p[2]+Pvec[11] ;
				const float ux = (Pvec[0]*px3p[0]+Pvec[1]*px3p[1]+Pvec[2]*px3p[2]+Pvec[3])/wx ;
				const float vx = (Pvec[4]*px3p[0]+Pvec[5]*px3p[1]+Pvec[6]*px3p[2]+Pvec[7])/wx ;

				if(ux<HullSize.width-2 && vx<HullSize.height-2 && ux>1 && vx>1){
#pragma omp critical
					{
						vec_Points2.push_back(CG_Point2d((int)cvRound(ux) , (int)cvRound(vx))) ;
					}
				}
			}


			findAlphaEdges(ccHull , vec_Points2) ;



			return true ;
	}


	bool
		ConcaveHullExtractor::
		extractConcaveHullfromPointsArray3(
		std::vector<cv::Point>& ccHull ,
		const cv::Size& HullSize ,
		const std::vector<float>& Pvec , 
		const std::vector<float>& x3P) {


			const int NPoints = x3P.size()/3 ;
			std::vector<CG_Point2d> vec_Points2;
			vec_Points2.reserve(NPoints);


#pragma omp parallel for schedule(dynamic,1)
			for(int np=0 ; np<NPoints ; np++) {

				const float *px3p = &x3P[np*3] ;

				const float wx = Pvec[8]*px3p[0]+Pvec[9]*px3p[1]+Pvec[10]*px3p[2]+Pvec[11] ;
				const float ux = (Pvec[0]*px3p[0]+Pvec[1]*px3p[1]+Pvec[2]*px3p[2]+Pvec[3])/wx ;
				const float vx = (Pvec[4]*px3p[0]+Pvec[5]*px3p[1]+Pvec[6]*px3p[2]+Pvec[7])/wx ;

				if(ux<HullSize.width-2 && vx<HullSize.height-2 && ux>1 && vx>1){
#pragma omp critical
					{
						vec_Points2[np]=CG_Point2d((int)cvRound(ux) , (int)cvRound(vx));
					}
				}
			}


			findAlphaEdges(ccHull , vec_Points2) ;
			

			return 0;
	}


	bool
		ConcaveHullExtractor::
		extractConcaveHullfromPointsArray2(
		std::vector<cv::Point>& ccHull ,
		const std::vector<int>& x2P) {


			const int NPoints = x2P.size()/2 ;
			std::vector<CG_Point2d> vec_Points2(NPoints);


#pragma omp parallel for schedule(dynamic,1)
			for(int np=0 ; np<NPoints ; np++) {
				vec_Points2[np]=CG_Point2d(x2P[np*2] , x2P[np*2+1]);
			}


			findAlphaEdges(ccHull , vec_Points2) ;


			return 0;
	}


}