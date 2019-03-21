
#ifndef TRIMESH_GENERATOR_H_
#define TRIMESH_GENERATOR_H_


#include <vector>

#include "../../SceneConfiguration.h"


#ifdef _CROSS_PRODUCT_ASSIGN_
#undef _CROSS_PRODUCT_ASSIGN_
#endif
#define _CROSS_PRODUCT_ASSIGN_(_dst3 , _x3 , _y3)\
	(_dst3[0] = _x3[1]*_y3[2]-_x3[2]*_y3[1]),\
	(_dst3[1] = _x3[2]*_y3[0]-_x3[0]*_y3[2]),\
	(_dst3[2] = _x3[0]*_y3[1]-_x3[1]*_y3[0]);\


#ifdef _CROSS_PRODUCT_DECLARE_
#undef _CROSS_PRODUCT_DECLARE_
#endif
#define _CROSS_PRODUCT_DECLARE_(_dst3 , _x3 , _y3)\
	float _dst[3] = { (_x3[1]*_y3[2]-_x3[2]*_y3[1]),\
	(_x3[2]*_y3[0]-_x3[0]*_y3[2]), \
	(_x3[0]*_y3[1]-_x3[1]*_y3[0]) };\


namespace stereoscene{


	struct TrimeshGenerator{

	public:

		TrimeshGenerator(
			std::vector<float>& _x2Points=std::vector<float>() ,
			std::vector<cind>& _x2TriCells=std::vector<cind>()) 
			: _x2Points_(_x2Points) ,
			_x2TriCells_(_x2TriCells) {}



		bool 
			operator() (
			const bool BInverse,
			const int NPointsDim=3) {

				if(!_x2Points_.empty()) {

					const bool BDoConstrained = false ;

					generateConstrainedTrimesh(NPointsDim , BInverse , BDoConstrained) ;
				}

				return true ;
		}



	public:

		static bool
			triangulateDelaunayMesh(
			std::vector<cind>& _xTriCells ,
			const std::vector<float>& _xPoints, 
			const int Nxdim=3  , 
			const bool BInverse=false ,
			std::vector<float>& _x3Normals=std::vector<float>() ,
			const bool BComputePointsNormals=false);


		static bool
			triangulateDelaunayMesh(
			std::vector<cind>& _xTriCells ,
			const std::vector<const float*>& _xPoints, 
			const int Nxdim=3 , 
			const bool BInverse=false ,
			std::vector<float>& _x3Normals=std::vector<float>() ,
			const bool BComputePointsNormals=false) ;


	private:

		std::vector<float> & _x2Points_ ;

		std::vector<cind>& _x2TriCells_ ;



	private:

		bool
			generateConstrainedTrimesh(
			const int Nxdim ,
			const bool BInverse ,
			const bool BConstrained) ;

	} ;


}


#endif