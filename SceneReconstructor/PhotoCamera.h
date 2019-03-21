
#ifndef PHOTO_CAMERA_H_
#define PHOTO_CAMERA_H_

#include <opencv2/core.hpp>


#include <vector>



namespace stereoscene{


		struct PhotoCamera{

		public:

			PhotoCamera(
				const std::vector<float>& K_ = std::vector<float>(9, 0), 
				const std::vector<float>& R_ = std::vector<float>(9, 0) , 
				const std::vector<float>& Tvec_ = std::vector<float>(3, 0)) {

					_K_.swap(std::vector<float>(K_.begin() , K_.end())) ;
					_R_.swap(std::vector<float>(R_.begin() , R_.end())) ;
					_Tvec_.swap(std::vector<float>(Tvec_.begin() , Tvec_.end())) ;

					calculateProjectiveMatrix(_P_ , _K_ , _R_ , _Tvec_) ;

					calculateOpticalCenter(_Cvec_ , _R_ , _Tvec_) ;
			} 



		public:
			/// Calculate projective matrix from K,R,t
			void 
				calculateProjectiveMatrix(
				std::vector<float>& P , 
				std::vector<float>& K , 
				std::vector<float>& R ,
				std::vector<float>& Tvec) ;


			/// Calculate optical center from t
			void
				calculateOpticalCenter(
				std::vector<float>& Cvec , 
				std::vector<float>& R ,
				std::vector<float>& Tvec) ;

			/// Deeply copy
			void
				deepCopyMembers(
				const PhotoCamera& PCam) ;


			void
				swap(PhotoCamera& _PCam) ;


		public:
			/// See all of data members in Photo Camera
			static void
				seeMembers(
				const PhotoCamera& _PCam ,
				const bool BProjection=true ,
				const bool BIntrinsic=true , 
				const bool BRotation=true , 
				const bool BTranslation=true , 
				const bool BOpticalCenter=true) ;


			/// Reproject 3d points to get coincidence 2d-coordinates  
			static void
				reproject3dPoints(
				std::vector<float>& x2 , 
				const std::vector<float>& x3 , 
				const std::vector<float>& P) ;


			/// Acquire transposed matrix
			static void
				transposeMatrix(
				std::vector<float>& dst ,
				const std::vector<float>& src , 
				const int srcRows ,
				const int srcCols) ;


		public:
			/// Projection: 3x4
			std::vector<float> _P_ ;


			/// Intrinsic: 3x3
			std::vector<float> _K_ ;


			/// Rotation: 3x3
			std::vector<float> _R_ ;


			/// Translation: 3x1
			std::vector<float> _Tvec_ ; 

			/// Optical Center: 3x1
			std::vector<float> _Cvec_ ; 

		} ;


}




#endif
