
#include <iostream>
#include <iterator>
#include <utility>


#include "PhotoCamera.h"


#ifdef _PRINT_FLOATVEC_
#undef _PRINT_FLOATVEC_
#endif
#define _PRINT_FLOATVEC_(_vec_data)\
	fflush(stdout);\
	std::copy(_vec_data.begin() , _vec_data.end() , (std::ostream_iterator<float>(std::cout , "\t "))) ;\
	std::cout<<"\n" ;\
	fflush(stdout);


namespace stereoscene{
	namespace reconstructing{
	

		void 
			PhotoCamera::
			calculateProjectiveMatrix(
			std::vector<float>& P , 
			std::vector<float>& K , 
			std::vector<float>& R ,
			std::vector<float>& tvec) {

				if(P.size()<12) {
					P.resize(12 ,0) ;
				}


				cv::Mat P_ = cv::Mat(3 ,4 ,CV_32FC1 , P.data()); 
				{
					for(int rr=0 ; rr<3  ; rr++) {
						memcpy(P_.ptr<float>(0)+4*rr , &R[rr*3] , sizeof(float)*3) ;
						memcpy(P_.ptr<float>(0)+4*rr+3 , &tvec[rr] , sizeof(float)) ;
					}
				}
				

				cv::Mat K_(3 , 3 , CV_32FC1 , K.data()) ;

				P_ = K_*P_ ;
		}



		void
			PhotoCamera::
			calculateOpticalCenter(
			std::vector<float>& cvec , 
			std::vector<float>& R ,
			std::vector<float>& tvec) {

				if(cvec.size()<3) {
					cvec.resize(3 , 0) ;
				}

				cv::Mat& cvec_ = cv::Mat(3 , 1 , CV_32FC1 , cvec.data()) ;
				cv::Mat& tvec_ = cv::Mat(3 , 1 , CV_32FC1 , tvec.data()) ;
				cv::Mat& R_ = cv::Mat(3 , 3,  CV_32FC1 , R.data()) ;


				cvec_ = -R_.t() * tvec_ ;
		}



		void
			PhotoCamera::
			deepCopyMembers(const PhotoCamera& _PCam) {

				_P_.assign(_PCam._P_.begin() , _PCam._P_.end());
				_K_.assign(_PCam._K_.begin() , _PCam._K_.end());
				_R_.assign(_PCam._R_.begin() , _PCam._R_.end());
				_tvec_.assign(_PCam._tvec_.begin() , _PCam._tvec_.end()) ;
				_cvec_.assign(_PCam._cvec_.begin() , _PCam._cvec_.end()) ;
		}

	

		void
			PhotoCamera::
			transposeMatrix(
			std::vector<float>& _dst ,
			const std::vector<float>& _src , 
			const int srcRows ,
			const int srcCols) {

				_dst.resize(srcCols*srcRows);

				cv::Mat src_ = cv::Mat(srcRows , srcCols , CV_32FC1 , (void*)_src.data()) ;
				cv::Mat dst_ = cv::Mat(srcCols , srcRows , CV_32FC1 , (void*)_dst.data()) ;

				cv::transpose(src_ , dst_) ;

				//std::cout<<src_<<"\n"<<dst_<<"\n\n" ;
		}


		void
			PhotoCamera::
			reproject3dPoints(
			std::vector<float>& x2 , 
			const std::vector<float>& x3 ,
			const std::vector<float>& P) {

				cv::Mat x2h_ = cv::Mat::ones(3 , 1 , CV_32FC1) ;
				cv::Mat x3h_ = cv::Mat::ones(4 , 1 , CV_32FC1) ;
				cv::Mat& P_ = cv::Mat(3 , 4 , CV_32FC1) ;
				
				memcpy(x3h_.ptr<float>(0) , x3.data() , sizeof(float)*3) ;
				memcpy(P_.ptr<float>(0) , P.data() , sizeof(float)*12) ;

				x2h_ = P_*x3h_ ;

				x2h_/=x2h_.at<float>(2) ;

				if(x2.empty()) {
					x2.resize(2 ,0) ;
				}
				

				memcpy(x2.data() , x2h_.ptr<float>(0) , sizeof(float)*2) ;
		}



		void
			PhotoCamera::
			seeMembers(
			const PhotoCamera& _PCam ,
			const bool BProjection,
			const bool BIntrinsic,
			const bool BRotation, 
			const bool BTranslation,
			const bool BOpticalCenter) {

				if(BProjection) {
					fprintf(stdout, "< Projection > :\n") ;
					_PRINT_FLOATVEC_(_PCam._P_) ;
					fprintf(stdout, "\n") ;
				}

				if(BIntrinsic) {
					fprintf(stdout, "< Intrinsic > :\n") ;
					_PRINT_FLOATVEC_(_PCam._K_) ;
					fprintf(stdout, "\n") ;
				}

				if(BRotation) {
					fprintf(stdout, "< Rotation > :\n") ;
					_PRINT_FLOATVEC_(_PCam._R_) ;
					fprintf(stdout, "\n") ;
				}

				if(BTranslation) {
					fprintf(stdout, "< Translation > :\n") ;
					_PRINT_FLOATVEC_(_PCam._tvec_) ;
					fprintf(stdout, "\n") ;
				}

				if(BOpticalCenter) {
					fprintf(stdout, "< Optical Center > :\n") ;
					_PRINT_FLOATVEC_(_PCam._cvec_) ;
					fprintf(stdout, "\n") ;
				}

		}


	}
}