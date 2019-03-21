
#include <opencv2/highgui.hpp>

#include <cmath>

#include <omp.h>


#include "DepthDenoiser.h"


#undef _SEE_IMAGE_
#define _SEE_IMAGE_(img_ , sWinName_)\
	cv::namedWindow(sWinName_ , 1) ;\
	cv::imshow(sWinName_ , img_);\
	cv::waitKey(0);\
	cv::destroyWindow(sWinName_);


namespace stereoscene{


	bool 
		DepthDenoiser::
		denoiseDepth(
		cv::Mat& _Dst ,
		const cv::Mat& _Src , 
		const cv::Mat& _SrcMask , 
		const int NIters ,
		const double DLambda , 
		const double DTheta , 
		const double DTau , 
		const bool BGWeighted ,
		const cv::Mat _Grad) {


			const int NCols = _Src.cols , 
				NRows = _Src.rows ,
				NPad = 2 ;


			const double DSigma = 0.1/DTau ;


			cv::Mat U_ ,P_ = cv::Mat::zeros(NRows, NCols, CV_64FC2) , V_ = cv::Mat::zeros(NRows , NCols , CV_64FC1) ;

			const cv::Mat &UMask = _SrcMask ;


			if(_Src.type()!=CV_64FC1 && _Src.type()!=CV_64F) {

				fflush(stderr) ;
				fprintf(stderr , "Error in denoise_ROF() : inputs must be double type \n") ;
				fflush(stderr);

				exit(0);

			} else {
				_Src.copyTo(U_) ;
			}


			for(int niter = 0; niter < NIters; niter++ ){

				/// Updating P:
				///
				const double DSigmaP = (!niter) ? (1 + DSigma) : DSigma;

				/// P' = P + Sigma*Grad(U)
				/// P(x,y) = P'(x,y)/max(||P(x,y)||,1)
#pragma omp parallel for schedule(dynamic, 1)
				for(int ny=NPad; ny <NRows-NPad; ny++ ){

					const double* p_U11 = U_.ptr<double>(ny)+NPad, 
						* p_U21 = p_U11+1 ,
						* p_U12 = U_.ptr<double>(ny+1)+NPad;

					cv::Point2d* p_P = P_.ptr<cv::Point2d>(ny)+NPad;

					const unsigned char *p_Mask = UMask.ptr<unsigned char>(ny)+NPad ;

					if(BGWeighted){

						const double *p_G = _Grad.ptr<double>(ny)+NPad ;

						for(int nx =NPad; nx < NCols-NPad; nx++ , 
							p_U11++ , p_U12++ , p_U21++ , p_P++ , p_G++, p_Mask++){
								if(*p_Mask) {
									/// Forward Difference: P = P + Sigma*Div(P)
									const double &&px_ = (p_U21[0] - p_U11[0])*DSigmaP + p_P[0].x;
									const double &&py_ = (p_U12[0] - p_U11[0])*DSigmaP + p_P[0].y;

									/// Gradient Weighted Normalization : 1/(1+|Div(P(x))|/G(x))
									const double &&DInvPNorm = 1.0/(1.0 + std::sqrt(px_*px_ + py_*py_)/(*p_G)) ;									

									p_P[0].x = px_ * DInvPNorm;
									p_P[0].y = py_ * DInvPNorm;
								}
						}

					} else {

						for(int nx =NPad; nx < NCols-NPad; nx++ , 
							p_U11++ , p_U12++ , p_U21++ , p_P++ , p_Mask++){
								if(*p_Mask) {
									/// Forward difference: P = P + Sigma*Div(P)
									const double &&px_ = (p_U21[0] - p_U11[0])*DSigmaP + p_P[0].x;
									const double &&py_ = (p_U12[0] - p_U11[0])*DSigmaP + p_P[0].y;

									/// Normalization
									const double &&DInvPNorm = 1.0/(std::sqrt(px_*px_ + py_*py_)+ 1.0) ;

									p_P[0].x = px_ * DInvPNorm;
									p_P[0].y = py_ * DInvPNorm;
								}
						}
					}
				}


				/// Updating V: 
				///
				/// V := {-lambda , v+sigma*(u-f) , lambda}
				///    =>	lambda			; iff: v+sigma*(u-s)>lambda
				///			v+sigma*(u-s)	; iff: -lambda<v+sigma<lambda
				///			-lambda			; iff: v+sigma*(u-s)<-lambda
				///
#pragma omp parallel for schedule(dynamic , 1)
				for(int ny=NPad ; ny<NRows-NPad ; ny++) {

					double *p_V = V_.ptr<double>(ny)+NPad ;
					const double *p_U = U_.ptr<double>(ny)+NPad , 
						*p_Src = _Src.ptr<double>(ny)+NPad;

					const unsigned char* p_Mask = _SrcMask.ptr<unsigned char>(ny)+NPad;

					for(int nx=NPad ; nx<NCols-NPad ; nx++) {

						if(p_Mask[nx]) {

							const double && Vxyn = p_V[nx]+DSigma*(p_U[nx]-p_Src[nx]);

							///marching step should be constrained into [-lambad , lambda]
							p_V[nx] = std::max(std::min(Vxyn , DLambda) , -DLambda) ;
						}
					}
				}


#pragma omp parallel for schedule(dynamic , 1)
				for(int ny = NPad; ny < NRows-NPad; ny++ ){

					/// Updating U:
					double* p_U = U_.ptr<double>(ny)+NPad ;
					const cv::Point2d* p_P11 = P_.ptr<cv::Point2d>(ny)+NPad ,
						*p_P10 = P_.ptr<cv::Point2d>(ny-1)+NPad , 
						*p_P01 = p_P11 - 1;

					const unsigned char* p_UMask = UMask.ptr<unsigned char>(ny)+NPad ;

					const double* p_V = V_.ptr<double>(ny)+NPad ;

					/// U1 = U + Tau*(-nablaT(P))
					for(int nx = NPad; nx < NCols-NPad; nx++ ,
						p_U++ , p_P01++ , p_P10++ , p_P11++ , p_UMask++ , p_V++ ){

							if(*p_UMask) {

								/// U1 = U + Tau*(-nablaT(P))
								const double&& Unew = *p_U + DTau*(p_P11->x - p_P01->x + p_P11->y - p_P10->y - p_V[nx]);

								/// U = U2 + Theta*(U2 - U)
								*p_U = Unew + DTheta*(Unew - *p_U);
							}
					}
				}
			}


			U_.copyTo(_Dst);


			return true ;
	}


}
