
#include <iostream>
#include <cmath>
#include <utility>

#include <omp.h>

#include "PrimalDualSolver.h"



namespace stereoscene{


	bool
		PrimalDualSolver::
		operator() (
		const int NIters ,
		const double DLambda ,
		const double DTheta ,
		const double DTau ,  
		const bool BWeighted ,
		const cv::Mat& _WMap ,
		const bool BUseGPU) {


			const int _NWidth_ = _UMask_.cols , 
				_NHeight_ = _UMask_.rows  ,
				NMaps = _vec_VMaps_.size() ;


			const double DSigma = 0.1/DTau ;

			for(int niter = 0; niter < NIters; niter++ ){


				///step.1: update primal: P
				///
				const double DSigmaP = (!niter) ? (1 + DSigma) : DSigma;

				if(BWeighted){
					updateWeightedPrimal(DSigmaP , _WMap) ;
				} else {
					updateUnWeightedPrimal(DSigmaP);
				}


				///step.2: update residuals: V
				///
				updateSourceResidual(DLambda , DSigma);


				///step.3: update dual: U
				///
				updateDual(DTheta , DTau) ;
			}

			_UMap_.convertTo(_DstMap_ , CV_32FC1);


			return true ;
	}



	bool
		PrimalDualSolver::
		updateDual(
		const double DTheta ,
		const double DTau ) {


			///update dual: U
			const int NMaps = _vec_VMaps_.size() ;


#pragma omp parallel for schedule(dynamic , 1)
			for(int ny = _NPad_; ny < _NHeight_-_NPad_; ny++ ){

				double* pU= _UMap_.ptr<double>(ny)+_NPad_ ;
				const cv::Point2d* pP11 = _PMap_.ptr<cv::Point2d>(ny)+_NPad_ ,
					*pP10 = _PMap_.ptr<cv::Point2d>(ny-1)+_NPad_ , 
					*p_P01 = pP11 - 1;

				const unsigned char* pUMask = _UMask_.ptr<unsigned char>(ny)+_NPad_ ;


				///U1 = U + DTau*(-nablaT(P))
				for(int nx = _NPad_; nx < _NWidth_-_NPad_; nx++ ,
					pU++ , p_P01++ , pP10++ , pP11++ , pUMask++){

						double vs_(0.0) ;
						for(int nm=0; nm<NMaps ; nm++){
							vs_ += ( bool(_vec_VMasks_[nm]->ptr<unsigned char>(ny)[nx]) ? _vec_VMaps_[nm].ptr<double>(ny)[nx] : 0 ) ;	
						}

						///U1 = U + DTau*(-nablaT(P))
						if(*pUMask) {
							const double&& Unew = *pU+ DTau*(pP11->x - p_P01->x + pP11->y - pP10->y - vs_);

							///U = U2 + DTheta*(U2 - U)
							*pU= Unew + DTheta*(Unew - *pU);
						}
				}
			}



			return  true ;
	}



	bool 
		PrimalDualSolver::
		updateSourceResidual(
		const double DLambda ,
		const double DSigma) {


			for(int nm=0; nm<_vec_VMaps_.size() ;nm++){

				///update residuals of source: V 
				///
				///	 V := {-Lambda , v+sigma*(u-f) , Lambda}
				///		=>	Lambda			; iff: v+sigma*(u-s)>Lambda
				///			v+sigma*(u-s)	; iff: -Lambda<v+sigma<Lambda
				///			-Lambda			; iff: v+sigma*(u-s)<-Lambda
				///
#pragma omp parallel for schedule(dynamic , 1)
				for(int ny=_NPad_ ; ny<_NHeight_-_NPad_ ; ny++) {

					double *pV = _vec_VMaps_[nm].ptr<double>(ny)+_NPad_ ;
					const double *pU= _UMap_.ptr<double>(ny)+_NPad_ , 
						*pSrc = _vec_SrcMaps_[nm]->ptr<double>(ny)+_NPad_;

					const unsigned char* pMask =  _vec_VMasks_[nm]->ptr<unsigned char>(ny)+_NPad_;

					for(int nx=_NPad_ ; nx<_NWidth_-_NPad_ ; nx++ , 
						pMask++ , pV++ , pU++ , pSrc++) {

							if(*pMask) {

								const double && Vxyn = *pV+DSigma*(pU[0]-pSrc[0]);

								pV[0] = std::max(std::min(Vxyn , DLambda) , -DLambda) ;
							}

					}

				}
			}


			return true ;
	}



	bool
		PrimalDualSolver::
		updateUnWeightedPrimal(
		const double DSigmaP) {

			///P' = P + Sigma*Grad(U)
			///P(x,y) = P'(x,y)/(||P(x,y)||+1)
#pragma omp parallel for schedule(dynamic , 1)
			for(int ny=_NPad_; ny <_NHeight_-_NPad_; ny++ ){

				const double* pU11 = _UMap_.ptr<double>(ny)+_NPad_, 
					* pU21 = pU11+1 ,
					* pU12 = _UMap_.ptr<double>(std::min(ny+1, _NHeight_-1))+_NPad_;


				cv::Point2d* pP = _PMap_.ptr<cv::Point2d>(ny)+_NPad_;

				for(int nx =_NPad_; nx < _NWidth_-_NPad_; nx++ , 
					pU11++ , pU12++ , pU21++ , pP++){

						///forward difference: P = P + Sigma*Div(P)
						const double &&px_ = (pU21[0] - pU11[0])*DSigmaP + pP[0].x;
						const double &&py_ = (pU12[0] - pU11[0])*DSigmaP + pP[0].y;

						const double &&DInvPNorm = 1.0/(std::sqrt(px_*px_ + py_*py_) + 1.0) ;

						pP[0].x = px_ * DInvPNorm;
						pP[0].y = py_ * DInvPNorm;
				}
			}


			return true ;
	}



	bool
		PrimalDualSolver::
		updateWeightedPrimal(
		const double DSigmaP , 
		const cv::Mat& _WMap) {


			///P' = P + Sigma*Grad(U)
			///P(x,y) = P'(x,y)/(||P(x,y)||/w(,y)+1)
#pragma omp parallel for schedule(dynamic , 1)
			for(int ny=_NPad_; ny <_NHeight_-_NPad_; ny++ ){

				const double* pU11 = _UMap_.ptr<double>(ny)+_NPad_, 
					* pU21 = pU11+1 ,
					* pU12 = _UMap_.ptr<double>(ny+1)+_NPad_;


				cv::Point2d* pP = _PMap_.ptr<cv::Point2d>(ny)+_NPad_;


				const double *pW = _WMap.ptr<double>(ny)+_NPad_ ;

				for(int nx =_NPad_; nx <_NWidth_-_NPad_; nx++ , 
					pU11++ , pU12++ , pU21++ , pP++ , pW++){

						///forward difference: P = P + Sigma*Div(P)
						const double &&px_ = (pU21[0] - pU11[0])*DSigmaP + pP[0].x;
						const double &&py_ = (pU12[0] - pU11[0])*DSigmaP + pP[0].y;

						///inv-weighted normal: 1/(1+|Div(P(x))|/G(x))
						const double &&DInvPNorm = 1.0/(1.0 + std::sqrt(px_*px_ + py_*py_)/pW[0]) ;

						pP[0].x = px_ * DInvPNorm;
						pP[0].y = py_ * DInvPNorm;
				}
			}

			return true ;
	}

}