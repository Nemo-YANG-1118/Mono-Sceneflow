
#ifndef PRIMAL_DUAL_SOLVER_HPP_
#define PRIMAL_DUAL_SOLVER_HPP_

#include <opencv2/core.hpp>

#include <vector>


namespace stereoscene{


		struct PrimalDualSolver{

		public:

			PrimalDualSolver(
				cv::Mat& _DstMap , 
				const std::vector<const cv::Mat*>& _vec_SrcMap ,
				const std::vector<const cv::Mat*>& _vec_VMasks ,
				const cv::Mat& _UMask ,
				const int _UInd )
				: _DstMap_(_DstMap) ,
				_vec_SrcMaps_(_vec_SrcMap) , 
				_vec_VMasks_(_vec_VMasks) ,
				_UMask_(_UMask) , 
				_UInd_(_UInd) ,
				_NWidth_(_UMask.cols) ,
				_NHeight_(_UMask.rows) ,
				_NPad_(2) {

					_PMap_.create(_UMask_.size() , CV_64FC2);
					memset(_PMap_.ptr<double>(0) , 0 , sizeof(double)*_NWidth_*_NHeight_);


					_UMap_.create(_UMask_.size() , CV_64FC1) ;
					memcpy(_UMap_.ptr<double>(0) , _vec_SrcMaps_[_UInd_]->ptr<double>(0) , sizeof(double)*_NWidth_*_NHeight_);
					_UMap_.setTo(0 , 255-_UMask_);


					_vec_VMaps_.resize(_vec_SrcMaps_.size());
					std::for_each(
						_vec_VMaps_.begin() , 
						_vec_VMaps_.end() , 
						[&_vec_SrcMap](cv::Mat& _VMap){
							_VMap.create(_vec_SrcMap[0]->size() , CV_64FC1);
							memset(_VMap.ptr<double>(0) , 0 ,sizeof(double)*_VMap.cols*_VMap.rows);
					});
			}



			bool 
				operator() (
				const int NIters=50 ,
				const double DLambda=0.20 ,
				const double DTheta=1.0 ,
				const double DTau=0.20 ,  
				const bool BWeighted=false ,
				const cv::Mat& _WMap=cv::Mat() ,
				const bool BUseGPU=false) ;



		private:

			const std::vector<const cv::Mat*>& _vec_SrcMaps_ ;

			const std::vector<const cv::Mat*>& _vec_VMasks_ ;

			const cv::Mat& _UMask_ ;

			const int _UInd_ , _NPad_ , _NWidth_ , _NHeight_ ;



		private:

			cv::Mat _UMap_ , _PMap_ , &_DstMap_ ;

			std::vector<cv::Mat> _vec_VMaps_ ;



		private:


			bool
				solvePrimalDualL1() ;


			bool
				updateDual(
				const double DTheta ,
				const double DTau) ;



			bool 
				updateSourceResidual(
				const double DLambda ,
				const double DSigma) ;


			bool
				updateWeightedPrimal(
				const double DSigmaP , 
				const cv::Mat& _WMap) ;



			bool
				updateUnWeightedPrimal(
				const double DSigmaP) ;


		} ;


}


#endif