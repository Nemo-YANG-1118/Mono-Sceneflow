
#include <opencv2/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/cudalegacy.hpp>

#include <iostream>
#include <algorithm>
#include <iterator>
#include <utility>
#include <numeric>

#include <omp.h>


#include "OpticalFlowMatcher.h"
#include "MSOpticalFlow/OpticalFlow.h"


#undef _PRINT_VEC_
#define _PRINT_VEC_(_biter , _eiter , _typename)\
	fflush(stdout);\
	std::copy(_biter , _eiter , (std::ostream_iterator<_typename>(std::cout , "\t\t"))) ;\
	std::cout<<std::endl ;\
	fflush(stdout);


#ifdef _MIN_ABS_
#undef _MIN_ABS_
#endif
#define _MIN_ABS_(_x , _y)\
	( std::fabs(_x)<std::fabs(_y)? _x : _y)\


#undef _DO_BACKWARDS_TRACKING_
#define _DO_BACKWARDS_TRACKING_ 0



namespace stereoscene{



	bool
		OpticalFlowMatcher::
		denseTracking_SIMPLE(
		std::map<int , cv::Mat>& _map_uFlow ,
		std::map<int , cv::Mat>& _map_vFlow ,
		const std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>& _map_ImagesPair) {



			const int NPairs = _map_ImagesPair.size() ;

#pragma omp parallel for schedule(dynamic ,1)
			for(int np=0 ; np<NPairs ; np++) {

				std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>::const_iterator it_d_Image01 = _map_ImagesPair.begin() ;
				std::advance(it_d_Image01 , np);


				const cv::Size ImageSize = it_d_Image01->second.first->size() ;


				cv::Mat uvflow ;
				cv::optflow::calcOpticalFlowSF(*it_d_Image01->second.first , *it_d_Image01->second.second ,\
					uvflow,\
					3, 20 , 10) ;

				//cv::Ptr<cv::DenseOpticalFlow> dense_ =  cv::optflow::createOptFlow_DeepFlow() ;

								/*
				sigma = 0.6f;
				minSize = 25;
				downscaleFactor = 0.95f;
				fixedPointIterations = 5;
				sorIterations = 25;
				alpha = 1.0f;
				delta = 0.5f;
				gamma = 5.0f;
				omega = 1.6f;
				*/
	
				//dense_->setDouble("downscaleFactor" , 0.8);
				//dense_->setInt("sorIterations" , 30);
				//std::vector<std::string> vec_Params ;
				//dense_->getParams(vec_Params);
				//dense_->

				//_PRINT_VEC_(vec_Params.begin() , vec_Params.end() , std::string) ;

				//dense_->calc(*it_d_Image01->second.first , *it_d_Image01->second.second , uvflow);


				std::cout<<"flow type : "<<uvflow.type()<<std::endl;

				cv::Mat uFlow(ImageSize , CV_32FC1) , vFlow(ImageSize , CV_32FC1);
				
				for(int nh=0 ; nh<ImageSize.height ; nh++) {
					const cv::Point2f* puvflow = uvflow.ptr<cv::Point2f>(nh) ;
					float *puflow = uFlow.ptr<float>(nh) ,
						*pvflow = vFlow.ptr<float>(nh) ;

					for(int nw=0 ; nw<ImageSize.width ; nw++, puvflow++ , puflow++ , pvflow++) {
						*puflow = puvflow->x ;
						*pvflow = puvflow->y ;
					}
				}


				uFlow.convertTo(uFlow , CV_32FC1) , vFlow.convertTo(vFlow , CV_32FC1);

				_map_uFlow.insert(std::make_pair(np , uFlow)) , _map_vFlow.insert(std::make_pair(np , vFlow)) ;

			}


			return true ;
	}


	bool
		OpticalFlowMatcher::
		denseTracking_SOR(
		std::map<int , cv::Mat>& _map_uFlow ,
		std::map<int , cv::Mat>& _map_vFlow ,
		const std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>& _map_ImagesPair){


			const int NPairs = _map_ImagesPair.size() ;

#pragma omp parallel for schedule(dynamic ,1)
			for(int np=0 ; np<NPairs ; np++) {

				std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>::const_iterator it_d_Image01 = _map_ImagesPair.begin() ;
				std::advance(it_d_Image01 , np);

				const cv::Size ImageSize = it_d_Image01->second.first->size() ;


				DImage DImage0(ImageSize.width ,ImageSize.height , 1) , 
					DImage1(ImageSize.width , ImageSize.height , 1) ;

				const float* psrc0 = it_d_Image01->second.first->ptr<float>(0),
					*psrc1 = it_d_Image01->second.second->ptr<float>(0);

				double *pdst0 = DImage0.pData , *pdst1 = DImage1.pData ;

				for(int npCnt=0 ; npCnt<ImageSize.width*ImageSize.height ; npCnt++) {
					*pdst0++ = static_cast<double>(*psrc0++) , *pdst1++ = static_cast<double>(*psrc1++) ;
				}
				//cv::imshow("temp" , *it_d_Image01->second.second);
				//cv::waitKey(0);

				const double DAlpha = 0.020 ,
					DRatio = 0.80 ;	//gaussian pyramid scalar factor

				const int NMinWidth = int(std::ceil(ImageSize.width/50.0)) , 
					NOuterFPIters = 6 ,  
					NInnerFPIters = 1 ,
					NSORIters= 30 ;

				DImage du , dv , dwarp1;

				OpticalFlow::Coarse2FineFlow(du, dv ,dwarp1 , DImage0 , DImage1 ,
					DAlpha, DRatio, NMinWidth, NOuterFPIters, NInnerFPIters, NSORIters);
			

				cv::Mat uFlow(ImageSize , CV_64FC1) , vFlow(ImageSize , CV_64FC1);
				memcpy(uFlow.ptr<double>(0) , du.data() , sizeof(double)*ImageSize.width*ImageSize.height);
				memcpy(vFlow.ptr<double>(0) , dv.data() , sizeof(double)*ImageSize.width*ImageSize.height);

				uFlow.convertTo(uFlow , CV_32FC1) , vFlow.convertTo(vFlow , CV_32FC1);

				_map_uFlow.insert(std::make_pair(np , uFlow)) , _map_vFlow.insert(std::make_pair(np , vFlow)) ;
			}


			return true ;
	}


	bool 
		OpticalFlowMatcher::
		denseTracking_BROX(
		std::map<int , cv::Mat>& _map_uFlow ,
		std::map<int , cv::Mat>& _map_vFlow ,
		const std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat>>& _map_ImagesPair) {


			const int NPairs = _map_ImagesPair.size() ;

#pragma omp parallel for schedule(dynamic ,1)
			for(int np=0 ; np<NPairs ; np++) {

				std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat>>::const_iterator it_d_Image01 = _map_ImagesPair.begin() ;
				std::advance(it_d_Image01 , np);


				const cv::cuda::GpuMat &d_Image0 = it_d_Image01->second.first ,
					&d_Image1 = it_d_Image01->second.second ;

				cv::cuda::GpuMat duFlow(d_Image0.size() , CV_32FC1) , dvFlow(d_Image0.size() , CV_32FC1) ;



				///default parameters for size under the 640x480
				///0.197f-alpha , 50.0f-gamma , 0.8f-scale_factor , 10-inner_iterations , 77-outer_iterations , 10-solver_iterations
				cv::cuda::BroxOpticalFlow Brox(0.196 , 50.0f , 0.75f , 10 , 77 , 20);

				if(d_Image0.cols>640 || d_Image0.rows>480){
					if(d_Image0.cols>1280 || d_Image0.rows>960) {

						Brox.alpha = 0.216 ;
						Brox.gamma = 60.0f ;
						Brox.scale_factor = 0.80f ;
						Brox.inner_iterations = 10 ;
						Brox.outer_iterations = 100 ;
						Brox.solver_iterations = 30 ;

					} else {
						Brox.alpha = 0.206 ;
						Brox.gamma = 55.0f ;
						Brox.scale_factor = 0.80f ;
						Brox.inner_iterations = 10 ;
						Brox.outer_iterations = 100 ;
						Brox.solver_iterations = 26 ;
						//Brox.buf
					}
				} 


				cv::Mat uFlow , vFlow;
#pragma omp critical 
				{
					Brox(d_Image0 , d_Image1 , duFlow , dvFlow);
				}

				duFlow.download(uFlow) , _map_uFlow.insert(std::make_pair(np , uFlow)) ;
				dvFlow.download(vFlow) , _map_vFlow.insert(std::make_pair(np , vFlow)) ;
			}


			return true ;
	}



	bool
		OpticalFlowMatcher::
		denseTracking_BROX_NCV(
		std::map<int , cv::Mat>& _map_uFlow ,
		std::map<int , cv::Mat>& _map_vFlow ,
		const std::map<int , std::pair<const cv::Mat* , const cv::Mat*>>& _map_ImagesPair) {


			const int NPairs = _map_ImagesPair.size() ;

#pragma omp parallel for schedule(dynamic ,1)
			for(int np=0 ; np<NPairs ; np++) {

				std::map<int , std::pair<const cv::Mat*, const cv::Mat*>>::const_iterator it_d_Image01 = _map_ImagesPair.begin() ;
				std::advance(it_d_Image01 , np);

				const cv::Mat &img0 = *it_d_Image01->second.first , &img1 = *it_d_Image01->second.second ;

				const int NWidth = img0.cols , NHeight = img0.rows ;

				cv::Mat uflow_(NHeight , NWidth , CV_32FC1) , vflow_(NHeight , NWidth , CV_32FC1) ;


				///BroxOptFlow data structure
				NCVBroxOpticalFlowDescriptor desc;
				desc.alpha = 0.15f;
#if _DO_BACKWARDS_TRACKING_
				desc.gamma = 50.0f ;
#else
				desc.gamma = 60.0f;
#endif

				if(NWidth>640 || NHeight>480){
					if(NWidth>1280 || NHeight>960) {
						desc.number_of_inner_iterations  = 10;
						desc.number_of_outer_iterations  = 150;
						desc.number_of_solver_iterations = 30;

						desc.scale_factor = 0.9f;

					} else {
						desc.number_of_inner_iterations  = 10;
						desc.number_of_outer_iterations  = 100;
#if _DO_BACKWARDS_TRACKING_
						desc.number_of_solver_iterations = 12 ;
#else
						desc.number_of_solver_iterations = 20 ;
#endif

						desc.scale_factor = 0.86f;
					}

				} else {
					desc.number_of_inner_iterations  = 10;
					desc.number_of_outer_iterations  = 80;
					desc.number_of_solver_iterations = 20;

					desc.scale_factor = 0.8f;
				}


				///allocate memory for both host and device
				///
				int devId;
				cudaGetDevice(&devId);

				cudaDeviceProp devProp;
				cudaGetDeviceProperties(&devProp, devId);

				cv::Ptr<INCVMemAllocator> p_CUDAMemAllocator = cv::Ptr<INCVMemAllocator> (new NCVMemNativeAllocator (NCVMemoryTypeDevice, static_cast<Ncv32u>(devProp.textureAlignment)));		
				cv::Ptr<INCVMemAllocator> p_HostMemAllocator = cv::Ptr<INCVMemAllocator> (new NCVMemNativeAllocator (NCVMemoryTypeHostPageable, static_cast<Ncv32u>(devProp.textureAlignment)));



				cv::Ptr<NCVMatrixAlloc<Ncv32f> > h_img0 = cv::Ptr<NCVMatrixAlloc<Ncv32f> > (new NCVMatrixAlloc<Ncv32f> (*p_HostMemAllocator, NWidth , NHeight)),
					h_img1 = cv::Ptr<NCVMatrixAlloc<Ncv32f> > (new NCVMatrixAlloc<Ncv32f> (*p_HostMemAllocator, NWidth , NHeight));



				//#pragma omp parallel for schedule(dynamic , 1)
				for(int nh=0 ; nh<NHeight ; nh++) {
					const float *pimg0 = img0.ptr<float>(nh), 
						*pimg1 = img1.ptr<float>(nh) ;

					float *ph_img0 = &h_img0->ptr()[nh*h_img0->stride()] ,
						*ph_img1 = &h_img1->ptr()[nh*h_img1->stride()] ;

					for (int nw = 0; nw < NWidth ; ++nw , pimg0++ , pimg1++ , ph_img0++ , ph_img1++){
						*ph_img0 = *pimg0 , *ph_img1 = *pimg1 ;
					}
				}


				cv::Ptr<NCVMatrixAlloc<Ncv32f> > d_img0 (new NCVMatrixAlloc<Ncv32f> (*p_CUDAMemAllocator,h_img0->width (), h_img0->height ()));
				cv::Ptr<NCVMatrixAlloc<Ncv32f> > d_img1 (new NCVMatrixAlloc<Ncv32f> (*p_CUDAMemAllocator, h_img0->width (), h_img0->height ()));

				///dump data from host into device
				h_img0->copySolid(*d_img0 , 0 );
				h_img1->copySolid(*d_img1 , 0 );


				NCVMatrixAlloc<Ncv32f> d_u01(*p_CUDAMemAllocator , NWidth , NHeight) ,
					d_v01(*p_CUDAMemAllocator , NWidth , NHeight) ;

#if _DO_BACKWARDS_TRACKING_
				NCVMatrixAlloc<Ncv32f> d_u10(*p_CUDAMemAllocator , NWidth , NHeight) ,
					d_v10(*p_CUDAMemAllocator , NWidth , NHeight) ;
#endif

				const cudaError_t cuState_0 = cudaGetLastError() ;
				if (cuState_0!=cudaSuccess){
					fflush(stderr) ;
					fprintf(stderr , "\nError: NCVBroxOpticalFlow failed in cuState_0\n");
					fflush(stderr) ;

					exit(0);
				}


				//std::cout << "Estimating optical flow\nForward...\n";
#pragma omp critical
				{
					///forwards matching from 0 to 1
					if (NCV_SUCCESS != NCVBroxOpticalFlow (desc, *p_CUDAMemAllocator, *d_img0 , *d_img1, d_u01, d_v01, 0)){
						fflush(stderr) ;
						fprintf(stderr , "\nError: BroxOpticalFlow failed on forwards matching\n");
						fflush(stderr) ;

						exit(0);
					}

#if _DO_BACKWARDS_TRACKING_
					///backwards matching from 1 to 0
					if(NCV_SUCCESS != NCVBroxOpticalFlow (desc, *p_CUDAMemAllocator, *d_img1 , *d_img0, d_u10, d_v10, 0)){
						fflush(stderr) ;
						fprintf(stderr , "\nError: BroxOpticalFlow failed on backwards matching\n");
						fflush(stderr) ;

						exit(0);
					}
#endif
				}


				NCVMatrixAlloc<Ncv32f> h_u01(*p_HostMemAllocator, d_u01.width(), d_u01.height()) ;
				NCVMatrixAlloc<Ncv32f> h_v01(*p_HostMemAllocator, d_u01.width(), d_u01.height());

				d_u01.copySolid (h_u01, 0);
				d_v01.copySolid (h_v01, 0);


#if _DO_BACKWARDS_TRACKING_
				NCVMatrixAlloc<Ncv32f> h_u10(*p_HostMemAllocator, d_u01.width(), d_u01.height()) ,
					h_v10(*p_HostMemAllocator, d_u01.width (), d_u01.height ());

				d_u10.copySolid(h_u10, 0) ;
				d_v10.copySolid(h_v10, 0) ;
#endif


				const cudaError_t cuState_1 = cudaGetLastError() ;
				if (cuState_1!=cudaSuccess){
					fflush(stderr) ;
					fprintf(stderr , "\nError: NCVBroxOpticalFlow failed in cuState_1\n");
					fflush(stderr) ;

					exit(0);
				}


				///Convert data into unaligned

				//#pragma omp parallel for schedule(dynamic ,1)
				for (int nh = 0; nh < NHeight; ++nh){

					const float *ph_u01 = &h_u01.ptr()[nh*h_u01.stride()] , 
						*ph_v01 = &h_v01.ptr()[nh*h_v01.stride()] ;

#if _DO_BACKWARDS_TRACKING_
					const float *ph_u10 = &h_u10.ptr()[nh*h_u10.stride()] ,
						*ph_v10 = &h_v10.ptr()[nh*h_v10.stride()] ;
#endif

					float *pdu = uflow_.ptr<float>(nh) ,
						*pdv = vflow_.ptr<float>(nh) ;

					for (int nw = 0; nw<NWidth ; ++nw , 
						ph_u01++ , ph_v01++ , 
#if _DO_BACKWARDS_TRACKING_
						ph_u10++ , ph_v10++ ,
#endif
						pdu++ , pdv++){
#if _DO_BACKWARDS_TRACKING_
							*pdv = - ( fabsf(*ph_v01+ *ph_v10)<1e-2 ? *ph_v01 : (*ph_v01-*ph_v10)/2 ) , 
								*pdu = ( fabsf(*ph_u01+ *ph_u10)<1e-2 ? *ph_u01 : (*ph_u01-*ph_u10)/2 ) ;
#else 
							*pdv = -*ph_v01 , *pdu = *ph_u01 ;		
#endif
					}
				}		

				_map_uFlow.insert(std::make_pair(np , uflow_)) , _map_vFlow.insert(std::make_pair(np , vflow_)) ;
			}


			return true ;
	}



	bool 
		OpticalFlowMatcher::
		denseTracking_TVL1(
		std::map<int , cv::Mat>& _map_uFlow ,
		std::map<int , cv::Mat>& _map_vFlow ,
		const std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat>>& _map_ImagesPair) {


			const int NPairs = _map_ImagesPair.size() ;

#pragma omp parallel for schedule(dynamic ,1)
			for(int np=0 ; np<NPairs ; np++) {

				std::map<int , std::pair<cv::cuda::GpuMat , cv::cuda::GpuMat>>::const_iterator it_d_Image01 = _map_ImagesPair.begin() ;
				std::advance(it_d_Image01 , np);


				const cv::cuda::GpuMat &d_Image0 = it_d_Image01->second.first ,
					&d_Image1 = it_d_Image01->second.second ;

				cv::cuda::GpuMat duFlow(d_Image0.size() , CV_32FC1) , dvFlow(d_Image0.size() , CV_32FC1) ;

				cv::cuda::OpticalFlowDual_TVL1_CUDA DenseTracker_ ;
				DenseTracker_.lambda = 0.10 ;
				DenseTracker_.theta = 0.15 ;
				DenseTracker_.tau = 0.10 ;
		

				if(d_Image0.cols>640 || d_Image0.rows>480){
					if(d_Image0.cols>1280 || d_Image0.rows>960) {

						DenseTracker_.nscales = 10 ;

						DenseTracker_.iterations = 100 ;

					} else {
						DenseTracker_.nscales = 9 ;

						DenseTracker_.iterations = 70 ;
					}

				} else {
					DenseTracker_.nscales = 6 ;

					DenseTracker_.iterations = 60 ;
				}


				cv::Mat uFlow , vFlow;
#pragma omp critical 
				{
					DenseTracker_(d_Image0 , d_Image1 , duFlow , dvFlow);
				}

				duFlow.download(uFlow) , _map_uFlow.insert(std::make_pair(np , uFlow)) ;
				dvFlow.download(vFlow) , _map_vFlow.insert(std::make_pair(np , vFlow)) ;

			}


			return true ;
	}

}