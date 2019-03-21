
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>


namespace stereoscene{


	__global__
	static void
	approximateMotionField_L2_Kernel(
	float* const d_dval , 
	float* const d_x3 , 
	const unsigned char* const d_dmask , 
	const float* const d_ray3 ,
	const float* const d_du , 
	const float* const d_dv ,
	const float* const d_Pvec , 
	const float* const d_Cvec ,
	const unsigned int NPoints , 
	const unsigned int NPad , 
	const unsigned int NWidth , 
	const unsigned int NHeight) {
		
		const unsigned int x3Ind = blockDim.x*blockIdx.x + threadIdx.x ;
		
		if(x3Ind< NPoints) {

			const unsigned int nw = x3Ind%NWidth , nh = x3Ind/NWidth ;

			const unsigned char* const pdmask = d_dmask+x3Ind ;

			if(*pdmask) {

				float *const px3 = d_x3 + x3Ind*3 ,
					*const pdval = d_dval + x3Ind ;

				const float *const pray3 = d_ray3 + x3Ind*3 ;

				float sx3Mot(0) , x2Mot(0) ;

				for(unsigned int npd=0 ; npd< NPad*2 ; npd++) {

					const float *const Pvec = d_Pvec+npd*12 ;


						///step.1: calculate jacobian of each x3 point
						///
						const float xh = Pvec[0]*px3[0]+Pvec[1]*px3[1]+Pvec[2]*px3[2]+Pvec[3],
							yh = Pvec[4]*px3[0]+Pvec[5]*px3[1]+Pvec[6]*px3[2]+Pvec[7] ,
							zh = Pvec[8]*px3[0]+Pvec[9]*px3[1]+Pvec[10]*px3[2]+Pvec[11] ,
							zh2 = zh*zh ;

						const float Jux = (Pvec[0]*zh-xh*Pvec[8])/zh2 , 
							Juy = (Pvec[1]*zh-xh*Pvec[9])/zh2 , 
							Juz = (Pvec[2]*zh-xh-Pvec[10])/zh2 ,
							Jvx = (Pvec[4]*zh-yh*Pvec[8])/zh2 , 
							Jvy = (Pvec[5]*zh-yh*Pvec[9])/zh2 ,
							Jvz = (Pvec[6]*zh-yh*Pvec[10])/zh2 ;


						///step.2: calculate 3d ray and 2d directionary vector
						///
						const float x2uDir = Jux*pray3[0]+Juy*pray3[1]+Juz*pray3[2] ;
						const float x2vDir = Jvx*pray3[0]+Jvy*pray3[1]+Jvz*pray3[2] ;
 

						///step.3: calculate x3's motion scale
						///
						const float *const pdu = d_du + (NHeight*NWidth)*npd + nh*NWidth+nw,
							*const pdv = d_dv + (NHeight*NWidth)*npd + nh*NWidth+nw ;

						sx3Mot += (x2uDir*(*pdu) + x2vDir*(*pdv)) ;
						x2Mot += (x2uDir*x2uDir + x2vDir*x2vDir) ;
				}

				sx3Mot /= x2Mot ;


				///step.4: update x3's position and depth value by using motion variation
				///
				px3[0] += sx3Mot*pray3[0] , px3[1] += sx3Mot*pray3[1] , px3[2] += sx3Mot*pray3[2] ;

				pdval[0] = sqrtf((px3[0]-d_Cvec[0])*(px3[0]-d_Cvec[0]) + 
					(px3[1]-d_Cvec[1])*(px3[1]-d_Cvec[1]) + 
					(px3[2]-d_Cvec[2])*(px3[2]-d_Cvec[2]));

			}
		}

	}


}


#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cuda.hpp>

#include "SceneFlowOptimizer.h"


#ifdef _SEE_IMAGE_
#undef _SEE_IMAGE_
#endif
#define _SEE_IMAGE_(img_ , wname_)\
	cv::namedWindow(wname_ , 0);\
	if(img_.cols<1680 && img_.rows<1050)\
	cv::resizeWindow(wname_ , img_.cols , img_.rows);\
		else\
	cv::resizeWindow(wname_ , img_.cols/2 , img_.rows/2);\
	cv::imshow(wname_ , img_);\
	cv::waitKey(0);\
	cv::destroyWindow(wname_);


#undef _PRINT_VEC_
#define _PRINT_VEC_(_biter , _eiter , _typename)\
	fflush(stdout);\
	std::copy(_biter , _eiter , (std::ostream_iterator<_typename>(std::cout , "\t\t"))) ;\
	std::cout<<std::endl ;\
	fflush(stdout);


#undef _MAX_ITERS_
#define _MAX_ITERS_ 1


namespace stereoscene{


	bool
		SceneFlowOptimizer::
		approximateMotionField_CUDA() {


		const cv::Size& MapSize = _DepthMap_.size() ;

		
		const unsigned int NPoints = _vec_x3Points_.size()/3,
			NCams = _vec_pPCams_.size(),
			NPad = NCams/2,
			KthRefPCam = NCams/2 ,
			NWidth = MapSize.width ,
			NHeight = MapSize.height ;


		const std::vector<float>& Cvec = _vec_pPCams_[KthRefPCam]->_Cvec_ ;


		///step.1: import data from host to device
		///
		float *d_x3(NULL) , *d_ray3(NULL) , *d_Cvec(NULL) , *d_dval(NULL) ;

		cudaMalloc( &d_x3 , sizeof(float)*NPoints*3 ) ;
		cudaMemcpyAsync(d_x3,  _vec_x3Points_.data() , sizeof(float)*NPoints*3 , cudaMemcpyHostToDevice) ;

		cudaMalloc( &d_ray3 , sizeof(float)*NPoints*3 ) ;
		cudaMemcpyAsync(d_ray3 , _vec_x3Rays_.data() , sizeof(float)*NPoints*3 , cudaMemcpyHostToDevice) ;

		cudaMalloc( &d_Cvec , sizeof(float)*3 ) ;
		cudaMemcpyAsync(d_Cvec , Cvec.data() , sizeof(float)*3 , cudaMemcpyHostToDevice) ;

		cudaMalloc( &d_dval , sizeof(float)*NWidth*NHeight ) ;
		cudaMemcpyAsync(d_dval , _DepthMap_.ptr<float>(0) , sizeof(float)*NWidth*NHeight , cudaMemcpyHostToDevice);



		unsigned char *d_dmask(NULL);

		cudaMalloc(&d_dmask,sizeof(unsigned char)*NWidth*NHeight) ;
		cudaMemcpyAsync(d_dmask,_DepthMask_.ptr<unsigned char>(0),sizeof(unsigned char)*NWidth*NHeight,cudaMemcpyHostToDevice) ;


		float *d_du(NULL) , *d_dv(NULL) , *d_Pvec(NULL);

		cudaMalloc( &d_Pvec , sizeof(float)*12*2*NPad) ;
		cudaMalloc( &d_du , sizeof(float)*(NWidth*NHeight)*2*NPad ) ;
		cudaMalloc( &d_dv , sizeof(float)*(NWidth*NHeight)*2*NPad ) ;


		std::map<int,cv::Mat>::const_iterator it_uFlow=_map_uFlow_.begin(), it_vflow=_map_vFlow_.begin() ;

		for(int cntPCam=0 , cntPad=0 ; cntPCam<NCams ; cntPCam++) {
			if(cntPCam != KthRefPCam){

				cudaMemcpyAsync(d_du + cntPad*NWidth*NHeight , it_uFlow->second.ptr<float>(0),sizeof(float)*NWidth*NHeight,cudaMemcpyHostToDevice) ;
				cudaMemcpyAsync(d_dv + cntPad*NWidth*NHeight , it_vflow->second.ptr<float>(0),sizeof(float)*NWidth*NHeight,cudaMemcpyHostToDevice) ;

				cudaMemcpyAsync(d_Pvec + cntPad*12 , _vec_pPCams_[cntPCam]->_P_.data(),sizeof(float)*12,cudaMemcpyHostToDevice) ;

				//std::cout<<it_uFlow->second.cols<<"\t"<<it_uFlow->second.rows<<"\n";

				++it_uFlow,++it_vflow;
	
				cntPad++ ;
				//_PRINT_VEC_(_vec_pPCams_[cntPCam]->_P_.data() ,_vec_pPCams_[cntPCam]->_P_.data()+12 , float);
		
				//std::cout<<cntPCam<<"\n";
			}
		}


		//cudaError_t cuError0 = cudaGetLastError() ;
		//if(cuError0 != cudaSuccess) {
		//	fflush(stderr) ;
		//	fprintf(stderr,"\nError: Error happened in < MotionFieldApproximator.cu / cuError0 >\n") ;
		//	fflush(stderr) ;
		//	exit(0);
		//}


		///step.2: implement monocular scene flow
		///
		const unsigned int NThreads = 256 ,
			NBlocks = (NPoints+NThreads-1)/NThreads ;

		for(int niter=0 ; niter<_MAX_ITERS_ ; niter++) {
			approximateMotionField_L2_Kernel<<< NBlocks , NThreads >>>( 
				d_dval , d_x3 , d_dmask , d_ray3 , d_du , d_dv  , d_Pvec , d_Cvec ,
				NPoints , NPad , NWidth , NHeight) ;
		}


		cudaError_t cuError1 = cudaDeviceSynchronize() ;
		if(cuError1 != cudaSuccess){
			fflush(stderr) ;
			fprintf(stderr , "\nError: Error happened in < MotionFieldApproximatror.cu / cuError1>\n");
			fflush(stderr) ;
			exit(0);
		}


		///step.3: export data from device to host
		///
		cv::Mat tmpDepth(_DepthMap_.size() , CV_32FC1);
		cudaMemcpy(tmpDepth.ptr<float>(0) , d_dval , sizeof(float)*NPoints , cudaMemcpyDeviceToHost) ;

		cudaFree(d_du) ;
		cudaFree(d_dv) ;
		cudaFree(d_ray3) ;
		cudaFree(d_x3) ;
		cudaFree(d_Pvec);
		cudaFree(d_Cvec) ;
		cudaFree(d_dval) ;
		cudaFree(d_dmask);



		cv::bilateralFilter(tmpDepth , _DepthMap_ , 5 , 0.1 , 5);


		return true ;
	}


}