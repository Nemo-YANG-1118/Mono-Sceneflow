
#ifndef POINTS_CLEANER_HPP_
#define POINTS_CLEANER_HPP_


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>


#include <iostream>
#include <vector>
#include <iterator>
#include <utility>
#include <algorithm>
#include <cmath>


#include <omp.h>



namespace stereoscene{


	template<typename T>
	struct PointsCleaner{

	public:

		PointsCleaner(const bool BInverse=true , const int NNeighbors=12 , const double DPlaneDistance=3.0 , const double DStdVar=1.0)
			:_BInverse_(BInverse) ,
			_NNeighbors_(NNeighbors) , 
			_DPlaneDistance_(DPlaneDistance) ,
			_DStdVar_(DStdVar) ,
			_Pcld_(new pcl::PointCloud<pcl::PointXYZ>()){}



		bool 
			operator() (
			std::vector<std::vector<T>>& _vec_x3NewPoints ,
			const std::vector<std::vector<T>>& _vec_x3OrgPoints){


				importVectorIntoPointsCloud(_vec_x3OrgPoints) ;


				///step.1: calculate average gap between specific number of neighbors points by using kdtree
				///
				const double DGap = searchAverageGap() ;
				//printf("%.5f\n", DGap) ;


				///step.2: fit a random plane to remove further outliers
				///
				fitRandomPlane(DGap) ;


				///step.3: calculate variance to remove outliers with relatively bigger standard variance
				///
				calculateStatisticVariance() ;


				///step.4: search a specific radius to remove further outliers
				///
				searchRadiusPoints(DGap) ;



				std::vector<std::vector<T>> vec_x3New ;

				exportPointsCloudIntoVector(vec_x3New) ;


				_vec_x3NewPoints.swap(vec_x3New) ;


				return true ;
		} ;



	private:

		const bool _BInverse_ ;

		const int _NNeighbors_ ;

		const double _DPlaneDistance_ ;

		const double _DStdVar_ ;


	private:

		pcl::PointCloud<pcl::PointXYZ>::Ptr _Pcld_ ;



	private:

		bool
			searchRadiusPoints(const double DGap) {

				pcl::PointCloud<pcl::PointXYZ>::Ptr PcldNew(new pcl::PointCloud<pcl::PointXYZ>);


				pcl::RadiusOutlierRemoval<pcl::PointXYZ> Remover(false) ;
				Remover.setInputCloud (_Pcld_);
				Remover.setRadiusSearch(DGap*3.0);
				Remover.setMinNeighborsInRadius(_NNeighbors_);
				Remover.filter (*PcldNew);

				_Pcld_->swap(*PcldNew) ;


				return true ;
		}



		bool
			calculateStatisticVariance(){

				pcl::PointCloud<pcl::PointXYZ>::Ptr PcldNew(new pcl::PointCloud<pcl::PointXYZ>);


				pcl::StatisticalOutlierRemoval<pcl::PointXYZ> Remover(false) ;
				Remover.setInputCloud (_Pcld_);
				Remover.setMeanK (_NNeighbors_*3.0);
				Remover.setStddevMulThresh (_DStdVar_);
				Remover.filter (*PcldNew);

				_Pcld_->swap(*PcldNew) ;


				return true;
		}



		bool
			fitRandomPlane(
			const double DGap) {

				pcl::PointCloud<pcl::PointXYZ> PcldNew ;
				pcl::PointIndicesPtr PcldInliers (new pcl::PointIndices) ;
				pcl::ModelCoefficients::Ptr PcldCoefficients (new pcl::ModelCoefficients);


				pcl::SACSegmentation<pcl::PointXYZ> RandomPlane;
				RandomPlane.setOptimizeCoefficients(1) ;
				RandomPlane.setDistanceThreshold(_DPlaneDistance_) ; 
				RandomPlane.setModelType(pcl::SACMODEL_PLANE) ;
				RandomPlane.setMethodType(pcl::SAC_RANSAC) ;
				RandomPlane.setRadiusLimits(0.1*DGap , 0.6*DGap) ;
				RandomPlane.setInputCloud(_Pcld_) ;
				RandomPlane.segment(*PcldInliers , *PcldCoefficients) ;


				///update
				const int NInliers = PcldInliers->indices.size() ;
				PcldNew.reserve(NInliers) ;

				for(int ni=0 ; ni<NInliers ; ni++) {
					PcldNew.push_back(pcl::PointXYZ(_Pcld_->points[PcldInliers->indices[ni]])) ;
				}

				_Pcld_->swap(PcldNew) ;


				return true ;
		}



		bool
			exportPointsCloudIntoVector(
			std::vector<std::vector<T>>& _vec_x3Points){

				const int NPoints = _Pcld_->size() ;

				if(!_BInverse_) {

					_vec_x3Points.reserve(NPoints) ;

					std::for_each(_Pcld_->points.begin() , _Pcld_->points.end() ,
						[&_vec_x3Points](const pcl::PointXYZ &_pt3){
							_vec_x3Points.push_back(std::vector<T>(&_pt3.data[0] , &_pt3.data[3]))  ;
					}) ;
				} else {

					_vec_x3Points.resize(NPoints);

#pragma omp parallel for
					for(int np=0 ; np<NPoints ; np++) {
						_vec_x3Points[np].swap(std::vector<T>(&_Pcld_->points[np].data[0] , &_Pcld_->points[np].data[3])),
							_vec_x3Points[np][1]*= -1 , _vec_x3Points[np][2]*= -1 ;
					}
				}


				return true ;
		}



		bool 
			importVectorIntoPointsCloud(
			const std::vector<std::vector<T>>& _vec_x3Points) {

				const int NPoints = _vec_x3Points.size() ;

				_Pcld_->clear() , _Pcld_->points.reserve(NPoints) ;


				pcl::PointCloud<pcl::PointXYZ>& PclPoints = (*_Pcld_.get()) ;


				if(_BInverse_) {
					std::for_each(_vec_x3Points.begin() , _vec_x3Points.end() , 
						[&PclPoints](const std::vector<T>& _x3Points){
							PclPoints.push_back(pcl::PointXYZ(_x3Points[0] , -_x3Points[1] , -_x3Points[2])) ;
					}) ;
				} else {
					std::for_each(_vec_x3Points.begin() , _vec_x3Points.end() , 
						[&PclPoints](const std::vector<T>& _x3Points){
							PclPoints.push_back(pcl::PointXYZ(_x3Points[0] , _x3Points[1] , _x3Points[2])) ;
					}) ;
				}



				return true ;
		}



		double 
			searchAverageGap() {

				const int NPoints = _Pcld_->points.size() ;

				pcl::KdTreeFLANN<pcl::PointXYZ> kdTree ;
				kdTree.setInputCloud(_Pcld_);

				std::vector<float> vec_Norm2Dist(NPoints ,0) ;

#pragma omp parallel for schedule(dynamic ,1)
				for(int np=0 ; np<NPoints ; np++) {
					std::vector<float> vec_sqDistNN(_NNeighbors_) ;

					if(kdTree.nearestKSearch(_Pcld_->points[np] , _NNeighbors_ , std::vector<int>(_NNeighbors_) , vec_sqDistNN) > 0) {				
						vec_Norm2Dist[np] = std::sqrt(std::accumulate(vec_sqDistNN.begin() , vec_sqDistNN.end() , float(0))/_NNeighbors_ ) ;
					}
				}


				double DGap(0);
				std::for_each(vec_Norm2Dist.begin() , vec_Norm2Dist.end() , 
					[&DGap](const float& fval){
						DGap += (fval>1e-5 ? fval : 0) ;
				}) ;


				return (DGap/NPoints) ;
		}


	} ;


}


#endif