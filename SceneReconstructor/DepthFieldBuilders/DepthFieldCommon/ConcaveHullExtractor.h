
#ifndef CONCAVE_HULL_EXTRACTOR_H_
#define CONCAVE_HULL_EXTRACTOR_H_


#include <opencv2/core.hpp>

#include <vector>
#include <list>


namespace stereoscene{


	struct ConcaveHullExtractor{

	public:
		static bool
			extractConcaveHullfromPointsVector3(
			std::vector<cv::Point>& ccHull ,
			const cv::Size& HullSize ,
			const std::vector<float>& Pvec , 
			const std::vector<std::vector<float> >& vec_x3P) ;


		static bool 
			extractConcaveHullfromPointsArray3(
			std::vector<cv::Point>& ccHull ,
			const cv::Size& HullSize ,
			const std::vector<float>& Pvec , 
			const std::vector<float>& x3P);


		static bool
			extractConcaveHullfromPointsArray2(
			std::vector<cv::Point>& ccHull ,
			const std::vector<int>& x2P);

	};


}


#endif
