#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<double> getNoveltyScores(const std::vector<int>& times, double K,
		int kernelSize, double kernelSigma);
cv::Mat getFeatures(const std::vector<int>& times);

#endif
