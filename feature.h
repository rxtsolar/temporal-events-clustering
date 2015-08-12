#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <vector>
#include <opencv2/opencv.hpp>

struct PhotoInfo;

std::vector<double> getNoveltyScores(const std::vector<int>& times, double K,
		int kernelSize, double kernelSigma);
cv::Mat getTimeFeatures(const std::vector<PhotoInfo>& info);
cv::Mat getLabels(const std::vector<PhotoInfo>& info);
cv::Mat getHistogram(const cv::Mat& image);
cv::Mat getHistImage(const cv::Mat& histogram);
cv::Mat getDCTHist(const cv::Mat& image);
cv::Mat getGistFeatures(const cv::Mat& image);
cv::Mat blendFeatures(const std::vector<cv::Mat>& features,
		const std::vector<double>& rates);

void preprocess(cv::Mat& image, int orientation);

cv::Mat spectralClustering(const cv::Mat& features,
		int n1, int n2, double r1, double r2);
int countFaces(const cv::Mat& image);

#endif
