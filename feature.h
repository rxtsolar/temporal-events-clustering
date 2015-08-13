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
		const std::vector<int>& nFeatures, const std::vector<double>& rates);

extern const int HIST_SIZE;
extern const int SMALL_WIDTH;
extern const int MIN_NUM_EVENTS;
extern const int MAX_NUM_EVENTS;
extern const double GIST_SIGMA;
extern const double COLOR_SIGMA;
extern const char* modelName;

#endif
