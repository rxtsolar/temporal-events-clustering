#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

static Mat getSimilarityMatrix(const vector<int>& times, double K)
{
	Mat S(times.size(), times.size(), CV_64F);

	for (int i = 0; i < times.size(); i++) {
		for (int j = 0; j < times.size(); j++) {
			S.at<double>(j, i) = exp(-fabs(times[i] - times[j]) / K);
		}
	}

	return S;
}

static Mat getKernel(int size, double sigma)
{
	Mat kernel(size, size, CV_64F);
	Mat gaussian = getGaussianKernel(size, sigma);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i < size / 2 && j < size / 2 || i >= size / 2 && j >= size / 2)
				kernel.at<double>(j, i) = 1.0;
			else
				kernel.at<double>(j, i) = -1.0;
		}
	}

	gaussian = gaussian * gaussian.t();
	kernel = kernel.mul(gaussian);

	return kernel;
}

static vector<double> getNoveltyScores(const vector<int>& times, double K,
		int kernelSize, double kernelSigma)
{
	vector<double> scores(times.size());
	Mat S = getSimilarityMatrix(times, K);
	Mat kernel = getKernel(kernelSize, kernelSigma);

	for (int i = 0; i < times.size(); i++) {
		Mat roi(S, Rect(Point(i, i), Size(1, 1)));
		Mat output;
		filter2D(roi, output, -1, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
		scores[i] = output.at<double>(0, 0);
	}

	return scores;
}

Mat getFeatures(const vector<int>& times)
{
	/*int nK = 10;*/
	/*int nSize = 4;*/
	/*int nSigma = 4;*/
	int nK = 2;
	int nSize = 1;
	int nSigma = 1;
	double K;
	int size;
	double sigma;

	Mat features;
	vector<double> scores;
	for (int k = 0; k < nK; k++) {
		for (int sz = 0; sz < nSize; sz++) {
			for (int sg = 0; sg < nSigma; sg++) {
				K = 10000.0 * (1 + k);
				size = 2 * (1 + sz);
				sigma = 0.5 * sigma + 1;
				scores = getNoveltyScores(times, K, size, sigma);
				Mat f(scores);
				if (features.empty())
					features = f;
				else
					hconcat(features, f, features);
			}
		}
	}
	features.convertTo(features, CV_32F);

	return features;
}

#endif
