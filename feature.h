#ifndef _FEATURE_H_
#define _FEATURE_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

Mat getSimilarityMatrix(const vector<int>& times, double K)
{
	Mat S(times.size(), times.size(), CV_64F);

	for (int i = 0; i < times.size(); i++) {
		for (int j = 0; j < times.size(); j++) {
			S.at<double>(j, i) = exp(-fabs(times[i] - times[j]) / K);
		}
	}

	return S;
}

Mat getKernel(int size, double sigma)
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

vector<double> getNoveltyScores(const vector<int>& times, double K,
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

#endif
