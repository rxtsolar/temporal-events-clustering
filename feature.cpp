#include "feature.h"
#include "parser.h"

using namespace std;
using namespace cv;

static Mat getSimilarityMatrix(const vector<PhotoInfo>& info, double K)
{
	Mat S(info.size(), info.size(), CV_64F);

	for (int i = 0; i < info.size(); i++) {
		for (int j = 0; j < info.size(); j++) {
			S.at<double>(j, i) = exp(-fabs(info[i].time - info[j].time) / K);
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

vector<double> getNoveltyScores(const vector<PhotoInfo>& info, double K,
		int kernelSize, double kernelSigma)
{
	vector<double> scores(info.size());
	Mat S = getSimilarityMatrix(info, K);
	Mat kernel = getKernel(kernelSize, kernelSigma);

	for (int i = 0; i < info.size(); i++) {
		Mat roi(S, Rect(Point(i, i), Size(1, 1)));
		Mat output;
		filter2D(roi, output, -1, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
		scores[i] = output.at<double>(0, 0);
	}

	return scores;
}

static Mat getNeighbors(const vector<double>& scores, int n)
{
	Mat features;
	for (int i = 0; i < scores.size(); i++) {
		vector<double> f;
		for (int j = -n; j <= n; j++) {
			int k = i + j;
			if (k < 0)
				k = 0;
			if (k >= scores.size())
				k = scores.size() - 1;
			f.push_back(scores[k]);
		}
		if (features.empty())
			features = Mat(f).t();
		else
			vconcat(features, Mat(f).t(), features);
	}
	return features;
}

Mat getTimeFeatures(const vector<PhotoInfo>& info)
{
	int nK = 10;
	int nSize = 4;
	int nSigma = 4;
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
				scores = getNoveltyScores(info, K, size, sigma);
				Mat f = getNeighbors(scores, 3);
				if (features.empty())
					f.copyTo(features);
				else
					hconcat(features, f, features);
			}
		}
	}
	features.convertTo(features, CV_32F);

	return features;
}

Mat getLabels(const vector<PhotoInfo>& info)
{
	Mat result;
	vector<int> labels;
	for (int i = 0; i < info.size(); i++)
		labels.push_back(info[i].label);

	Mat(labels).copyTo(result);
	return result;
}
