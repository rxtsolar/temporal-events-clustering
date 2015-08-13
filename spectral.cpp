#include "feature.h"

using namespace std;
using namespace cv;

const int MIN_NUM_EVENTS = 3;
const int MAX_NUM_EVENTS = 30;

static Mat getSimilarityMatrix(const Mat& features,
		const vector<int>& nFeatures, const vector<double>& rates)
{
	int n = features.rows;
	int count = 0;
	Mat S(n, n, CV_64F, Scalar(0.0));
	vector<Mat> fts;

	for (int i = 0; i < nFeatures.size(); i++) {
		Mat f = features.colRange(count, count + nFeatures[i]);
		fts.push_back(f);
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			for (int k = 0; k < nFeatures.size(); k++) {
				double d = norm(fts[k].row(i) - fts[k].row(j));
				d = exp(-d / rates[k]);
				S.at<double>(i, j) = max(d, S.at<double>(i, j));
			}
		}
	}

	return S;
}

static Mat getLaplacianMatrix(const Mat& S)
{
	int n = S.rows;
	Mat D(1, n, CV_64F, Scalar(0.0));
	Mat L;

	for (int i = 0; i < n; i++)
		D.at<double>(0, i) = sum(S.row(i))[0];
	//pow(D, -0.5, D);
	pow(D, -1, D);
	D = Mat::diag(D);

	//L = Mat::eye(n, n, CV_64F) - D * S * D;
	L = Mat::eye(n, n, CV_64F) - D * S;

	return L;
}

Mat spectralClustering(const Mat& features,
		const vector<int>& nFeatures, const vector<double>& rates)
{
	Mat labels;
	Mat eigenValues;
	Mat eigenVectors;
	Mat S = getSimilarityMatrix(features, nFeatures, rates);
	Mat L = getLaplacianMatrix(S);
	int n = features.rows;
	int k = 0;

	eigen(L, eigenValues, eigenVectors);
	eigenVectors.convertTo(eigenVectors, CV_32F);

	for (int i = eigenValues.rows - 1; i >= 0; i--) {
		if (abs(eigenValues.at<double>(i, 0)) < 1e-5)
			k++;
	}

	if (k < n / MAX_NUM_EVENTS)
		k = n / MAX_NUM_EVENTS;
	if (k == 0)
		k = 1;

	eigenVectors = eigenVectors.rowRange(eigenVectors.rows - k, eigenVectors.rows).t();

	Mat f;
	features.convertTo(f, CV_32F);
	kmeans(f, k, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1e-5),
			2, KMEANS_RANDOM_CENTERS);

	return labels;
}
