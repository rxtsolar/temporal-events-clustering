#include "feature.h"

using namespace std;
using namespace cv;

const int MIN_NUM_EVENTS = 3;
const int MAX_NUM_EVENTS = 30;

static Mat getSimilarityMatrix(const Mat& features,
		int n1, int n2, double r1, double r2)
{
	int n = features.rows;
	Mat S(n, n, CV_64F, Scalar(0.0));

	Mat feature1 = features.colRange(0, n1);
	Mat feature2 = features.colRange(n1, n1 + n2);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double d1 = norm(feature1.row(i) - feature1.row(j));
			double d2 = norm(feature2.row(i) - feature2.row(j));
			d1 = exp(-d1 / r1);
			d2 = exp(-d2 / r2);
			S.at<double>(i, j) = max(d1, d2);
		}
	}

	cerr << S << endl << endl;

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

	cerr << L << endl << endl;

	return L;
}

Mat spectralClustering(const Mat& features,
		int n1, int n2, double r1, double r2)
{
	Mat labels;
	Mat eigenValues;
	Mat eigenVectors;
	Mat S = getSimilarityMatrix(features, n1, n2, r1, r2);
	Mat L = getLaplacianMatrix(S);
	int n = features.rows;
	int k = 0;

	eigen(L, eigenValues, eigenVectors);
	eigenVectors.convertTo(eigenVectors, CV_32F);

	cerr << eigenValues << endl;

	for (int i = eigenValues.rows - 1; i >= 0; i--) {
		if (abs(eigenValues.at<double>(i, 0)) < 1e-5)
			k++;
	}

	if (k < n / MAX_NUM_EVENTS)
		k = n / MAX_NUM_EVENTS;
	if (k == 0)
		k = 1;

	cerr << "k = " << k << endl;

	eigenVectors = eigenVectors.rowRange(eigenVectors.rows - k, eigenVectors.rows).t();

	//kmeans(eigenVectors, k, labels,
	//TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1e-5),
	//2, KMEANS_RANDOM_CENTERS);
	Mat f;
	features.convertTo(f, CV_32F);
	kmeans(f, k, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1e-5),
			2, KMEANS_RANDOM_CENTERS);

	return labels;
}
