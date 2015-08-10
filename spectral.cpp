#include "feature.h"

using namespace std;
using namespace cv;

static Mat getSimilarityMatrix(const Mat& features, double K, double thresh)
{
	int n = features.rows;
	Mat S(n, n, CV_64F, Scalar(0.0));
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double d = norm(features.row(i) - features.row(j));
			d = exp(-d / K);
			if (d < thresh)
				d = 0;
			S.at<double>(i, j) = d;
		}
	}

	cerr << S << endl << endl;

	return S;
}

static Mat getLaplacianMatrix(const Mat& features, double K, double thresh)
{
	int n = features.rows;
	Mat S = getSimilarityMatrix(features, K, thresh);
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

Mat spectralClustering(const Mat& features, double K, double thresh)
{
	Mat labels;
	Mat eigenValues;
	Mat eigenVectors;
	Mat laplacian = getLaplacianMatrix(features, K, thresh);
	int k = 0;

	eigen(laplacian, eigenValues, eigenVectors);
	eigenVectors.convertTo(eigenVectors, CV_32F);

	cerr << eigenValues << endl;

	for (int i = eigenValues.rows - 1; i >= 0; i--) {
		if (abs(eigenValues.at<double>(i, 0)) < 1e-5)
			k++;
	}

	cerr << "k = " << k << endl;

	eigenVectors = eigenVectors.rowRange(eigenVectors.rows - k, eigenVectors.rows).t();

	kmeans(eigenVectors, k, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1e-5),
			2, KMEANS_RANDOM_CENTERS);

	return labels;
}
