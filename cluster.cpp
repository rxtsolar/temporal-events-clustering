#include "feature.h"

using namespace std;
using namespace cv;

static Mat getSimilarityMatrix(const Mat& features, double K)
{
	int n = features.rows;
	Mat S(n, n, CV_64F);
	
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			double d = norm(features.row(i) - features.row(j));
			S.at<double>(i, j) = exp(-d / K);
		}
	}

	return S;
}

static Mat getLaplacianMatrix(const Mat& features, double K)
{
	int n = features.rows;
	Mat S = getSimilarityMatrix(features, K);
	Mat D(n, n, CV_64FC1);
	Mat L;

	for (int i = 0; i < n; i++)
		D.at<double>(i, i) = sum(S.row(i))[0];
	pow(D, -0.5, D);

	L = Mat::eye(n, n, CV_64F) - D * S * D;

	return L;
}

Mat spectralClustering(const Mat& features)
{
	Mat labels;
	Mat eigenValues;
	Mat eigenVectors;
	Mat laplacian = getLaplacianMatrix(features, 2);

	eigen(laplacian, eigenValues, eigenVectors);
	cerr << eigenValues << endl;
	kmeans(eigenVectors, 2, labels,
			TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 1000, 1e-5),
			2, KMEANS_RANDOM_CENTERS);

	return labels;
}
