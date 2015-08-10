#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature.h"

using namespace std;
using namespace cv;

int histSize = 16;
//const char* path = "model/haarcascade_frontalface_default.xml";
const char* path = "model/lbpcascade_profileface.xml";

Mat getHistogram(const Mat& image)
{
	vector<Mat> channels;
	Mat hist;
	Mat result;

	split(image, channels);

	for (int i = 0; i < channels.size(); i++) {
		calcHist(&channels[i], 1, 0, Mat(), hist, 1, &histSize, 0);
		if (result.empty())
			hist.copyTo(result);
		else
			vconcat(result, hist, result);
	}

	return result;
}

Mat getHistImage(const Mat& histogram)
{
	Mat hist;
	Mat histImage = Mat::ones(600, 1200, CV_8U) * 255;
	normalize(histogram, hist, 0, histImage.rows, CV_MINMAX, CV_32F);

	histImage = Scalar::all(255);
	int binW = cvRound(static_cast<double>(histImage.cols / hist.rows));

	for (int i = 0; i < hist.rows; i++) {
		rectangle(histImage, Point(i * binW, histImage.rows),
				Point((i + 1) * binW, histImage.rows - cvRound(hist.at<float>(i))),
				Scalar::all(0), -1, 8, 0);
	}

	return histImage;
}

Mat getDCTHist(const Mat& big)
{
	vector<Mat> channels;
	int bSize = 8;
	int w = big.cols / bSize;
	int h = big.rows / bSize;
	vector<double> features(bSize * bSize * channels.size(), 0.0);
	Mat image = big;
	Mat result;

	while (image.rows * image.cols > 500000)
		pyrDown(image, image, Size(image.cols / 2, image.rows / 2));

	split(image, channels);

	for (int c = 0; c < channels.size(); c++) {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				Mat roi(channels[c], Rect(Point(j * w, i * h), Size(bSize, bSize)));
				Mat dctImage;
				dct(roi, dctImage);
				for (int k = 0; k < features.size(); k++) {
					features[c * bSize * bSize + k] += dctImage.at<double>(k / bSize, k % bSize);
				}
			}
		}
	}

	result = Mat(features);
	result /= w * h;
	return result;
}

Mat getGistFeatures(const Mat& big)
{
	vector<Mat> channels;
	int nScale = 3;
	int nOrient = 8;
	int nBlock = 4;
	vector<double> features;
	Mat image = big;

	while (image.rows * image.cols > 500000)
		pyrDown(image, image, Size(image.cols / 2, image.rows / 2));

	int kSize = min(image.rows, image.cols) / 8;
	int w = image.cols / nBlock;
	int h = image.rows / nBlock;
	double sigma;
	double lambd;
	double theta;
	double gamma = 1.0;

	split(image, channels);

	for (int s = 0; s < nScale; s++) {
		sigma = kSize * 0.12;
		lambd = kSize * 0.05;
		for (int o = 0; o < nOrient; o++) {
			double theta = CV_PI * 2 * o / nOrient;
			Mat kernel = getGaborKernel(Size(kSize, kSize),
					sigma, theta, lambd, gamma);
			Mat featureMap;

			for (int c = 0; c < channels.size(); c++) {
				filter2D(channels[c], featureMap, -1, kernel,
						Point(-1, -1), 0, BORDER_REPLICATE);
				for (int i = 0; i < nBlock; i++) {
					for (int j = 0; j < nBlock; j++) {
						Mat roi(featureMap, Rect(Point(i * w, j * h), Size(w, h)));
						features.push_back(mean(roi)[0]);
					}
				}
			}
		}
	}

	//normalize(kernel, kernel, 0, 1.0, CV_MINMAX, CV_64F);
	//kernel.convertTo(kernel, CV_8U, 255.0);

	//imshow("kernel", kernel);
	//waitKey(0); } kSize *= 0.7;

	return Mat(features);
}

int countFaces(const Mat& image)
{
	vector<Rect> faces;
	Mat rotation;
	int number = 0;

	CascadeClassifier classifier;
	classifier.load(path);

	cvtColor(image, rotation, CV_BGR2GRAY);
	equalizeHist(rotation, rotation);

	classifier.detectMultiScale(rotation, faces);
	number = faces.size();
	for (int i = 0; i < 3; i++) {
		transpose(rotation, rotation);
		flip(rotation, rotation, 1);
		classifier.detectMultiScale(rotation, faces);
		number += faces.size();
	}

	return faces.size();
}
