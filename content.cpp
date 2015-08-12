#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature.h"
#include "parser.h"

using namespace std;
using namespace cv;

const int HIST_SIZE = 64;
const int SMALL_WIDTH = 300;
//const char* path = "model/haarcascade_frontalface_default.xml";
const char* path = "model/lbpcascade_profileface.xml";

Mat getHistogram(const Mat& image)
{
	vector<Mat> channels;
	Mat hist;
	Mat result;

	split(image, channels);

	for (int i = 0; i < channels.size(); i++) {
		calcHist(&channels[i], 1, 0, Mat(), hist, 1, &HIST_SIZE, 0);
		if (result.empty())
			hist.copyTo(result);
		else
			vconcat(result, hist, result);
	}

	result.convertTo(result, CV_64F);
	result /= image.rows * image.cols;
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

Mat getDCTHist(const Mat& image)
{
	vector<Mat> channels;
	int bSize = 8;
	int w = image.cols / bSize;
	int h = image.rows / bSize;
	vector<double> features(bSize * bSize * channels.size(), 0.0);
	Mat result;

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

	Mat(features).copyTo(result);
	result /= w * h;
	return result;
}

Mat getGistFeatures(const Mat& image)
{
	vector<Mat> channels;
	int nScale = 3;
	int nOrient = 8;
	int nBlock = 2;
	vector<double> features;
	Mat result;

	int kSize = min(image.rows, image.cols) / 4;
	int w = image.cols / nBlock;
	int h = image.rows / nBlock;
	double sigma;
	double lambd;
	double theta;
	double gamma = 1.0;

	split(image, channels);

	//for (int c = 0; c < channels.size(); c++) {
		//for (int i = 0; i < nBlock; i++) {
			//for (int j = 0; j < nBlock; j++) {
				//Mat roi(channels[c], Rect(Point(i * w, j * h), Size(w, h)));
				//features.push_back(mean(roi)[0]);
			//}
		//}
	//}

	//for (int i = 0; i < channels.size(); i++)
	//equalizeHist(channels[i], channels[i]);

	for (int s = 0; s < nScale; s++) {
		sigma = kSize * 0.12;
		//lambd = kSize * 0.05;
		lambd = kSize * 0.12;
		for (int o = 0; o < nOrient; o++) {
			double theta = CV_PI * 2 * o / nOrient;
			Mat kernel = getGaborKernel(Size(kSize, kSize),
					sigma, theta, lambd, gamma);
			Mat featureMap;

			for (int c = 0; c < channels.size(); c++) {
				filter2D(channels[c], featureMap, -1, kernel,
						Point(-1, -1), 0, BORDER_REPLICATE);

				//Mat show;
				//normalize(featureMap, show, 0, 1.0, CV_MINMAX, CV_64F);
				//show.convertTo(show, CV_8U, 255.0);
				//imshow("kernel", show);
				//imshow("image", channels[c]);
				//waitKey(0);

				for (int i = 0; i < nBlock; i++) {
					for (int j = 0; j < nBlock; j++) {
						Mat roi(featureMap, Rect(Point(i * w, j * h), Size(w, h)));
						features.push_back(mean(roi)[0]);
					}
				}
			}
		}
		kSize *= 0.7;
	}

	Mat(features).copyTo(result);
	return result;
}

Mat blendFeatures(const vector<Mat>& features, const vector<double>& rates)
{
	Mat blend;
	double sum = 0.0;
	for (int i = 0; i < features.size(); i++) {
		if (blend.empty())
			blend = features[i] * rates[i];
		else
			vconcat(blend, features[i] * rates[i], blend);
		sum += rates[i];
	}
	blend /= sum;
	return blend;
}

void preprocess(Mat& image, int orientation)
{
	int small = min(image.rows, image.cols);
	double rate = static_cast<double>(SMALL_WIDTH) / small;

	resize(image, image, Size(image.cols * rate, image.rows * rate));

	switch (orientation) {
	case TOP:
		transpose(image, image);
		flip(image, image, 1);
		break;
	case BOTTOM:
		transpose(image, image);
		flip(image, image, 0);
		break;
	case LEFT:
		break;
	case RIGHT:
		flip(image, image, -1);
		break;
	default:
		break;
	}
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
