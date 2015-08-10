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

int countFaces(const Mat& image)
{
	vector<Rect> faces;
	Mat rotation = image;
	int number = 0;

	CascadeClassifier classifier;
	classifier.load(path);

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
