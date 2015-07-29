#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature.h"

using namespace std;
using namespace cv;

vector<bool> labels;
vector<string> names;
vector<int> times;
vector<double> data;

int scale1 = 1;
int shift1 = 0;
int scale2 = 1;
int shift2 = 0;
int KK = 10000;
int kSize = 2;
int kSigma = 1;
double minimum = DBL_MAX;
double maximum = 0;

void plot1(int, void*)
{
	Mat image = Mat::zeros(768, 1024, CV_8UC3);
	int y = image.rows / 2;
	int x = 0;
	int margin = image.cols / 20;

	double rate = (double)(image.cols - 2 * margin) / (maximum - minimum);

	for (int i = 0; i < data.size(); i++) {
		x = margin + rate * scale1 * (data[i] - minimum);
		x -= (double)(scale1 - 1) * (image.cols - 2 * margin) * shift1 / 100;
		if (labels[i])
			circle(image, Point(x, y), 2, CV_RGB(255, 0, 0));
		else
			circle(image, Point(x, y), 2, CV_RGB(255, 255, 0));
	}
	imshow("plot1", image);
}

void plot2(int, void*)
{
	Mat image = Mat::zeros(768, 1024, CV_8UC3);
	int margin = image.cols / 20;
	int x;
	int y;

	vector<double> scores = getNoveltyScores(times, KK, 2 * (kSize + 1), kSigma);
	data = scores;
	//vector<double> peaks = getPeaks(scores);

	for (int i = 0; i < data.size(); i++) {
		if (minimum > data[i])
			minimum = data[i];
		if (maximum < data[i])
			maximum = data[i];
	}

	for (int i = 0; i < data.size(); i++) {
		x = margin + i * scale2 * (image.cols - 2 * margin) / data.size();
		x -= (double)(scale2 - 1) * (image.cols - 2 * margin) * shift2 / 100;
		y = image.rows - margin - (double)(image.rows - 2 * margin) *
			(data[i] - minimum) / (maximum - minimum);
		if (labels[i])
			line(image, Point(x, y), Point(x, image.rows - margin),
					CV_RGB(255, 0, 0));
		else
			line(image, Point(x, y), Point(x, image.rows - margin),
					CV_RGB(255, 255, 0));
	}
	imshow("plot2", image);
}

void draw(void)
{
	namedWindow("plot2");
	createTrackbar("K", "plot2", &KK, 100000, plot2);
	createTrackbar("kSize", "plot2", &kSize, 10, plot2);
	createTrackbar("kSigma", "plot2", &kSigma, 1000, plot2);
	createTrackbar("scale2", "plot2", &scale2, 100, plot2);
	createTrackbar("shift2", "plot2", &shift2, 100, plot2);
	plot2(0, 0);
	while (waitKey() != 27);
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		return -1;

	fstream file(argv[1]);

	if (!file)
		return -1;

	while (true) {
		bool label;
		string name;
		int time;
		file >> label >> name >> time;
		if (file.eof())
			break;
		labels.push_back(label);
		names.push_back(name);
		times.push_back(time);
	}

	draw();

	return 0;
}
