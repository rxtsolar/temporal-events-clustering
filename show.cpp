#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<bool> labels;
vector<string> names;
vector<double> times;
int scale1 = 1;
int shift1 = 0;
int scale2 = 1;
int shift2 = 0;
double minimum = DBL_MAX;
double maximum = 0;

void plot1(int, void*)
{
	Mat image = Mat::zeros(768, 1024, CV_8UC3);
	int y = image.rows / 2;
	int x = 0;
	int margin = image.cols / 20;

	double rate = (double)(image.cols - 2 * margin) / (maximum - minimum);

	for (int i = 0; i < times.size(); i++) {
		x = margin + rate * scale1 * (times[i] - minimum);
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

	for (int i = 0; i < times.size(); i++) {
		x = margin + i * scale2 * (image.cols - 2 * margin) / times.size();
		x -= (double)(scale2 - 1) * (image.cols - 2 * margin) * shift2 / 100;
		y = image.rows - margin - (double)(image.rows - 2 * margin) *
			(times[i] - minimum) / (maximum - minimum);
		if (labels[i])
			line(image, Point(x, y), Point(x, image.rows - margin),
					CV_RGB(255, 0, 0));
		else
			line(image, Point(x, y), Point(x, image.rows - margin),
					CV_RGB(255, 255, 0));
	}
	imshow("plot2", image);
}

int main(int argc, char* argv[])
{
	if (argc < 2) {
		cerr << "usage: " << argv[0] << " <labeled-data>" << endl;
		return -1;
	}

	fstream file(argv[1]);

	if (!file)
		return -1;

	while (true) {
		bool label;
		string name;
		double time;
		file >> label >> name >> time;
		if (file.eof())
			break;
		labels.push_back(label);
		names.push_back(name);
		times.push_back(time);
	}

	for (int i = 0; i < times.size(); i++) {
		if (minimum > times[i])
			minimum = times[i];
		if (maximum < times[i])
			maximum = times[i];
	}

	namedWindow("plot1");
	namedWindow("plot2");
	createTrackbar("scale1", "plot1", &scale1, 100, plot1);
	createTrackbar("shift1", "plot1", &shift1, 100, plot1);
	createTrackbar("scale2", "plot2", &scale2, 100, plot2);
	createTrackbar("shift2", "plot2", &shift2, 100, plot2);
	plot1(0, 0);
	plot2(0, 0);
	while (waitKey() != 27);

	return 0;
}
