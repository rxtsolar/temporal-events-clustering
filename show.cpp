#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<string> names;
vector<unsigned int> times;
int scale = 1;
int shift = 0;
unsigned int minimum = UINT_MAX;
unsigned int maximum = 0;

void plot1(int, void*)
{
	Mat image = Mat::zeros(768, 1024, CV_8UC3);
	int y = image.rows / 2;
	int x = 0;
	int margin = image.cols / 20;

	double rate = (double)(image.cols - 2 * margin) / (maximum - minimum);

	for (int i = 0; i < times.size(); i++) {
		int x = margin + rate * scale * (times[i] - minimum);
		x -= (double)(scale - 1) * (image.cols - 2 * margin) * shift / 100;
		circle(image, Point(x, y), 2, CV_RGB(255, 255, 0));
	}
	imshow("plot1", image);
}

void plot2(void)
{
	Mat image = Mat::zeros(768, 1024, CV_8UC3);
	int margin = image.cols / 20;
	int x;
	int y;

	for (int i = 0; i < times.size(); i++) {
		x = margin + i * (image.cols - 2 * margin) / times.size();
		y = image.rows - margin - (double)(image.rows - 2 * margin) *
			(times[i] - minimum) / (maximum - minimum);
		circle(image, Point(x, y), 2, CV_RGB(255, 255, 0));
	}
	imshow("plot2", image);
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		return -1;

	fstream file(argv[1]);

	if (!file)
		return -1;

	while (true) {
		string name;
		int time;
		file >> name >> time;
		if (file.eof())
			break;
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
	createTrackbar("scale", "plot1", &scale, 100, plot1);
	createTrackbar("shift", "plot1", &shift, 100, plot1);
	plot1(0, 0);
	plot2();
	while (waitKey() != 27);

	return 0;
}
