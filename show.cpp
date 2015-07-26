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

void plot(int, void*)
{
	Mat image = Mat::zeros(768, 1024, CV_8UC3);
	int y = image.rows / 2;
	int x = 0;
	int margin = image.cols / 20;

	int minimum = UINT_MAX;
	int maximum = 0;

	for (int i = 0; i < times.size(); i++) {
		if (minimum > times[i])
			minimum = times[i];
		if (maximum < times[i])
			maximum = times[i];
	}
	double rate = (double)(image.cols - 2 * margin) / (maximum - minimum);

	for (int i = 0; i < times.size(); i++) {
		int x = margin + rate * scale * (times[i] - minimum);
		x -= (double)(scale - 1) * (image.cols - 2 * margin) * shift / 100;
		circle(image, Point(x, y), 2, CV_RGB(255, 255, 0));
	}
	imshow("plot", image);
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

	namedWindow("plot");
	createTrackbar("scale", "plot", &scale, 100, plot);
	createTrackbar("shift", "plot", &shift, 100, plot);
	plot(0, 0);
	while (waitKey() != 27);

	return 0;
}
