#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cmath>

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

void draw(const vector<double>& d)
{
	data = d;

	for (int i = 0; i < data.size(); i++) {
		if (minimum > data[i])
			minimum = data[i];
		if (maximum < data[i])
			maximum = data[i];
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
}

Mat getSimilarityMatrix(const vector<int>& times, double K)
{
	Mat S(times.size(), times.size(), CV_64F);

	for (int i = 0; i < times.size(); i++) {
		for (int j = 0; j < times.size(); j++) {
			S.at<double>(j, i) = exp(-fabs(times[i] - times[j]) / K);
		}
	}

	return S;
}

Mat getKernel(int size, double sigma)
{
	Mat kernel(size, size, CV_64F);
	Mat gaussian = getGaussianKernel(size, sigma);

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i < size / 2 && j < size / 2 || i >= size / 2 && j >= size / 2)
				kernel.at<double>(j, i) = 1.0;
			else
				kernel.at<double>(j, i) = -1.0;
		}
	}

	gaussian = gaussian * gaussian.t();
	kernel = kernel.mul(gaussian);

	return kernel;
}

vector<double> getNoveltyScores(const vector<int>& times, double K,
		int kernelSize, double kernelSigma)
{
	vector<double> scores(times.size());
	Mat S = getSimilarityMatrix(times, K);
	Mat kernel = getKernel(kernelSize, kernelSigma);

	for (int i = 0; i < times.size(); i++) {
		Mat roi(S, Rect(Point(i, i), Size(1, 1)));
		Mat output;
		filter2D(roi, output, -1, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
		scores[i] = output.at<double>(0, 0);
	}

	//Mat output;
	//filter2D(S, output, -1, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
	//for (int i = 0; i < times.size(); i++) {
		//scores[i] = output.at<double>(i, i);
	//}

	return scores;
}

vector<double> getPeaks(const vector<double>& scores)
{
	vector<double> result;
	Mat s(scores);
	Mat d;
	Mat kernel(3, 1, CV_64F);
	//Mat gaussian = getGaussianKernel(kernelSize, kernelSigma);
	kernel.at<double>(0, 0) = -1;
	kernel.at<double>(1, 0) = 2;
	kernel.at<double>(2, 0) = -1;
	//kernel = kernel.mul(gaussian);
	s = s.mul(1000);
	filter2D(s, d, -1, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
	for (int i = 0; i < scores.size(); i++) {
		result.push_back(d.at<double>(i, 0));
	}
	return result;
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

	vector<double> scores = getNoveltyScores(times, 1, 4, 1);
	vector<double> peaks = getPeaks(scores);

	draw(scores);

	return 0;
}
