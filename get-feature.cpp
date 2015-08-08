#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc < 2) {
		cerr << "usage: " << argv[0] << " <input-image>" << endl;
		return -1;
	}

	Mat image = imread(argv[1], 1);
	Mat hist = getHistogram(image);
	hist = hist / (image.rows * image.cols);
	cout << hist.rows;
	for (int i = 0; i < hist.rows; i++)
		cout << ' ' << hist.at<float>(i);
	cout << endl;
	return 0;
}
