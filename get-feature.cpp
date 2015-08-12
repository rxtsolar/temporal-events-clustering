#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature.h"
#include "parser.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc < 3) {
		cerr << "usage: " << argv[0] << " <input-data> <path-to-photo>" << endl;
		return -1;
	}

	vector<PhotoInfo> info;
	string path(argv[2]);
	parseFile(argv[1], info);

	for (int i = 0; i < info.size(); i++) {
		string full = path + info[i].name;
		Mat image = imread(full, 1);

		preprocess(image, info[i].orientation);
	
		Mat gist = getGistFeatures(image);
		Mat hist = getHistogram(image);

		cout << info[i].name << ' ';
		cout << gist.rows << ' ';
		cout << hist.rows << ' ';
		for (int i = 0; i < gist.rows; i++)
			cout << ' ' << gist.at<double>(i, 0);
		for (int i = 0; i < hist.rows; i++)
			cout << ' ' << hist.at<double>(i, 0);
		cout << endl;
	}

	return 0;
}
