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
	
		//Mat features = getGistFeatures(image);
		Mat features = getHistogram(image);

		cout << info[i].name << ' ';
		cout << features.rows;
		for (int i = 0; i < features.rows; i++)
			cout << ' ' << features.at<double>(i);
		cout << endl;
	}

	return 0;
}
