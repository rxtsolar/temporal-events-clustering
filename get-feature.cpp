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
	
		vector<Mat> features; 
		vector<double> rates;
		Mat gist = getGistFeatures(image);
		Mat hist = getHistogram(image);
		features.push_back(gist);
		rates.push_back(1);
		features.push_back(hist);
		rates.push_back(200);

		Mat finalFeature = blendFeatures(features, rates);

		cout << info[i].name << ' ';
		cout << finalFeature.rows;
		for (int i = 0; i < finalFeature.rows; i++)
			cout << ' ' << finalFeature.at<double>(i, 0);
		cout << endl;
	}

	return 0;
}
