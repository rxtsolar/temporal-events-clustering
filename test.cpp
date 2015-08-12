#include <iostream>
#include <fstream>
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
		cerr << "usage: " << argv[0] << " <parsed-data> <model-to-use>" << endl;
		return -1;
	}

	string model(argv[2]);

	vector<PhotoInfo> info;
	parseFile(argv[1], info);

	Mat testingData = getTimeFeatures(info);

	CvSVM svm;
	svm.load(model.c_str());

	for (int i = 0; i < testingData.rows; i++)
		info[i].label = svm.predict(testingData.row(i));

	writeFile(0, info);

	return 0;
}
