#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc < 3) {
		cerr << "usage: " << argv[0] << " <parsed-data> <model-to-use>" << endl;
		return -1;
	}

	fstream file(argv[1]);
	string model(argv[2]);

	if (!file)
		return -1;

	vector<int> labels;
	vector<string> names;
	vector<int> times;

	while (true) {
		int label;
		string name;
		int time;
		file >> label >> name >> time;
		if (file.eof())
			break;
		labels.push_back(label);
		names.push_back(name);
		times.push_back(time);
	}

	Mat testingData = getFeatures(times);

	CvSVM svm;
	svm.load(model.c_str());

	for (int i = 0; i < testingData.rows; i++) {
		cout << svm.predict(testingData.row(i)) << ' ';
		cout << names[i] << ' ' << times[i] << endl;
	}

	return 0;
}
