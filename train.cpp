#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "feature.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc < 5)
		return -1;

	fstream file(argv[1]);
	string model(argv[2]);
	double C = atof(argv[3]);
	double gamma = atof(argv[4]);

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

	Mat trainingData = getFeatures(times);

	// svm params
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	params.C = C;
	params.gamma = gamma;

	CvSVM svm;
	svm.train(trainingData, Mat(labels), Mat(), Mat(), params);
	svm.save(model.c_str());

	return 0;
}
