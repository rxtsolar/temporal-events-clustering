#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include "feature.h"
#include "parser.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	if (argc < 5) {
		cerr << "usage: " << argv[0] << " <labeled-data> <model-to-save> ";
		cerr << "<SVM-C> <SVM-gamma>" << endl;
		return -1;
	}

	string model(argv[2]);
	double C = atof(argv[3]);
	double gamma = atof(argv[4]);

	vector<PhotoInfo> info;
	parseFile(argv[1], info);

	Mat trainingData = getTimeFeatures(info);
	Mat labels = getLabels(info);

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
