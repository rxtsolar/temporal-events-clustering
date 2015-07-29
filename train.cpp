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
	if (argc < 2)
		return -1;

	fstream file(argv[1]);

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

	// feature params
	int nK = 1;
	int nSize = 1;
	int nSigma = 1;
	//int nK = 10;
	//int nSize = 4;
	//int nSigma = 4;
	double K;
	int size;
	double sigma;

	Mat trainingData;
	vector<double> scores;
	for (int k = 0; k < nK; k++) {
		for (int sz = 0; sz < nSize; sz++) {
			for (int sg = 0; sg < nSigma; sg++) {
				K = 10000.0 * (1 + k);
				size = 2 * (1 + sz);
				sigma = 0.5 * sigma + 1;
				scores = getNoveltyScores(times, K, size, sigma);
				Mat f(scores);
				if (trainingData.empty())
					trainingData = f;
				else
					vconcat(trainingData, f, trainingData);
			}
		}
	}
	trainingData.convertTo(trainingData, CV_32F);

	// svm params
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

	CvSVM svm;
	svm.train(trainingData, Mat(labels), Mat(), Mat(), params);
	svm.save("model.xml");

	return 0;
}
