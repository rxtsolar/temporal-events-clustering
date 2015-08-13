#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <string>
#include "feature.h"
#include "parser.h"

using namespace std;
using namespace cv;

const double GIST_SIGMA = 10.0;
const double COLOR_SIGMA = 0.015;

const char* modelName = "model/time.xml";

int main(int argc, char* argv[])
{
	if (argc < 2) {
		cerr << "usage: " << argv[0] << " <parsed-data>" << endl;
		return -1;
	}

	vector<PhotoInfo> info;
	vector<vector<PhotoInfo> > events;
	vector<vector<PhotoInfo> > timeGroups;
	parseFile(argv[1], info);

	Mat testingData = getTimeFeatures(info);

	CvSVM svm;
	svm.load(modelName);

	vector<PhotoInfo> group;
	for (int i = 0; i < testingData.rows; i++) {
		info[i].label = svm.predict(testingData.row(i));
		if (info[i].label == 1) {
			timeGroups.push_back(group);
			group.clear();
		}
		group.push_back(info[i]);
	}
	if (!group.empty())
		timeGroups.push_back(group);

	for (int i = 0; i < timeGroups.size(); i++) {
		if (timeGroups[i].size() < MIN_NUM_EVENTS)
			continue;

		Mat features;
		vector<int> nFeatures;
		vector<double> rates;
		map<int, vector<PhotoInfo> > cluster;
		Mat labels;

		rates.push_back(GIST_SIGMA);
		rates.push_back(COLOR_SIGMA);

		for (int j = 0; j < timeGroups[i].size(); j++) {
			Mat image = imread(timeGroups[i][j].name.c_str(), 1);

			preprocess(image, timeGroups[i][j].orientation);

			Mat gist = getGistFeatures(image);
			Mat color = getHistogram(image);

			if (features.empty()) {
				hconcat(gist.t(), color.t(), feautres);
				nFeatures.push_back(gist.rows);
				nFeatures.push_back(color.rows);
			} else {
				Mat temp;
				hconcat(gist.t(), color.t(), temp);
				vconcat(features, temp, features);
			}
		}

		labels = spectralClustering(features, nFeatures, rates);

		for (int j = 0; j < labels.size(); j++) {
			cluster[label[j]].push_back(timeGroups[i][j]);
		}

		for (map<int, vector<PhotoInfo> >::const_iterator it = cluster.begin();
				it != cluster.end(); it++) {
			if (it->second.size() < MIN_NUM_EVENTS)
				continue;
			events.push_back(it->second);
		}
	}

	for (int i = 0; i < events.size(); i++) {
		for (int j = 0; j < events[i].size(); j++) {
			cout << i << ' ' << events[i].size();
			cout << ' ' << events[i][j].name;
		}
		cout << endl;
	}

	//cout << timeGroups.size() << endl;
	//for (int i = 0; i < timeGroups.size(); i++) {
		//cout << "    " << timeGroups[i].size() << endl;
	//}

	return 0;
}
