#include <iostream>
#include <fstream>

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

	vector<string> names;
	Mat features;
	Mat labels;

	while (true) {
		string name;
		int n;
		double f;
		vector<double> feature;

		file >> name >> n;
		if (file.eof())
			break;
		for (int i = 0; i < n; i++) {
			file >> f;
			feature.push_back(f);
		}

		names.push_back(name);
		if (features.empty())
			features = Mat(feature).t();
		else
			vconcat(features, Mat(feature).t(), features);
	}

	labels = spectralClustering(features, 0.2, 0.3);

	for (int i = 0; i < names.size(); i++)
		cout << labels.at<int>(i, 0) << ' ' << names[i] << endl;

	return 0;
}
