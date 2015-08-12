#include "parser.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

void parse(const string& line, PhotoInfo& info)
{
	stringstream ss;
	ss << line;
	ss >> info.label >> info.name >> info.time >> info.orientation >> info.lens;
}

void parseFile(const char* fileName, vector<PhotoInfo>& info)
{
	ifstream file(fileName);
	if (!file)
		return;

	info.clear();

	while (true) {
		string line;
		PhotoInfo i;
		getline(file, line);
		if (file.eof())
			break;
		parse(line, i);
		info.push_back(i);
	}
}

void writeFile(const char* fileName, vector<PhotoInfo>& info)
{
	ofstream file(fileName);
	ostream& os = file ? file : cout;

	for (int i = 0; i < info.size(); i++) {
		os << info[i].label << ' ';
		os << info[i].name << ' ';
		os << info[i].time << ' ';
		os << info[i].orientation << ' ';
		os << info[i].lens << endl;
	}
}
