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

void parseFile(const char* fileName, vector<PhotoInfo>& result)
{
	fstream file(fileName);
	if (!file)
		return;

	result.clear();

	while (true) {
		string line;
		PhotoInfo info;
		getline(file, line);
		if (file.eof())
			break;
		parse(line, info);
		result.push_back(info);
	}
}
