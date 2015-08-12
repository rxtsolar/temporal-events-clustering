#ifndef _PARSER_H_
#define _PARSER_H_

#include <vector>
#include <string>

enum Orientation {
	TOP = 6,
	BOTTOM = 8,
	LEFT = 1,
	RIGHT = 3,
};

struct PhotoInfo {
	std::string name;
	int label;
	int time;
	int orientation;
	int lens;
};

void parse(const std::string& line, PhotoInfo& info);
void parseFile(const char* fileName, std::vector<PhotoInfo>& info);

#endif
