/*
 * keyword_util.h
 *
 *  Created on: Apr 26, 2010
 *      Author: amerlo
 */

#ifndef KEYWORD_UTIL_H_
#define KEYWORD_UTIL_H_

#include <vector>
#include <iostream>

const char SPACE = ' ';

namespace keyword {

inline void split(const std::string& line, const char& delimiter, std::vector<std::string>& col) {
	std::istringstream iss(line);
	std::string tok;
	while (std::getline(iss, tok, delimiter)) {
		if (tok.size() <= 1)
			continue;

		col.push_back(tok);
	}
}

}

#endif /* KEYWORD_UTIL_H_ */
