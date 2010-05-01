/*
 * matrix_util.hpp
 *
 *  Created on: Apr 30, 2010
 *      Author: amerlo
 */

#include <iostream>
/*
 * matrix_util.hpp
 *
 * Utility class to read a matrix market file for the tests
 *
 *  Created on: May 1, 2010
 *  Author: amerlo
 */

#include <sstream>
#include <fstream>

#include <boost/lexical_cast.hpp>
#include <boost/numeric/ublas/matrix.hpp>

using namespace std;
using boost::lexical_cast;
using boost::bad_lexical_cast;

#ifndef MATRIX_UTIL_HPP_
#define MATRIX_UTIL_HPP_

namespace util {

void split(const std::string& line, const char& delimiter, std::vector<std::string>& col) {
	std::istringstream iss(line);
	std::string tok;
	while (std::getline(iss, tok, delimiter)) {
		col.push_back(tok);
	}
}

}

matrix<float> load_matrixmarketc(const char* input_file) {
	ifstream in;
	string row;

	cout << "loading matrix market file " << input_file << endl;
	in.open(input_file);
	if (!in) {
		cerr << "failed to open file " << input_file << endl;
		exit(-1);
	}

	// ignore the first line
	getline(in, row);

	// get size and create matrix
	getline(in, row);
	std::vector<string> r;
	util::split(row, SPACE, r);
	int n = lexical_cast<int> (r[0]);
	int m = lexical_cast<int> (r[1]);
	matrix<float> A(m, n);

	getline(in, row);
	while (!in.eof()) {
		std::vector<string> t;
		util::split(row, SPACE, t);

		int i = lexical_cast<int> (t[0]) - 1;
		int j = lexical_cast<int> (t[1]) - 1;
		float v = lexical_cast<float> (t[2]);

		A(i, j) = v;

		getline(in, row);
	}
	in.close();

	return A;
}

#endif /* MATRIX_UTIL_HPP_ */
