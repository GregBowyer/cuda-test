/*
 * KeywordMatrix.h
 *
 *  Created on: Apr 29, 2010
 *      Author: amerlo
 */

#ifndef KEYWORDMATRIX_H_
#define KEYWORDMATRIX_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <math.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost::numeric::ublas;

const char SPACE = ' ';

class KeywordMatrix {

public:
	KeywordMatrix(const char* input_file);
	virtual ~KeywordMatrix();

	matrix<float> calc_covariance();
	void create_matirx_market(const string& output_file);

	map<string, int> get_tokens() {
		return tokens;
	}

	map<string, set<int> > get_intersections() {
		return intersections;
	}

	int num_keywords() {
		return keywords.size();
	}

	int num_tokens() {
		return tokens.size();
	}

private:

	map<string, int> tokens; // Tokens
	set<string> keywords; // Keywords
	map<string, set<int> > intersections; // intersections between keyword and tokens
	int num_values;

	void process_keywords(const char* input_file);
	int get_intersections(const std::map<string, std::set<int> >& intersections, const std::string& t1, const std::string& t2);

};

#endif /* KEYWORDMATRIX_H_ */
