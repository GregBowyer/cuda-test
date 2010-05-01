/*
 * KeywordMatrix.cpp
 *
 *  Created on: Apr 29, 2010
 *  Author: amerlo
 */

#include "KeywordMatrix.h"

KeywordMatrix::KeywordMatrix(const char* input_file) {
	process_keywords(input_file);
}

KeywordMatrix::~KeywordMatrix() {

}

int KeywordMatrix::get_intersections(const std::map<string, std::set<int> >& intersections, const std::string& t1, const std::string& t2) {
	int n = 0;
	const std::set<int>& k1 = intersections.find(t1)->second;
	const std::set<int>& k2 = intersections.find(t2)->second;
	for (std::set<int>::iterator it = k1.begin(); it != k1.end(); it++) {
		if (k2.count(*it) > 0) {
			n++;
		}
	}

	return n;
}

void split(const std::string& line, const char& delimiter, std::vector<std::string>& col) {
	std::istringstream iss(line);
	std::string tok;
	while (std::getline(iss, tok, delimiter)) {
		if (tok.size() <= 1)
			continue;

		col.push_back(tok);
	}
}

matrix<float> KeywordMatrix::calc_covariance() {
	int wT = tokens.size();
	matrix<float> A(wT, wT);
	int n = 0;
	double K = (float) keywords.size();
	for (std::map<std::string, int>::iterator it = tokens.begin(); it != tokens.end(); it++) {
		int m = 0;
		float t1 = (float) (*it).second;

		for (std::map<std::string, int>::iterator itt = tokens.begin(); itt != tokens.end(); itt++) {
			float t2 = (float) (*itt).second;
			float v = 0;
			if (n >= m) {
				if (n == m) {
					// calculate diagonal
					v = ((pow((1 - t1 / K), 2) * t1) + ((pow((-t1 / K), 2) * (K - t1)))) / K;
				} else {
					int nn = get_intersections(intersections, (*it).first, (*itt).first);
					float t00 = -t1 / K;
					float t01 = 1 - t1 / K;
					float t10 = -t2 / K;
					float t11 = 1 - t2 / K;
					v = ((nn * t01 * t11) + ((t1 - nn) * t01 * t10) + ((t2 - nn) * t00 * t11) + ((K - (t2 + t1 - nn)) * t00 * t10)) / K;
				}
			}
			A(n, m) = v;
			A(m, n) = v;
			m++;
		}
		n++;
	}

	return A;
}

void KeywordMatrix::create_matirx_market(const string& output_file) {
	ofstream out;
	string keyword;

	out.open(output_file.c_str());
	if (!out) {
		cerr << "failed to open file " << output_file << endl;
		exit(-1);
	}

	out << "%%MatrixMarket matrix coordinate real general" << endl;
	out << keywords.size() << " " << tokens.size() << " " << num_values << endl;

	int n = 1;
	for (set<string>::iterator it = keywords.begin(); it != keywords.end(); it++) {
		keyword = *it;
		std::vector<string> t;
		set<string> keys;
		split(keyword, SPACE, t);
		for (unsigned int i = 0; i < t.size(); i++)
			keys.insert(t[i]);

		int m = 1;
		for (map<string, int>::iterator itt = tokens.begin(); itt != tokens.end(); itt++) {
			string token = (*itt).first;
			if (keys.count(token) > 0)
				out << n << " " << m << " 1" << endl;

			m++;
		}
		n++;
	}
	out.close();
}

void KeywordMatrix::process_keywords(const char* input_file) {
	ifstream in;
	string keyword;
	int token_reject = 1;
	num_values = 0;

	in.open(input_file);
	if (!in) {
		cerr << "failed to open file " << input_file << endl;
		exit(-1);
	}

	getline(in, keyword);
	while (!in.eof()) {
		if (keyword.size() > 0) {
			std::vector<string> t;
			split(keyword, SPACE, t);
			for (unsigned int i = 0; i < t.size(); i++) {
				if (t[i].size() <= 1) {
					continue;
				}
				tokens[t[i]] = tokens[t[i]] + 1;
			}
			keywords.insert(keyword);
		}
		getline(in, keyword);
	}
	in.close();

	// remove tokens with not enough intersections
	for (map<string, int>::iterator it = tokens.begin(); it != tokens.end(); it++) {
		if (token_reject > 0 && (*it).second <= token_reject) {
			map<string, int>::iterator removeMe(it);
			if (it != tokens.begin())
				it--;
			tokens.erase(removeMe);
		}
	}

	// remove keywords with no intersections
	for (set<string>::iterator it = keywords.begin(); it != keywords.end(); it++) {
		keyword = *it;
		std::vector<string> t;
		int c = 0;

		split(keyword, SPACE, t);
		for (unsigned int i = 0; i < t.size(); i++) {
			if (tokens.count(t[i]) > 0)
				c++;
		}
		if (c == 0) {
			set<string>::iterator removeMe(it);
			if (it != keywords.begin())
				it--;
			keywords.erase(removeMe);
			cout << "rejecting " << keyword << endl;
		}
	}

	// map token intersections
	int n = 0;
	for (set<string>::iterator it = keywords.begin(); it != keywords.end(); it++) {
		keyword = *it;
		std::vector<string> t;
		set<string> keys;
		split(keyword, SPACE, t);
		for (unsigned int i = 0; i < t.size(); i++)
			keys.insert(t[i]);

		for (map<string, int>::iterator itt = tokens.begin(); itt != tokens.end(); itt++) {
			string token = (*itt).first;
			if (keys.count(token) > 0) {
				set<int>& kws = intersections[token];
				kws.insert(n);
				num_values++;
			}
		}
		n++;
	}

	cout << ">> num tokens: " << tokens.size() << endl;
	cout << ">> num keywords: " << keywords.size() << endl;
	cout << ">> num vals: " << num_values << endl;
}
