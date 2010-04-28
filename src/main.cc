#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <set>

#include "external_dependency.h"

using namespace std;

const char SPACE = ' ';

map<string, int> tokens; // Tokens
set<string> keywords; // Keywords
map<string, set<int> > intersections; // intersections between keyword and tokens
int num_values = 0;

extern int covariance(map<string, int> tokens, map<string, set<int> > intersections, int wK);

void split(const std::string& line, const char& delimiter, std::vector<std::string>& col) {
	std::istringstream iss(line);
	std::string tok;
	while (std::getline(iss, tok, delimiter)) {
		if (tok.size() <= 1)
			continue;

		col.push_back(tok);
	}
}

void process_keywords(const string& input_file) {
	ifstream in;
	string keyword;
	int token_reject = 1;

	in.open(input_file.c_str());
	if (!in) {
		cerr << "failed to open file " << input_file << endl;
		exit(-1);
	}

	getline(in, keyword);
	while (!in.eof()) {
		if (keyword.size() > 0) {
			vector<string> t;
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
		vector<string> t;
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
		vector<string> t;
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

	cout << "num tokens: " << tokens.size() << endl;
	cout << "num keywords: " << keywords.size() << endl;
	cout << "num vals=" << num_values << endl;
}

void create_matirx_market(const string& output_file) {
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
		vector<string> t;
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

int main(int argc, char **argv) {

	string input_file = "/home/amerlo/workspace/cuda-test/dry_food.txt";
	string output_file = "/home/amerlo/workspace/cuda-test/matrix.mm";

	process_keywords(input_file);
	//create_matirx_market(output_file);
	covariance(tokens, intersections, keywords.size());

	return 0;
}
