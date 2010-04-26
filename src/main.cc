#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <set>

#include <cuda.h>
#include <cuda_runtime.h>

#include "keyword_util.h"
#include "external_dependency.h"

using namespace std;

map<string, int> T; // Tokens
set<string> K; // Keywords
map<string, set<int> > intersections; // intersections between keyword and tokens
int num_values = 0;

extern int doit();

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
			keyword::split(keyword, SPACE, t);
			for (unsigned int i = 0; i < t.size(); i++) {
				if (t[i].size() <= 1) {
					continue;
				}
				T[t[i]] = T[t[i]] + 1;
			}
			K.insert(keyword);
		}
		getline(in, keyword);
	}
	in.close();

	// remove tokens with not enough intersections
	for (map<string, int>::iterator it = T.begin(); it != T.end(); it++) {
		if (token_reject > 0 && (*it).second <= token_reject) {
			map<string, int>::iterator removeMe(it);
			if (it != T.begin())
				it--;
			T.erase(removeMe);
		}
	}

	// remove keywords with no intersections
	for (set<string>::iterator it = K.begin(); it != K.end(); it++) {
		keyword = *it;
		vector<string> t;
		int c = 0;

		keyword::split(keyword, SPACE, t);
		for (unsigned int i = 0; i < t.size(); i++) {
			if (T.count(t[i]) > 0)
				c++;
		}
		if (c == 0) {
			set<string>::iterator removeMe(it);
			if (it != K.begin())
				it--;
			K.erase(removeMe);
			cout << "rejecting " << keyword << endl;
		}
	}

	// map token intersections
	int n = 0;
	for (set<string>::iterator it = K.begin(); it != K.end(); it++) {
		keyword = *it;
		vector<string> t;
		set<string> keys;
		keyword::split(keyword, SPACE, t);
		for (unsigned int i = 0; i < t.size(); i++)
			keys.insert(t[i]);

		for (map<string, int>::iterator itt = T.begin(); itt != T.end(); itt++) {
			string token = (*itt).first;
			if (keys.count(token) > 0) {
				set<int>& kws = intersections[token];
				kws.insert(n);
				num_values++;
			}
		}
		n++;
	}

	cout << "num tokens: " << T.size() << endl;
	cout << "num keywords: " << K.size() << endl;
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
	out << K.size() << " " << T.size() << " " << num_values << endl;

	int n = 1;
	for (set<string>::iterator it = K.begin(); it != K.end(); it++) {
		keyword = *it;
		vector<string> t;
		set<string> keys;
		keyword::split(keyword, SPACE, t);
		for (unsigned int i = 0; i < t.size(); i++)
			keys.insert(t[i]);

		int m = 1;
		for (map<string, int>::iterator itt = T.begin(); itt != T.end(); itt++) {
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
	create_matirx_market(output_file);

	CHECK_CUDA_ERROR();
	return doit();

	return 0;
}
