#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "keyword_util.h"
#include "external_dependency.h"

using namespace std;

map<string, int> tokens; // Tokens
set<string> keywords; // Keywords
map<string, set<int> > intersections; // intersections between keyword and tokens
int num_values = 0;

extern int doit();

struct Vector {
	int size;
	int* values;
};


int get_intersections(Vector *itrs, int t1, int t2) {
	int n = 0;
	Vector k1 = itrs[t1];
	Vector k2 = itrs[t2];
	for (int i = 0; i < k1.size; i++) {
		int *v1 = k1.values;
		for (int j = 0; j < k2.size; j ++) {
			int *v2 = k2.values;
			if (v1[i] == v2[j])
				n++;
		}
	}

	return n;
}

/*
 int get_intersections (const std::map<string,
                        std::set<int> >& intersections,
                        const std::string& t1, const std::string& t2)
        {
                int n = 0;
                const std::set<int>& k1 = intersections.find(t1)->second;
                const std::set<int>& k2 = intersections.find(t2)->second;
                for (std::set<int>::iterator it = k1.begin(); it != k1.end(); it++) {
                        if (k2.count(*it)>0) {
                                n++;
                        }
                }

                return n;
        }
 */

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

		keyword::split(keyword, SPACE, t);
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
		keyword::split(keyword, SPACE, t);
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
		keyword::split(keyword, SPACE, t);
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
	create_matirx_market(output_file);

	int T = tokens.size();
	int K = keywords.size();
	int c = 0; // temp counter

	int tk[T]; // tokens
	Vector itrs[T]; // intersections

	// map token info to c array
	for (std::map<std::string, int>::iterator it = tokens.begin(); it != tokens.end(); it++) {
		tk[c++] = (float) (*it).second;
	}

	// map intersections info to c array
	c = 0;
	for (map<string, set<int> >::iterator it = intersections.begin(); it != intersections.end(); it++) {
		set<int> tokenSet = (*it).second;

		int *vals;
		vals = (int *) malloc(tokenSet.size() * sizeof(int));
		int d = 0;
		for (set<int>::iterator itt = tokenSet.begin(); itt != tokenSet.end(); itt++) {
			vals[d++] = *itt;
		}

		Vector vec;
		vec.size = tokenSet.size();
		vec.values = vals;

		itrs[c++] = vec;
	}


	for (int i = 0; i < T; i++) {
		float t1 = tk[i];
		for (int j = 0; j < T; j++) {
			float t2 = tk[j];
			float v = 0.0;
			if (i > 0 && j<i) {
				// already calculated
			} else {
				if (i == j) {
					// calculate diagonal
					v = ((pow((1-t1/K),2) * t1) + ((pow((-t1/K),2) * (K - t1)))) / K;
				} else {
                    float nn = (float) get_intersections(itrs, i, j);
                    //float nn = 0;
                    float t00 = -t1/K;
                    float t01 = 1-t1/K;
                    float t10 = -t2/K;
                    float t11 = 1-t2/K;
                    v = ((nn * t01 * t11)
                      + ((t1 - nn) * t01 * t10)
                      + ((t2 - nn) * t00 * t11)
                      + ((K - (t2 + t1 - nn)) * t00 * t10))
                      / K;
				}
			}
			cout << i << ", " << j << ", " << v << endl;
			// A(i, j) = v;
		}
	}

	//CHECK_CUDA_ERROR();
	//return doit();

	return 0;
}
