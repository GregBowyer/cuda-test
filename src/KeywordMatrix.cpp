/*
 * KeywordMatrix.cpp
 *
 *  Created on: Apr 29, 2010
 *  Author: amerlo
 */

#include "KeywordMatrix.h"
#include "timer.h"

KeywordMatrix::KeywordMatrix(const char* input_file) {
	process_keywords(input_file);
}

KeywordMatrix::~KeywordMatrix() {

}

float count_intersections(int* intr, int t1, int t2, int wI) {
	int n = 0;
	for (int i = 0; i < wI; i++) {
		int x1 = (t1 * wI) + i;
		if (intr[x1] == -1)
			break;

		for (int j = 0; j < wI; j++) {
			int x2 = (t2 * wI) + j;
			if (intr[x2] == -1)
				break;

			if (intr[x1] == intr[x2])
				n++;
		}
	}

	return (float) n;
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
	CUTimer *timer = start_timing("CPU covariance calculation");
	unsigned int wT = tokens.size();
	matrix<float> A(wT, wT);

	// map intersections info to c array
	unsigned int wI = 0;
	for (map<string, set<int> >::iterator it = intersections.begin(); it != intersections.end(); it++) {
		unsigned int s = ((*it).second).size();
		if (s > wI)
			wI = s;
	}

	int index = 0;
	size_t mem_size_I = sizeof(int) * wT * wI;
	int* intr = (int*) malloc(mem_size_I);
	for (map<string, set<int> >::iterator it = intersections.begin(); it != intersections.end(); it++) {
		set<int> tokenSet = (*it).second;
		for (set<int>::iterator itt = tokenSet.begin(); itt != tokenSet.end(); itt++) {
			intr[index++] = *itt;
		}

		// pad with -1
		if (tokenSet.size() < wI) {
			for (unsigned int i = 0; i < wI - tokenSet.size(); i++)
				intr[index++] = -1;
		}
	}

	unsigned int sizeT = wT * wT;
	size_t mem_size_intr = sizeof(int) * sizeT;
	int* iintr = (int*) calloc(sizeT, mem_size_intr);
	for (unsigned int i = 0; i<wT; i++) {
		for (unsigned int j = 0; j < wT; j++) {
			if (i != j && i < j) {
				int v = count_intersections(intr, i, j, wI);
				iintr[i + j * wT] = v;
			}

		}
	}

	int n = 0;
	double K = (float) keywords.size();
	for (std::map<std::string, int>::iterator it = tokens.begin(); it != tokens.end(); it++) {
		//cout << (*it).first << endl;
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
					//float nn = count_intersections(intr, n, m, wI);
					float nn = (float) iintr[m + n * wT];
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

	/*
	for (int i = 0; i<wT; i++) {
		for (int j = 0; j < wT; j++) {
			int v = count_intersections(intr, i, j, wI);
			//cout << v << "\t";
		}
		//cout << endl;
	}
	*/

	finish_timing(timer);

	return A;
}

void KeywordMatrix::create_matirx_market(const string& output_file) {
	CUTimer *timer = start_timing("Matrix Market File Creation");
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

	finish_timing(timer);
}

void KeywordMatrix::process_keywords(const char* input_file) {
	CUTimer *timer = start_timing("Keyword Processing");
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
			//cout << "rejecting " << keyword << endl;
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

	finish_timing(timer);
}
