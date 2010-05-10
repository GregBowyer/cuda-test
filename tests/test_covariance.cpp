/*
 * test_covariance.cpp
 *
 *  Created on: May 1, 2010
 *      Author: amerlo
 */

#define BOOST_TEST_MODULE test_covariance
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>

#include <boost/numeric/ublas/matrix.hpp>

#include "KeywordMatrix.h"
#include "matrix_util.hpp"

using namespace boost;
using namespace std;

const double EPSILON = 2.0e-4;

extern void covariance(float* result, map<string, int> tokens, map<string, set<int> > intersections, int wK);

BOOST_AUTO_TEST_CASE( test_covariance )
{
	string input_file = "keywords.txt";
	KeywordMatrix km(input_file.c_str());

	input_file = "covariance.mm";
	matrix<float> cov1 = load_matrixmarketc(input_file.c_str());

	int wT = km.num_tokens();
	unsigned int mem_size_result = sizeof(float) * wT * wT;
	float* result = (float*) malloc(mem_size_result);
	memset(result, 0, mem_size_result);

	covariance(result, km.get_tokens(), km.get_intersections(), km.num_keywords());

	for (unsigned int i = 0; i < cov1.size1(); i++) {
		for (unsigned int j = 0; j < cov1.size2(); j++) {
			printf("%u, %u = %f %f\n", i, j, cov1(i, j), result[i + (j * wT)]);
			BOOST_CHECK_CLOSE(cov1(i, j), result[i + (j * wT)], EPSILON);
		}
	}

}
