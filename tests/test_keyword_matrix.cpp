#define BOOST_TEST_MODULE keyword_matrix_test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <boost/numeric/ublas/matrix.hpp>

#include "KeywordMatrix.h"
#include "matrix_util.hpp"

using namespace boost;
using namespace std;

const double EPSILON = 2.0e-4;

BOOST_AUTO_TEST_CASE( keyword_matrix_test )
{
	string input_file = "keywords.txt";
	KeywordMatrix km(input_file.c_str());
	matrix<float> cov2 = km.calc_covariance();

	input_file = "covariance.mm";
	matrix<float> cov1 = load_matrixmarketc(input_file.c_str());

	BOOST_CHECK( km.num_keywords() == 99);
	BOOST_CHECK( km.num_tokens() == 48);
	BOOST_REQUIRE( cov1.size1() == 48 );
	BOOST_REQUIRE( cov1.size2() == 48 );
	BOOST_REQUIRE( cov2.size1() == 48 );
	BOOST_REQUIRE( cov2.size2() == 48 );

	for (unsigned int i = 0; i < cov1.size1(); i++) {
		for (unsigned int j = 0; j < cov1.size2(); j++) {
			BOOST_CHECK_CLOSE(cov1(i, j), cov2(i, j), EPSILON);
		}
	}

}
