# Unit tests for cuda-test libraries

include_directories(../src)

add_executable(test_keyword_matrix test_keyword_matrix.cpp)
add_executable(test_covariance test_covariance.cpp)

target_link_libraries(test_keyword_matrix ${Boost_LIBRARIES} KeywordMatrix Covariance)
target_link_libraries(test_covariance ${Boost_LIBRARIES} KeywordMatrix Covariance)

add_test(test_keyword_matrix test_keyword_matrix)
add_test(test_covariance test_covariance)
