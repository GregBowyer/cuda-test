# Unit tests for cuda-test libraries

include_directories(../src)

add_executable(test_keyword_matrix 
    test_keyword_matrix.cpp
)

target_link_libraries(test_keyword_matrix ${Boost_LIBRARIES} KeywordMatrix)

add_test(test_keyword_matrix test_keyword_matrix)




