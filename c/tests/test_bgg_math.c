#include <stdio.h>
#include "bgg_math.h"

int main(void) {
    double a = 10.0;
    double b = 2.0;
    int error_code = 0;

    double values[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    double add_result = bgg_add(a, b);
    double sub_result = bgg_subtract(a, b);
    double mul_result = bgg_multiply(a, b);
    double div_result = bgg_safe_divide(a, b, &error_code);
    double mean_result = bgg_mean(values, 5);

    double risk_score = bgg_risk_score(
        68.0,
        120.0,
        90.0,
        4.2
    );

    printf("============================================================\n");
    printf("BggDeepLearning C module test started\n");
    printf("============================================================\n");

    printf("bgg_add(10, 2) = %.2f\n", add_result);
    printf("bgg_subtract(10, 2) = %.2f\n", sub_result);
    printf("bgg_multiply(10, 2) = %.2f\n", mul_result);
    printf("bgg_safe_divide(10, 2) = %.2f, error_code = %d\n", div_result, error_code);
    printf("bgg_mean([1,2,3,4,5]) = %.2f\n", mean_result);
    printf("demo clinical risk score = %.2f\n", risk_score);

    printf("------------------------------------------------------------\n");
    printf("BggDeepLearning C module test finished successfully\n");
    printf("============================================================\n");

    return 0;
}