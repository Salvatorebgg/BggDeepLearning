#ifndef BGG_MATH_H
#define BGG_MATH_H

/*
 * BggDeepLearning C basic math utilities
 *
 * This file declares simple C functions.
 * Later, these functions can be called by C++, Python, or R.
 */

#ifdef __cplusplus
extern "C" {
#endif

double bgg_add(double a, double b);

double bgg_subtract(double a, double b);

double bgg_multiply(double a, double b);

double bgg_safe_divide(double a, double b, int *error_code);

double bgg_mean(const double *values, int length);

double bgg_risk_score(double age, double heart_rate, double systolic_bp, double lactate);

#ifdef __cplusplus
}
#endif

#endif