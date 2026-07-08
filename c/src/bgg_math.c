#include "bgg_math.h"

/*
 * Add two numbers.
 */
double bgg_add(double a, double b) {
    return a + b;
}

/*
 * Subtract b from a.
 */
double bgg_subtract(double a, double b) {
    return a - b;
}

/*
 * Multiply two numbers.
 */
double bgg_multiply(double a, double b) {
    return a * b;
}

/*
 * Safe division.
 *
 * error_code:
 * 0 = success
 * 1 = division by zero
 */
double bgg_safe_divide(double a, double b, int *error_code) {
    if (b == 0.0) {
        if (error_code != 0) {
            *error_code = 1;
        }
        return 0.0;
    }

    if (error_code != 0) {
        *error_code = 0;
    }

    return a / b;
}

/*
 * Calculate mean value of an array.
 *
 * If the input is invalid, return 0.0.
 */
double bgg_mean(const double *values, int length) {
    int i;
    double sum = 0.0;

    if (values == 0 || length <= 0) {
        return 0.0;
    }

    for (i = 0; i < length; i++) {
        sum += values[i];
    }

    return sum / length;
}

/*
 * A very simple clinical risk score demo.
 *
 * This is NOT a real medical model.
 * It is only used to test the C module.
 *
 * Inputs:
 * age         : patient age
 * heart_rate  : heart rate
 * systolic_bp : systolic blood pressure
 * lactate     : lactate level
 */
double bgg_risk_score(double age, double heart_rate, double systolic_bp, double lactate) {
    double score = 0.0;

    score += age * 0.01;
    score += heart_rate * 0.02;
    score -= systolic_bp * 0.015;
    score += lactate * 0.50;

    if (score < 0.0) {
        score = 0.0;
    }

    return score;
}