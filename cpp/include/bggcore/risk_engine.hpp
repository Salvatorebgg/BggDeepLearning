#ifndef BGGCORE_RISK_ENGINE_HPP
#define BGGCORE_RISK_ENGINE_HPP

/*
 * BggDeepLearning C++ risk engine
 *
 * This module is a minimal C++ demo for clinical risk calculation.
 * It is NOT a real medical model.
 *
 * Later, this module can be extended for:
 * 1. High-performance patient cohort simulation
 * 2. C++ inference engine
 * 3. Python bindings
 * 4. R bindings
 * 5. Image preprocessing acceleration
 */

#include <string>
#include <vector>

namespace bggcore {

struct PatientVitals {
    double age;
    double heart_rate;
    double systolic_bp;
    double lactate;
    double oxygen_saturation;
};

class RiskEngine {
public:
    RiskEngine();

    double compute_linear_score(const PatientVitals& patient) const;

    double compute_probability(const PatientVitals& patient) const;

    std::string classify_risk_level(double probability) const;

    std::vector<double> batch_predict(const std::vector<PatientVitals>& patients) const;

private:
    double sigmoid(double value) const;
};

double compute_demo_risk_probability(
    double age,
    double heart_rate,
    double systolic_bp,
    double lactate,
    double oxygen_saturation
);

}  // namespace bggcore

#endif