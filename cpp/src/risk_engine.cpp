#include "bggcore/risk_engine.hpp"

#include <cmath>
#include <stdexcept>

namespace bggcore {

RiskEngine::RiskEngine() = default;

double RiskEngine::sigmoid(double value) const {
    return 1.0 / (1.0 + std::exp(-value));
}

double RiskEngine::compute_linear_score(const PatientVitals& patient) const {
    /*
     * This is only a demo formula.
     * It is NOT a validated medical model.
     *
     * Higher risk:
     * - older age
     * - higher heart rate
     * - lower systolic blood pressure
     * - higher lactate
     * - lower oxygen saturation
     */

    double score = -5.0;

    score += 0.025 * patient.age;
    score += 0.018 * patient.heart_rate;
    score -= 0.020 * patient.systolic_bp;
    score += 0.550 * patient.lactate;
    score -= 0.030 * patient.oxygen_saturation;

    return score;
}

double RiskEngine::compute_probability(const PatientVitals& patient) const {
    if (patient.age < 0.0) {
        throw std::invalid_argument("Age cannot be negative.");
    }

    if (patient.heart_rate < 0.0) {
        throw std::invalid_argument("Heart rate cannot be negative.");
    }

    if (patient.systolic_bp < 0.0) {
        throw std::invalid_argument("Systolic blood pressure cannot be negative.");
    }

    if (patient.lactate < 0.0) {
        throw std::invalid_argument("Lactate cannot be negative.");
    }

    if (patient.oxygen_saturation < 0.0 || patient.oxygen_saturation > 100.0) {
        throw std::invalid_argument("Oxygen saturation must be between 0 and 100.");
    }

    const double score = compute_linear_score(patient);
    return sigmoid(score);
}

std::string RiskEngine::classify_risk_level(double probability) const {
    if (probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("Probability must be between 0 and 1.");
    }

    if (probability < 0.30) {
        return "low";
    }

    if (probability < 0.70) {
        return "medium";
    }

    return "high";
}

std::vector<double> RiskEngine::batch_predict(const std::vector<PatientVitals>& patients) const {
    std::vector<double> probabilities;
    probabilities.reserve(patients.size());

    for (const auto& patient : patients) {
        probabilities.push_back(compute_probability(patient));
    }

    return probabilities;
}

double compute_demo_risk_probability(
    double age,
    double heart_rate,
    double systolic_bp,
    double lactate,
    double oxygen_saturation
) {
    RiskEngine engine;

    PatientVitals patient{
        age,
        heart_rate,
        systolic_bp,
        lactate,
        oxygen_saturation
    };

    return engine.compute_probability(patient);
}

}  // namespace bggcore