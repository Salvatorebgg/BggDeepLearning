#include "bggcore/risk_engine.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    try {
        bggcore::RiskEngine engine;

        bggcore::PatientVitals patient_1{
            45.0,
            82.0,
            125.0,
            1.4,
            98.0
        };

        bggcore::PatientVitals patient_2{
            72.0,
            128.0,
            86.0,
            5.6,
            88.0
        };

        double probability_1 = engine.compute_probability(patient_1);
        double probability_2 = engine.compute_probability(patient_2);

        std::string level_1 = engine.classify_risk_level(probability_1);
        std::string level_2 = engine.classify_risk_level(probability_2);

        std::vector<bggcore::PatientVitals> patients = {
            patient_1,
            patient_2,
            bggcore::PatientVitals{60.0, 105.0, 100.0, 3.2, 94.0}
        };

        std::vector<double> batch_results = engine.batch_predict(patients);

        std::cout << "============================================================\n";
        std::cout << "BggDeepLearning C++ risk engine test started\n";
        std::cout << "============================================================\n";

        std::cout << std::fixed << std::setprecision(4);

        std::cout << "Patient 1 probability: " << probability_1
                  << " | risk level: " << level_1 << "\n";

        std::cout << "Patient 2 probability: " << probability_2
                  << " | risk level: " << level_2 << "\n";

        std::cout << "------------------------------------------------------------\n";
        std::cout << "Batch prediction results:\n";

        for (std::size_t i = 0; i < batch_results.size(); ++i) {
            std::cout << "Patient " << i + 1
                      << " probability: " << batch_results[i]
                      << " | risk level: "
                      << engine.classify_risk_level(batch_results[i])
                      << "\n";
        }

        std::cout << "------------------------------------------------------------\n";
        std::cout << "BggDeepLearning C++ risk engine test finished successfully\n";
        std::cout << "============================================================\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "C++ risk engine test failed: " << error.what() << "\n";
        return 1;
    }
}