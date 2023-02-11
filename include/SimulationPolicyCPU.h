#ifndef _SIMULATIONPOLICYCPU_H
#define _SIMULATIONPOLICYCPU_H

#include <vector>

#include "SimulationPolicy.h"

namespace Mancala {

class SimulationPolicyCPU : public SimulationPolicy {
public:
    SimulationPolicyCPU() = default;
    ~SimulationPolicyCPU() = default;

    float runSimulation(std::vector<int> gameState) override;

}; // end class SimulationPolicy

} // end namespace Mancala

#endif