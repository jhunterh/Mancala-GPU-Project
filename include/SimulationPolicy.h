#ifndef _SIMULATIONPOLICY_H
#define _SIMULATIONPOLICY_H

#include <vector>

namespace Mancala {

class SimulationPolicy {
public:

    virtual float runSimulation(std::vector<int> gameState, int playerTurn){return 0.0f;}

protected:
    SimulationPolicy() = default;

}; // end class SimulationPolicy

} // end namespace Mancala

#endif