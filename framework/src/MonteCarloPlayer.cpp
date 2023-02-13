#include <ctime>
#include <cstdlib>
#include <iostream>

#include "MonteCarloPlayer.h"

namespace Mancala {

    void MonteCarloPlayer::setSimulationPolicy(std::shared_ptr<SimulationPolicy> policy) {
        m_simPolicy = policy;
    } // end method setSimulationPolicy

    int MonteCarloPlayer::makeMove(std::vector<int> board) {

        return -1;
    } // end method makeMove

} // end namespace Mancala