#ifndef _MONTECARLOPLAYER_H
#define _MONTECARLOPLAYER_H

#include <vector>
#include <memory>

#include "MancalaStatic.h"
#include "Player.h"
#include "SimulationPolicy.h"

namespace Mancala {

class MonteCarloPlayer : public Player {
public:
    MonteCarloPlayer() = default;
    ~MonteCarloPlayer() = default;

    void setSimulationPolicy(std::shared_ptr<SimulationPolicy> policy);

    int makeMove(std::vector<int> board) override;

private:

    enum SearchState {
        SELECTION,
        EXPANSION,
        SIMULATION,
        BACKPROPAGATION
    };

    std::shared_ptr<SimulationPolicy> m_simPolicy;

}; // end class Player

} // end namespace Mancala

#endif