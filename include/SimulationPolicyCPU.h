#ifndef _SIMULATIONPOLICYCPU_H
#define _SIMULATIONPOLICYCPU_H

#include <vector>

#include "SimulationPolicy.h"
#include "RandomPlayer.h"

namespace Mancala {

class SimulationPolicyCPU : public SimulationPolicy {
public:
    SimulationPolicyCPU() = default;
    ~SimulationPolicyCPU() = default;

    float runSimulation(std::vector<int> gameState, int playerTurn) override;

private:
    std::vector<int> m_gameState;
    int m_playerTurn = -1;

    RandomPlayer m_player1;
    RandomPlayer m_player2;

    int determineWinner();
    bool makeMoveOnBoard(int move);


}; // end class SimulationPolicy

} // end namespace Mancala

#endif