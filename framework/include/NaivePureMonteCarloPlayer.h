#ifndef _NAIVEPUREMONTECARLOPLAYER_H
#define _NAIVEPUREMONTECARLOPLAYER_H

#include <vector>

#include "RandomPlayer.h"
#include "MonteCarloTypes.h"
#include "MonteCarloUtility.h"

namespace Player {

// Definition of Naive Pure Monte Carlo Player
// This player selects a move based on the Pure Monte Carlo Search Algorithm
class NaivePureMonteCarloPlayer : public Player {
public:
    NaivePureMonteCarloPlayer() = default;
    ~NaivePureMonteCarloPlayer() = default;

    player_t getPlayerType() override { return 5; }
	std::string getDescription() override { return "Naive Pure Monte Carlo Player"; }
	Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum);
    std::vector<MonteCarlo::SimulationPerformanceReport> getSimulationReports() {return m_simulationReports;}
    std::string getPerformanceDataString() override;

    // unit testing interface
    void setDeterministic(bool isPreDetermined, int value) 
    {
        m_deterministicData.isPreDetermined = isPreDetermined;
        m_deterministicData.value = value;
    }

    void setNumSimulations(int num)
    {
        m_numSimulations = num;
    }

    void setRootNode(std::shared_ptr<MonteCarlo::TreeNode> node)
    {
        m_rootNode = node;
    }

private:
    void runSimulation(unsigned int& simulationResults, unsigned int& simulationNumMoves, int moveNum);
    int m_numSimulations = 1000;
    std::shared_ptr<MonteCarlo::TreeNode> m_rootNode = nullptr;
    std::vector<MonteCarlo::SimulationPerformanceReport> m_simulationReports;
    deterministic_data m_deterministicData;
};

}

#endif