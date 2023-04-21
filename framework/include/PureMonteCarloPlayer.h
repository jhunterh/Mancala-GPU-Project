#ifndef _PUREMONTECARLOPLAYER_H
#define _PUREMONTECARLOPLAYER_H

#include <vector>

#include "RandomPlayer.h"
#include "MonteCarloTypes.h"
#include "MonteCarloUtility.h"

namespace Player {

// Definition of Pure Monte Carlo Player
// This player selects a move based on the Pure Monte Carlo Search Algorithm
class PureMonteCarloPlayer : public Player {
public:
    PureMonteCarloPlayer();
    ~PureMonteCarloPlayer() = default;

    player_t getPlayerType() override { return 4; }
	std::string getDescription() override { return "Pure Monte Carlo Player"; }
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
        (void) num;
    }

    void setRootNode(std::shared_ptr<MonteCarlo::TreeNode> node)
    {
        m_rootNode = node;
    }

    void simulateMove(int moveNum, std::vector<unsigned int>& simulationResults, std::vector<unsigned int>& simulationNumMoves);

private:
    std::shared_ptr<MonteCarlo::TreeNode> m_rootNode = nullptr;
    std::vector<MonteCarlo::SimulationPerformanceReport> m_simulationReports;
    deterministic_data m_deterministicData;
    std::vector<double> m_executionTimes;
};

}

#endif