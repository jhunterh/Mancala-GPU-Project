#ifndef _MONTECARLOPLAYER_H
#define _MONTECARLOPLAYER_H

#include <vector>

#include "RandomPlayer.h"
#include "MonteCarloTypes.h"

namespace Player {

// Definition of Monte Carlo Player
// This player selects a move based on the Monte Carlo Tree Search Algorithm
class MonteCarloPlayer : public Player {
public:
    MonteCarloPlayer();
    ~MonteCarloPlayer() = default;

    player_t getPlayerType() override { return 1; }
	std::string getDescription() override { return "Monte Carlo Player"; }
	Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum);
    void setNumIterations(int iterations) {m_numIterations = iterations;}
    std::vector<MonteCarlo::SimulationPerformanceReport> getSimulationReports() {return m_simulationReports;}
    void printPerformanceData() override;

    // unit testing interface
    virtual void selection();
    virtual unsigned int simulation();
    virtual void expansion();
    virtual void backpropagation();

    void setNumSimulations(int num)
    {
        m_numSimulations = num;
    }

    virtual void setDeterministic(bool isPreDetermined, int value)
    {
        m_randomPlayer->setDeterministic(isPreDetermined, value);
    }

    void setRootNode(std::shared_ptr<MonteCarlo::TreeNode> node)
    {
        m_rootNode = node;
    }

    void setSelectedNode(std::shared_ptr<MonteCarlo::TreeNode> node)
    {
        m_selectedNode = node;
    }

    std::shared_ptr<MonteCarlo::TreeNode> getSelectedNode()
    {
        return m_selectedNode;
    }

    void setExplorationParam(double ep)
    {
        m_explorationParam = ep;
    }

protected:
    std::shared_ptr<RandomPlayer> m_randomPlayer;
    std::shared_ptr<MonteCarlo::TreeNode> m_rootNode = nullptr;
    std::shared_ptr<MonteCarlo::TreeNode> m_selectedNode = nullptr;
    double m_explorationParam = 1;
    int m_numIterations = 1000;
    std::vector<MonteCarlo::SimulationPerformanceReport> m_simulationReports;

private:
    void runSearch();
    int m_numSimulations = 1;
};

}

#endif