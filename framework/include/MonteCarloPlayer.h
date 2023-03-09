#ifndef _MONTECARLOPLAYER_H
#define _MONTECARLOPLAYER_H

#include <vector>

#include "RandomPlayer.h"
#include "MonteCarloTypes.h"

#define EXPLORATION_PARAM 1
#define ITERATION_COUNT 1000

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

    // unit testing interface
    virtual void selection();
    virtual void simulation();
    virtual void expansion();
    virtual void backpropagation();

    void setNumSimulations(int num)
    {
        m_numSimulations = num;
    }

    void setDeterministic(bool isPreDetermined, int value)
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

protected:
    std::shared_ptr<RandomPlayer> m_randomPlayer;
    std::shared_ptr<MonteCarlo::TreeNode> m_rootNode = nullptr;
    std::shared_ptr<MonteCarlo::TreeNode> m_selectedNode = nullptr;
    virtual void runSearch();

private:
    int m_numSimulations = 1;
};

}

#endif