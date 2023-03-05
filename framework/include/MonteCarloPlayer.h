#ifndef _MONTECARLOPLAYER_H
#define _MONTECARLOPLAYER_H

#include <vector>

#include "Player.h"
#include "MonteCarloTypes.h"

#define EXPLORATION_PARAM 2
#define ITERATION_COUNT 1000

namespace Player {

// Definition of Monte Carlo Player
// This player selects a move based on the Monte Carlo Tree Search Algorithm
class MonteCarloPlayer : public Player {
public:
    MonteCarloPlayer() = default;
    ~MonteCarloPlayer() = default;

    player_t getPlayerType() override { return 1; }
	std::string getDescription() override { return "Monte Carlo Player"; }
	Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum);

protected:
    std::shared_ptr<MonteCarlo::TreeNode> m_rootNode = nullptr;
    std::shared_ptr<MonteCarlo::TreeNode> m_selectedNode = nullptr;
    virtual void runSearch();
    virtual void selection();
    virtual void simulation();
    virtual void expansion();
    virtual void backpropagation();
};

}

#endif