#ifndef _RANDOMPLAYER_H
#define _RANDOMPLAYER_H

#include <vector>

#include "Player.h"

namespace Player {

// Definition of Random Player
// This player always selects a move at random
class RandomPlayer : public Player {
public:
    RandomPlayer();
    ~RandomPlayer() = default;

    player_t getPlayerType() override { return 0; }
	std::string getDescription() override { return "Random Player"; }
	Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum);

};

}

#endif