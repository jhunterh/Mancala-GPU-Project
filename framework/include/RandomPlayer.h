#ifndef _RANDOMPLAYER_H
#define _RANDOMPLAYER_H

#include <vector>

#include "Player.h"

namespace Player {

class RandomPlayer : public Player {
public:
    RandomPlayer();
    ~RandomPlayer() = default;

    playertype_t getPlayerType() override { return 0; }
	std::string getDescription() override { return "Random Player"; }
	Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum);

};

}

#endif