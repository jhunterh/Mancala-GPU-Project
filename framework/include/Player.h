#ifndef _PLAYER_H
#define _PLAYER_H

#include <vector>
#include <string>

#include "GameBoard.h"

namespace Player {

// Player type
// Used to store which type of player (AI)
typedef uint8_t playertype_t;

// Abstract Player class
// Not instantiated, but acts as base for other Player types
class Player {
public:

    // Return which level of Player AI
    virtual playertype_t getPlayerType() = 0;

    // Get string description of Player type
	virtual std::string getDescription() = 0;

    // Select a move from the given boardstate
	virtual Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum) = 0;

protected:
    Player() = default;

private:

};

}

#endif