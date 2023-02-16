#ifndef _PLAYER_H
#define _PLAYER_H

#include <vector>
#include <string>

#include "defines.h"
#include "GameBoard.h"

namespace Player {

class Player {
public:

    virtual playertype_t getPlayerType() = 0;
	virtual std::string getDescription() = 0;
	virtual Game::move_t selectMove(Game::GameBoard& board, playernum_t playerNum) = 0;

protected:
    Player() = default;

private:

}; // end class Player

} // end namespace Mancala

#endif