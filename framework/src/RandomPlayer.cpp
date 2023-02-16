#include <ctime>
#include <cstdlib>
#include <iostream>

#include "RandomPlayer.h"

namespace Player {
    
RandomPlayer::RandomPlayer()
{
    srand(time(NULL));
}

Game::move_t RandomPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    Game::movelist_t moveList;
    Game::movecount_t count = board.getMoves(moveList, playerNum);
    
    Game::movecount_t randomValue = (rand() % count);

    return moveList[randomValue];
}

} // end namespace Mancala