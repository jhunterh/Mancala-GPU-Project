#include <ctime>
#include <cstdlib>
#include <iostream>

#include "RandomPlayer.h"

namespace Player {

// RandomPlayer constructor
RandomPlayer::RandomPlayer()
{
    // Set random seed
    srand(time(NULL));
}

// Select a move from the given boardstate
Game::move_t RandomPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    // Get list of possible moves
    Game::movelist_t moveList;
    Game::movecount_t count = board.getMoves(moveList, playerNum);
    
    // Prevent floating point exceptions
    if(count > 0)
    {
        Game::movecount_t randomValue;

        // early return condition
        if(m_isPreDetermined) 
        {
            randomValue = m_preDeterminedValue;
        }
        else
        {
            // Get random move index
            randomValue = (rand() % count);
        }

        

        // Return random move
        return moveList[randomValue];
    }

    // Return empty move
    return Game::move_t();
}

}