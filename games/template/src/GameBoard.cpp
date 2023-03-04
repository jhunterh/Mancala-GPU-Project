#include "GameBoard.h"

namespace Game {

// Init the board state
void GameBoard::initBoard()
{
    // Implement code here
}

// Execute a move on the board for a given player
CUDA_CALLABLE_MEMBER moveresult_t GameBoard::executeMove(move_t move, Player::playernum_t playerNum)
{
    // Implement code here
    return 0;
}

// Return the possible move on the board for a given player
CUDA_CALLABLE_MEMBER movecount_t GameBoard::getMoves(movelist_t& movesOut, Player::playernum_t playerNum)
{
    // Implement code here
    return 0;
}

// Return the board result
CUDA_CALLABLE_MEMBER boardresult_t GameBoard::getBoardResult()
{
    // Implement code here
    return 0;
}

// Return the state of the board in string format
std::string GameBoard::getBoardStateString()
{
    // Implement code here
    return "";
}

}