#include "GameBoard.h"

namespace Game {

// Init the board state
void GameBoard::initBoard()
{
    boardState = boardstate_t {
        .isOccupiedBoard = 0xFFF00FFF,
        .isBlackBoard = 0xFFF00000,
        .isKingBoard = 0x00000000
    }
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
    // Build list of string characters for board
    char stateChars[GAME_BOARD_SIZE];
    for(boardpos_t pos = 0; pos < GAME_BOARD_SIZE; pos++)
    {
        stateChars[pos] = '.';
        bitboard_t mask = (1 << pos);
        if(boardState.isOccupiedBoard & mask)
        {
            stateChars[pos] = 'r';
            if(boardState.isBlackBoard & mask) stateChars[pos] -= 0x10;
            if(boardState.isKingBoard & mask) stateChars[pos] -= 0x20;
        }
    }

    // Format board
    std::stringstream boardStateBuf;
    boardStateBuf << "-------------------\n";
    for(uint8_t row = 0; row < 8; row++)
    {
        boardStateBuf << "| ";
        if((row % 2) == 0)
            boardStateBuf << "  ";
        for(uint8_t col = 0; col < 4; col++)
            boardStateBuf << stateChars[row*4 + col] << " ";
        if(row % 2)
            boardStateBuf << "  ";
        boardStateBuf << "|\n";
    }
    boardStateBuf << "-------------------";
    
    // Return board string
    return boardStateBuf.str();
}

}