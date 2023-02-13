#include "GameBoard.h"

namespace Game {

GameBoard::GameBoard()
{

}

GameBoard::~GameBoard()
{

}

void GameBoard::initBoard()
{
    for(uint8_t i = P1_START; i < P1_GOAL; i++)
    {
        boardState[i] = 4;
    }
    for(uint8_t i = P2_START; i < P2_GOAL; i++)
    {
        boardState[i] = 4;
    }
}

boardstate_t GameBoard::getBoardState()
{
    return boardState;
}

moveresult_t GameBoard::executeP1Move(move_t move)
{
    // TODO
}

moveresult_t GameBoard::executeP2Move(move_t move)
{
    // TODO
}

movecount_t GameBoard::getP1Moves(movelist_t* movesOut)
{
    // Loop through each move
    movecount_t moveCount = 0;
    for(uint8_t i = P1_START; i < P1_GOAL; i++)
    {
        movesOut[moveCount] = boardState[i];    // Store variable
        moveCount += (boardState[i] != 0);      // Move up list if valid move
    }
    movesOut[moveCount] = 0;    // Reset last variable in case of false set
    return moveCount;
}

movecount_t GameBoard::getP2Moves(movelist_t* movesOut)
{
    // Loop through each move
    movecount_t moveCount = 0;
    for(uint8_t i = P2_START; i < P2_GOAL; i++)
    {
        movesOut[moveCount] = boardState[i];    // Store variable
        moveCount += (boardState[i] != 0);      // Move up list if valid move
    }
    movesOut[moveCount] = 0;    // Reset last variable in case of false set
    return moveCount;
}

boardresult_t GameBoard::getBoardResult()
{
    uint8_t p1Score = 0;
    for(uint8_t i = P1_START; i < P1_GOAL; i++)
    {
        p1Score += boardState[i];
    }

    uint8_t p2Score = 0;
    for(uint8_t i = P2_START; i < P2_GOAL; i++)
    {
        p2Score += boardState[i];
    }

    return static_cast<boardresult_t>((p1Score != 0) + (p2Score != 0));
}

}