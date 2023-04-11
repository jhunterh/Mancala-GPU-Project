#include "GameBoard.h"
#include <sstream>
#include <iomanip>
#include <ctime>
#include <cstdlib>

// GameBoard implementation for Clobber

namespace Game {

// Init the board state
void GameBoard::initBoard()
{
    BoardSquare rowStart = P1_PIECE;
    BoardSquare current;
    for(int i = 0; i < 8; ++i)
    {
        current = rowStart;
        for(int j = 0; j < 8; ++j)
        {
            boardState[i*8+j] = current;
            current = (current == P1_PIECE) ? P2_PIECE : P1_PIECE;
        }
        rowStart = (rowStart == P1_PIECE) ? P2_PIECE : P1_PIECE;
    }
}

// Execute a move on the board for a given player
CUDA_CALLABLE_MEMBER moveresult_t GameBoard::executeMove(move_t move, Player::playernum_t playerNum)
{
    moveresult_t result = MOVE_INVALID;
    BoardSquare pieceType = (playerNum == Player::PLAYER_NUMBER_1) ? P1_PIECE : P2_PIECE;
    BoardSquare opponentPiece = (playerNum == Player::PLAYER_NUMBER_1) ? P2_PIECE : P1_PIECE;

    int moveDir = move & 0xC0;
    int row = (move & 0x3f) / 8;
    int col = (move & 0x3f) % 8;

    switch(moveDir)
    {
        case 0x00: // 0
        {
            if(boardState[row*8+col-1] == opponentPiece)
            {
                boardState[row*8+col-1] = pieceType;
                result = MOVE_SUCCESS;
            }
            break;
        }
        case 0x40: // 1
        {
            if(boardState[(row-1)*8+col] == opponentPiece)
            {
                boardState[(row-1)*8+col] = pieceType;
                result = MOVE_SUCCESS;
            }
            break;
        }
        case 0x80: // 2
        {
            if(boardState[row*8+col+1] == opponentPiece)
            {
                boardState[row*8+col+1] = pieceType;
                result = MOVE_SUCCESS;
            }
            break;
        }
        case 0xC0: // 3
        {
            if(boardState[(row+1)*8+col] == opponentPiece)
            {
                boardState[(row+1)*8+col] = pieceType;
                result = MOVE_SUCCESS;
            }
            break;
        }
    }

    if(result == MOVE_SUCCESS)
    {
        boardState[row*8+col] = EMPTY;
    }
    
    return result;
}

// Return the possible move on the board for a given player
CUDA_CALLABLE_MEMBER movecount_t GameBoard::getMoves(movelist_t& movesOut, Player::playernum_t playerNum)
{
    // Loop through each move
    movecount_t moveCount = 0;
    BoardSquare pieceType = (playerNum == Player::PLAYER_NUMBER_1) ? P1_PIECE : P2_PIECE;
    BoardSquare opponentType = (playerNum == Player::PLAYER_NUMBER_1) ? P2_PIECE : P1_PIECE;
    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            if(boardState[i*8+j] == pieceType)
            {
                if(j-1 >= 0 && boardState[(i*8+j-1)] == opponentType)
                {
                    move_t moveNum = (i*8+j);
                    movesOut[moveCount++] = moveNum;
                }
                
                if(i-1 >= 0 && boardState[(i-1)*8+j] == opponentType)
                {
                    move_t moveNum = (i*8+j) | 0x40;
                    movesOut[moveCount++] = moveNum;
                }
                
                if(j+1 < 8 && boardState[i*8+j+1] == opponentType)
                {
                    move_t moveNum = (i*8+j) | 0x80;
                    movesOut[moveCount++] = moveNum;
                }
                
                if(i+1 < 8 && boardState[(i+1)*8+j] == opponentType)
                {
                    move_t moveNum = (i*8+j) | 0xC0;
                    movesOut[moveCount++] = moveNum;
                }
            }
        }
    }

    return moveCount;
}

// Return the board result
// Current player number is needed for some games
CUDA_CALLABLE_MEMBER boardresult_t GameBoard::getBoardResult(Player::playernum_t currentPlayerNum)
{
    // Loop through each move
    BoardSquare pieceType = (currentPlayerNum == Player::PLAYER_NUMBER_1) ? P1_PIECE : P2_PIECE;
    BoardSquare opponentType = (currentPlayerNum == Player::PLAYER_NUMBER_1) ? P2_PIECE : P1_PIECE;
    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            if(boardState[i*8+j] == pieceType)
            {
                // if any of these conditons are true, then the current player
                // still has available moves and the game should continue
                if(j-1 >= 0 && boardState[(i*8+j-1)] == opponentType)
                {
                    return GAME_ACTIVE;
                }
                
                if(i-1 >= 0 && boardState[(i-1)*8+j] == opponentType)
                {
                    return GAME_ACTIVE;
                }
                
                if(j+1 < 8 && boardState[i*8+j+1] == opponentType)
                {
                    return GAME_ACTIVE;
                }
                
                if(i+1 < 8 && boardState[(i+1)*8+j] == opponentType)
                {
                    return GAME_ACTIVE;
                }
            }
        }
    }

    // current player is out of moves
    // return board result
    return (currentPlayerNum == Player::PLAYER_NUMBER_1) ? GAME_OVER_PLAYER2_WIN : GAME_OVER_PLAYER1_WIN;
}

// Return the state of the board in string format
std::string GameBoard::getBoardStateString()
{
    // Create stringstream
    std::stringstream out("");

    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            std::string piece("");
            if(boardState[i*8+j] == EMPTY)
            {
                piece = " ";
            }
            else if(boardState[i*8+j] == P1_PIECE)
            {
                piece = "1";
            }
            else if(boardState[i*8+j] == P2_PIECE)
            {
                piece = "2";
            }
            out << piece << " ";
        }
        out << std::endl;
    }

    // Return string stream
    return out.str();
}

std::string GameBoard::getMoveString(move_t move)
{
    int row = (move & 0x3f) / 8;
    int col = (move & 0x3f) % 8;
    int moveDir = move & 0xC0;
    
    std::stringstream out("");
    out << "HERE1: " << move << std::endl;
    out << "Row: " << row << " Col: " << col;

    switch(moveDir)
    {
        case 0x00: 
        {
            out << " LEFT";
            break;
        }
        case 0x40:
        {
            out << " UP";
            break;
        }
        case 0x80:
        {
            out << " RIGHT";
            break;
        }
        case 0xC0:
        {
            out << " DOWN";
            break;
        }
    }

    return out.str();
}

// Set the board to a random state
void GameBoard::scramble()
{
    srand(time(NULL));
    for(int i = 0; i < 8; ++i)
    {
        for(int j = 0; j < 8; ++j)
        {
            int num = rand() % 3;  // random number between 0 and 3
            boardState[i*8+j] = num;
        }
    }
}

}