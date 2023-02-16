#ifndef GAME_H
#define GAME_H

#include <cstdint>


// This file is for user defined variables for the game mancala

// Max possible amount of moves a player could make at one time
// Used for movelist
#define GAME_MAX_POSSIBLE_MOVES 6

// Size of game board (in square spaces)
#define GAME_BOARD_SIZE 14

namespace Game
{

// Definition of a single space on the board
typedef uint8_t squarestate_t;
enum BoardSquare : squarestate_t
{
    P1_START = 0,
    P1_GOAL = 6,
    P2_START = 7,
    P2_GOAL = 13
};

// Definition for an index of a single space
typedef int8_t boardpos_t;

// Definition for the board itself
typedef squarestate_t boardstate_t[GAME_BOARD_SIZE];

// Definition for a move
typedef boardpos_t move_t;

// Definition for a move result (success/failure)
typedef uint8_t moveresult_t;
enum MoveResult : moveresult_t
{
    MOVE_INVALID = 0,
    MOVE_SUCCESS = 1,
    MOVE_SUCCESS_GO_AGAIN = 2
};

// Definition of a movecount
// Used for how many moves are in a movelist
typedef uint8_t movecount_t;

// Definition of list of moves
// Used for returning moves from function
// This should be a static array and not a vector due to CUDA constraints
typedef move_t movelist_t[GAME_MAX_POSSIBLE_MOVES];

// Definition of board result
typedef int8_t boardresult_t;
enum BoardResult : boardresult_t
{
    GAME_ACTIVE = 0,
    GAME_OVER_PLAYER1_WIN = 1,
    GAME_OVER_PLAYER2_WIN = 2,
    GAME_OVER_TIE = 3
};

};

// This is here because a game may have more than two players
namespace Player
{

// Player index (player 1 vs player 2)
typedef uint8_t playernum_t;
enum PlayerNumber
{
    PLAYER_NUMBER_1 = 0,
    PLAYER_NUMBER_2 = 1,
    
    // Definition of highest index of player
    // Ex. a 2 person game would have a value of 1
    PLAYER_NUMBER_MAX = 1,
};

};

#endif // GAME_H