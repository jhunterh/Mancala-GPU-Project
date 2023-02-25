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

// Definition of a movecount
// Used for how many moves are in a movelist
typedef uint8_t movecount_t;

// Definition of list of moves
// Used for returning moves from function
// This should be a static array and not a vector due to CUDA constraints
typedef move_t movelist_t[GAME_MAX_POSSIBLE_MOVES];

};

#endif // GAME_H