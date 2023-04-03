#ifndef GAME_H
#define GAME_H

#include <cstdint>

// This file is for user defined variables for a specific game
// For #NUMBER, a number is needed (ex. 1)
// For #TYPE, a type is needed (ex. uint8_t)
// Some definitions can be either a typedef or a struct

// Max possible amount of moves a player could make at one time
// Used for movelist
#define GAME_MAX_POSSIBLE_MOVES 42

// Size of game board (in square spaces)
#define GAME_BOARD_SIZE 32

namespace Game
{

// Definition of a single space on the board
typedef uint8_t squarestate_t;
enum SquareState : squarestate_t
{
	SQUARE_EMPTY = 0,
	SQUARE_RED = 4,
	SQUARE_RED_KING = 5,
	SQUARE_BLACK = 6,
	SQUARE_BLACK_KING = 7
};

enum SquareStateBit
{
    BIT_ISKING = 0x1,
    BIT_ISBLACK = 0x2,
    BIT_ISEMPTY = 0x4
};

// Definition for an index of a single space
typedef int8_t boardpos_t;
#define BOARD_POS_INVALID -1

// Definition for the board itself
typedef uint32_t bitboard_t;
struct boardstate_t
{
    bitboard_t isOccupiedBoard, isBlackBoard, isKingBoard;
};

// Definition for a move
struct move_t
{
	boardpos_t oldPos = -1;
	boardpos_t newPos = -1;
	boardpos_t jumpPos = -1;
};

// Definition of a movecount
// Used for how many moves are in a movelist
typedef int8_t movecount_t;

// Definition of list of moves
// Used for returning moves from function
// This should be a static array and not a vector due to CUDA constraints
typedef move_t movelist_t[GAME_MAX_POSSIBLE_MOVES];

};

#endif // GAME_H