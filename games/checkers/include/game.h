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
enum SquareState : boardstate_t
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
	boardpos_t jumpPos - -1;
};

// Definition of a movecount
// Used for how many moves are in a movelist
typedef int8_t movecount_t;

// Definition of list of moves
// Used for returning moves from function
// This should be a static array and not a vector due to CUDA constraints
typedef move_t movelist_t[GAME_MAX_POSSIBLE_MOVES];

const boardpos_t cornerList[GAME_BOARD_SIZE][4] = {
	{-1, -1, 4, 5},{-1, -1, 5, 6},{-1, -1, 6, 7},{-1, -1, 7, -1},
	{-1, 0, -1, 8},{0, 1, 8, 9},{1, 2, 9, 10},{2, 3, 10, 11},
	{4, 5, 12, 13},{5, 6, 13, 14},{6, 7, 14, 15},{7, -1, 15, -1},
	{-1, 8, -1, 16},{8, 9, 16, 17},{9, 10, 17, 18},{10, 11, 18, 19},
	{12, 13, 20, 21},{13, 14, 21, 22},{14, 15, 22, 23},{15, -1, 23, -1},
	{-1, 16, -1, 24},{16, 17, 24, 25},{17, 18, 25, 26},{18, 19, 26, 27},
	{20, 21, 28, 29},{21, 22, 29, 30},{22, 23, 30, 31},{23, -1, 31, -1},
	{-1, 24, -1, -1},{24, 25, -1, -1},{25, 26, -1, -1},{26, 27, -1, -1}
};

};

#endif // GAME_H