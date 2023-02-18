#ifndef GAME_H
#define GAME_H

#include <cstdint>


// This file is for user defined variables for a specific game
// For #NUMBER, a number is needed (ex. 1)
// For #TYPE, a type is needed (ex. uint8_t)
// Some definitions can be either a typedef or a struct

// Max possible amount of moves a player could make at one time
// Used for movelist
#define GAME_MAX_POSSIBLE_MOVES #NUMBER

// Size of game board (in square spaces)
#define GAME_BOARD_SIZE #NUMBER

namespace Game
{

// Definition of a single space on the board
typedef #TYPE squarestate_t;
// OR
struct squarestate_t {};

// Definition for an index of a single space
typedef #TYPE boardpos_t;

// Definition for the board itself
typedef squarestate_t boardstate_t[GAME_BOARD_SIZE];
// OR
struct boardstate_t {};

// Definition for a move
typedef #TYPE move_t;
// OR
struct move_t {};

// Definition for a move result (success/failure)
typedef #TYPE moveresult_t;

// Definition of a movecount
// Used for how many moves are in a movelist
typedef #TYPE movecount_t;

// Definition of list of moves
// Used for returning moves from function
// This should be a static array and not a vector due to CUDA constraints
typedef move_t movelist_t[GAME_MAX_POSSIBLE_MOVES];

// Definition of board result
typedef #TYPE boardresult_t;

};

// This is here because a game may have more than two players
namespace Player
{

// Player index (player 1 vs player 2)
typedef #TYPE playernum_t;
enum PlayerNumber
{
    // Definition of highest index of player
    // Ex. a 2 person game would have a value of 1
    PLAYER_NUMBER_MAX = 1,
};

};

#endif // GAME_H