#ifndef GAME_H
#define GAME_H

#include <cctype>

#define GAME_MAX_POSSIBLE_MOVES 6
#define GAME_BOARD_SIZE 14

namespace Game {

enum squarestate_t : uint8_t
{
    P1_START = 0,
    P1_GOAL = 6,
    P2_START = 7,
    P2_GOAL = 13;
}

typedef int8_t boardpos_t;

typedef squarestate_t[GAME_BOARD_SIZE] boardstate_t;

typedef boardpos_t move_t;

enum moveresult_t : uint8_t
{
    MOVE_INVALID = 0,
    MOVE_SUCCESS = 1,
    MOVE_SUCCESS_GO_AGAIN = 2
}

typedef uint8_t movecount_t;

typedef move_t[GAME_MAX_POSSIBLE_MOVES] movelist_t;

enum boardresult_t : int8_t
{
    GAME_ACTIVE = 0,
    GAME_OVER_PLAYER1_WIN = 1,
    GAME_OVER_PLAYER2_WIN = 2,
    GAME_OVER_TIE = 2
}

}

#endif // GAME_H