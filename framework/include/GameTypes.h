#ifndef GAMETYPES_H
#define GAMETYPES_H

#include <string>
#include <map>

namespace Game {

// Definition for a move result (success/failure)
typedef uint8_t moveresult_t;
enum MoveResult : moveresult_t
{
    MOVE_INVALID = 0,
    MOVE_SUCCESS = 1,
    MOVE_SUCCESS_GO_AGAIN = 2
};

// Definition of board result
typedef int8_t boardresult_t;
enum BoardResult : boardresult_t
{
    GAME_ACTIVE = 0,
    GAME_OVER_PLAYER1_WIN = 1,
    GAME_OVER_PLAYER2_WIN = 2,
    GAME_OVER_TIE = 3
};

}

namespace Player {

typedef uint8_t playernum_t;
enum PlayerNumber : playernum_t
{
    PLAYER_NUMBER_1 = 0,
    PLAYER_NUMBER_2 = 1,
    N0_PLAYER = 2,
    TOTAL_PLAYERS = 2
};

}

namespace GameUtils {
    static Player::playernum_t getPlayerFromBoardResult(Game::boardresult_t result) {
        std::map<Game::boardresult_t, Player::playernum_t> s_map;
        s_map[Game::BoardResult::GAME_OVER_PLAYER1_WIN] = Player::PlayerNumber::PLAYER_NUMBER_1;
        s_map[Game::BoardResult::GAME_OVER_PLAYER2_WIN] = Player::PlayerNumber::PLAYER_NUMBER_2;

        Player::playernum_t retVal = Player::PlayerNumber::N0_PLAYER;

        if(s_map.find(result) != s_map.end()) {
            retVal = s_map[result];
        }

        return retVal;
    }
}



#endif // GAMETYPES_H