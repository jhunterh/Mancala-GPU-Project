#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include "defines.h"
#include "game.h"
#include <string>

namespace Game {

class GameBoard {
public:
    GameBoard() = default;
    ~GameBoard() = default;

    void initBoard();
    Game::boardstate_t* getBoardState();
    Game::moveresult_t executeMove(Game::move_t move, Player::playernum_t playerNum);
    Game::movecount_t getMoves(Game::movelist_t& movesOut, Player::playernum_t playerNum);
    Game::boardresult_t getBoardResult();
    std::string getBoardStateString();


private:
    Game::boardstate_t boardState;

}; // end class GameBoard

}

#endif // GAME_H