#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include "game.h"
#include <vector>

namespace Game {

class GameBoard {
public:
    GameBoard();
    ~GameBoard();

    void initBoard();
    Game::boardstate_t getBoardState();
    Game::moveresult_t executeP1Move(Game::move_t move);
    Game::moveresult_t executeP2Move(Game::move_t move);
    Game::movecount_t getP1Moves(Game::movelist_t* movesOut);
    Game::movecount_t getP2Moves(Game::movelist_t* movesOut);
    Game::boardresult_t getBoardResult();

private:
    Game::boardstate_t boardState;

}; // end class GameBoard

}

#endif // GAME_H