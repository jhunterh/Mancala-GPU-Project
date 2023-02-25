#ifndef GAMEBOARD_H
#define GAMEBOARD_H

#include "game.h"
#include "GameTypes.h"
#include <string>

namespace Game {

// GameBoard class
// Used to store and interface with the board state
// Implementation is defined by the user depending on the game
class GameBoard {
public:

    // Note: The following functions are constant for each game, do not modify
    GameBoard() = default;
    ~GameBoard() = default;

    // Note: The following functions are to be user-implemented for each game:

    // Init the board state
    void initBoard();

    // Execute a move on the board for a given player
    Game::moveresult_t executeMove(Game::move_t move, Player::playernum_t playerNum);

    // Return the possible move on the board for a given player
    Game::movecount_t getMoves(Game::movelist_t& movesOut, Player::playernum_t playerNum);

    // Return the board result
    // Current player number is needed for some games
    Game::boardresult_t getBoardResult(Player::playernum_t currentPlayerNum);

    // Return the state of the board in string format
    std::string getBoardStateString();

private:

    // Internal state of the board
    Game::boardstate_t boardState;

};

}

#endif // GAME_H