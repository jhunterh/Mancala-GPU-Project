#ifndef _GAMEMANAGER_H
#define _GAMEMANAGER_H

#include <memory>

#include "Player.h"
#include "MancalaTypes.h"

namespace Mancala {

class GameManager {
public:
    GameManager();
    ~GameManager() = default;

    void initGame(std::shared_ptr<Player> player1, std::shared_ptr<Player> player2);

    void startGame();

private:
    std::shared_ptr<Player> m_player1;
    std::shared_ptr<Player> m_player2;

    std::shared_ptr<MancalaBoard> m_gameBoard;

    int m_playerTurn = 1; // Player 1 goes first

    bool isEndState();
    bool makeMoveOnBoard(int move);

}; // end class GameManager

} // end namespace Mancala

#endif