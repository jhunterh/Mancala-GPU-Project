#ifndef _GAMEMANAGER_H
#define _GAMEMANAGER_H

#include <vector>

#include "Player.h"

namespace Mancala {

class GameManager {
public:
    GameManager();
    ~GameManager() = default;

    void initGame(std::shared_ptr<Player> player1, std::shared_ptr<Player> player2);

    void startGame();

private:

    uint8_t m_playerTurn = 1; // Player 1 goes first

}; // end class GameManager

} // end namespace Mancala

#endif