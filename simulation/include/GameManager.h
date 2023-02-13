#ifndef _GAMEMANAGER_H
#define _GAMEMANAGER_H

#include <memory>
#include <vector>

#include "Player.h"
#include "MancalaStatic.h"

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

    std::vector<int> m_gameBoard{std::vector<int>(14,4)};

    int m_playerTurn = 1; // Player 1 goes first

    bool makeMoveOnBoard(int move);
    int determineWinner();

}; // end class GameManager

} // end namespace Mancala

#endif