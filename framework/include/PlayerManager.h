#ifndef PLAYERMANAGER_H
#define PLAYERMANAGER_H

#include "Player.h"
#include <vector>
#include <memory>

namespace Player
{

// Player Manager class
// Used to store, manage, and interface with players
class PlayerManager
{
public:
    PlayerManager() = default;
    ~PlayerManager() = default;

    // Returns list of player types
    static std::string getPlayerTypeList();

    // Select player type for player
    // Returns false if not a valid player number or type
    bool selectPlayers(playernum_t playerNum, player_t playerType);

    // Get player-chosen move
    Game::move_t getMove(playernum_t playerNum, Game::GameBoard& board);

private:

    // Static declaration of types of players
    static const std::vector<std::shared_ptr<Player>> playerTypeList;

    // List of current players
    std::vector<std::shared_ptr<Player>> playerList = std::vector<std::shared_ptr<Player>>(
        PLAYER_NUMBER_2 + 1, playerTypeList[0]
    );
};

}

#endif // PLAYERMANAGER_H