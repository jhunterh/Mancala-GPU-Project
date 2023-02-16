#ifndef PLAYERMANAGER_H
#define PLAYERMANAGER_H

#include "Player.h"
#include <vector>
#include <memory>

namespace Player
{
    
class PlayerManager
{
public:
    PlayerManager() = default;
    ~PlayerManager() = default;

    static std::string getPlayerList();
    bool selectPlayers(playernum_t playerNum, playertype_t playerType);
    Game::move_t getMove(playernum_t playerNum, Game::GameBoard& board);

private:
    static const std::vector<std::shared_ptr<Player>> playerTypeList;
    std::vector<std::shared_ptr<Player>> playerList = { playerTypeList[0], playerTypeList[0] };
};

}

#endif // PLAYERMANAGER_H