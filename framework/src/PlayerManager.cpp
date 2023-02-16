#include "PlayerManager.h"
#include "RandomPlayer.h"

namespace Player
{

const std::vector<Player*> PlayerManager::playerTypeList = {
    new RandomPlayer()
};

PlayerManager::~PlayerManager()
{
    // TODO: Fix memory leak
    //for(uint8_t i = 0; i < playerList.size(); i++) delete playerList[i];
}

std::string PlayerManager::getPlayerList()
{
    std::string playerListString = "Player Types:\n";
    for(uint8_t i = 0; i < playerTypeList.size(); i++)
    {
        playerListString += " - " + std::to_string(i) + ":" + 
                            playerTypeList.at(i)->getDescription() + "\n";
    }
    return playerListString;
}

bool PlayerManager::selectPlayers(playernum_t playerNum, playertype_t playerType)
{
    if(playerNum < PLAYER_NUMBER_MIN || playerNum > PLAYER_NUMBER_MAX)
    {
        return false;
    }

    playerList[playerNum] = playerTypeList[playerType];

    return true;
}

Game::move_t PlayerManager::getMove(playernum_t playerNum, Game::GameBoard& board)
{
    if(playerNum < PLAYER_NUMBER_MIN || playerNum > PLAYER_NUMBER_MAX)
    {
        return Game::MOVE_INVALID;
    }

    return playerList[playerNum]->selectMove(board, playerNum);
}

}