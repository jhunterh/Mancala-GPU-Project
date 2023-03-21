#include <sstream>

#include "PlayerManager.h"
#include "RandomPlayer.h"
#include "MonteCarloPlayer.h"
#include "MonteCarloPlayerMT.h"
#include "MonteCarloHybridPlayer.h"

namespace Player
{

// Static declaration of player types
// Add new player types here
const std::vector<std::shared_ptr<Player>> PlayerManager::playerTypeList = {
    std::shared_ptr<Player>(new RandomPlayer()),
    std::shared_ptr<Player>(new MonteCarloPlayer()),
    std::shared_ptr<Player>(new MonteCarloPlayerMT()),
    std::shared_ptr<Player>(new MonteCarloHybridPlayer())
};

// Returns list of player types
std::string PlayerManager::getPlayerTypeList()
{
    std::string playerListString = "Player Types:\n";
    for(uint8_t i = 0; i < playerTypeList.size(); i++)
    {
        playerListString += " - " + std::to_string(i) + ": " + 
                            playerTypeList.at(i)->getDescription() + "\n";
    }
    return playerListString;
}

// Select player type for player
// Returns false if not a valid player number or type
bool PlayerManager::selectPlayers(playernum_t playerNum, player_t playerType)
{
    // Check that player number is not out of range
    if(playerNum > PLAYER_NUMBER_2)
    {
        return false;
    }

    // Check that player type is not out of range
    if(playerTypeList.size() <= playerType)
    {
        return false;
    }

    // Set player type
    playerList[playerNum] = playerTypeList[playerType];

    return true;
}

// Get player-chosen move
Game::move_t PlayerManager::getMove(playernum_t playerNum, Game::GameBoard& board)
{
    // Check that player number is not out of range
    if(playerNum > PLAYER_NUMBER_2)
    {
        return Game::MOVE_INVALID;
    }

    // Return player move
    return playerList[playerNum]->selectMove(board, playerNum);
}

// Print Player performance data
std::string PlayerManager::getPerformanceDataString(playernum_t playerNum) 
{
    std::stringstream out("");
    // Check that player number is not out of range
    if(playerNum > PLAYER_NUMBER_2)
    {
        out << "Invalid Player" << std::endl;
    }

    // Print data
    out << playerList[playerNum]->getPerformanceDataString();

    return out.str();
}

}