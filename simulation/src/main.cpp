#include <iostream>
#include <string>

#include "PlayerManager.h"

// Simulation for any two person board game
int main(int argc, char **argv)
{
    // Check CMD arguments
    if(argc != 3)
    {
        std::cout << "[ERROR] Not enough arguments!" << std::endl
                  << "USAGE: ./simulation <Player1Type> <Player2Type>" << std::endl
                  << Player::PlayerManager::getPlayerTypeList() << std::endl;
        return 1;
    }

    // Set players
    Player::PlayerManager playerManager;

    // Set player 1
    if(!playerManager.selectPlayers(0, std::stoul(argv[1])))
    {
        std::cout << "[ERROR] PlayerType not valid!" << std::endl
                  << Player::PlayerManager::getPlayerTypeList() << std::endl;
        return 1;
    }

    // Set player 2
    if(!playerManager.selectPlayers(1, std::stoul(argv[2])))
    {
        std::cout << "[ERROR] PlayerType not valid!" << std::endl
                  << Player::PlayerManager::getPlayerTypeList() << std::endl;
        return 1;
    }

    // Init game board
    Game::GameBoard gameBoard;
    gameBoard.initBoard();

    // Play game
    std::cout << "GAME STARTING" << std::endl << std::endl
              << "Beginning game state:" << std::endl 
              << gameBoard.getBoardStateString() << std::endl << std::endl;

    Player::playernum_t activePlayer = Player::PLAYER_NUMBER_1;
    Game::boardresult_t gameResult = Game::GAME_ACTIVE;
    while(gameResult == Game::GAME_ACTIVE)
    {
        // Get move from player
        Game::move_t move = playerManager.getMove(activePlayer, gameBoard);

        // Make move on board
        Game::moveresult_t moveResult = gameBoard.executeMove(move, activePlayer);

        // Verify valid move result
        if(!moveResult)
        {
            std::cout << "[ERROR] Incorrect move for player " << std::to_string(activePlayer) 
                        << " given: " << std::to_string(move) << std::endl;
            return 1;
        }

        // Print move and new board state
        std::cout   << "Player " << std::to_string(activePlayer + 1)
                    << " Makes Move: " << std::to_string(move) << std::endl 
                    << "New board state:" << std::endl 
                    << gameBoard.getBoardStateString() << std::endl << std::endl;

        // Check move result
        if(moveResult == Game::MOVE_SUCCESS)
        {
            // Switch players
            activePlayer = !activePlayer;

        }

        // Check if in end state
        gameResult = gameBoard.getBoardResult(activePlayer);
        
    }

    // Output game win
    switch(gameResult)
    {
        case Game::GAME_OVER_PLAYER1_WIN:
            std::cout << "Player 1 wins!" << std::endl;
            break;
        case Game::GAME_OVER_PLAYER2_WIN:
            std::cout << "Player 2 wins!" << std::endl;
            break;
        case Game::GAME_OVER_TIE:
            std::cout << "Game ended in a tie!" << std::endl;
            break;
    }

    // Print final boardstate
    std::cout << "Final board state: " << std::endl
              << gameBoard.getBoardStateString() << std::endl;

    return 0;
} // end main