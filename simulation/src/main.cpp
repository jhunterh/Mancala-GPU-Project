#include <iostream>
#include <string>

#include "PlayerManager.h"

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        std::cout << "[ERROR] Not enough arguments!" << std::endl
                  << "USAGE: ./mancala <PlayerType> <PlayerType>" << std::endl
                  << Player::PlayerManager::getPlayerList() << std::endl;
        return 1;
    }

    Player::PlayerManager playerManager;
    if(!playerManager.selectPlayers(std::stoul(argv[1]), std::stoul(argv[2])))
    {
        std::cout << "[ERROR] PlayerType not valid!" << std::endl
                  << Player::PlayerManager::getPlayerList() << std::endl;
        return 1;
    }

    // Play game
    Game::GameBoard gameBoard;
    gameBoard.initBoard();

    std::cout << "GAME STARTING" << std::endl << std::endl
              << "Beginning game state:" << std::endl 
              << gameBoard.getBoardStateString() << std::endl << std::endl;

    bool gameActive = true; // no winner yet
    Player::playernum_t activePlayer = Player::PLAYER_NUMBER_1;

    while(gameActive)
    {
        // Get move from player
        Game::move_t move = playerManager.getMove(activePlayer, gameBoard);

        // Make move on board
        Game::moveresult_t moveResult = gameBoard.executeMove(move, activePlayer);

        // Check if in end state
        Game::boardresult_t gameResult = gameBoard.getBoardResult();

        // Check game result
        if(gameResult)
        {
            // If game over, stop game
            gameActive = false;
        }
        else
        {
            // Check move result
            if(!moveResult)
            {
                std::cout << "[ERROR] Incorrect move for player " << std::to_string(activePlayer) 
                          << " given: " << std::to_string(move) << std::endl;
                return 1;
            }
            else
            {
                // Print move and new board state
                std::cout << "Player " << std::to_string(activePlayer + 1)
                          << " Makes Move: " << std::to_string(move) << std::endl 
                          << "New board state:" << std::endl 
                          <<gameBoard.getBoardStateString() << std::endl << std::endl;
                if(moveResult == Game::MOVE_SUCCESS)
                {
                    // Switch players
                    activePlayer = !activePlayer;

                }
            }
        }
        
    } // end while

    // Output game win
    Game::boardresult_t gameResult = gameBoard.getBoardResult();
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