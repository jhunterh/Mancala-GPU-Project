#include <iostream>
#include <sstream>
#include <string>
#include <sstream>

#include "PlayerManager.h"
#include "Timer.h"
#include "Logger.h"

// Simulation for any two person board game
int main(int argc, char **argv)
{
    Logging::Logger& logger = Logging::Logger::getInstance();
    std::stringstream out("");
    logger.log(Logging::PERFORMANCE_LOG,"Performance Log Begin");
    logger.log(Logging::SIMULATION_LOG,"Simulation Log Begin");
    // Check CMD arguments
    if(argc != 4)
    {
        std::cout << "[ERROR] Not enough arguments!" << std::endl
                  << "USAGE: ./simulation <Player1Type> <Player2Type> <NumberOfRounds>" << std::endl
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

    int numRounds = std::atoi(argv[3]);

    int p1wins = 0;
    int p2wins = 0;
    int ties = 0;

    for(int i = 0; i < numRounds; ++i) {
        // Init game board
        Game::GameBoard gameBoard;
        gameBoard.initBoard();

        // Play game
        out.str(std::string(""));
        out << "GAME STARTING" << std::endl << std::endl
                << "Beginning game state:" << std::endl 
                << gameBoard.getBoardStateString() << std::endl << std::endl;
        logger.log(Logging::SIMULATION_LOG, out.str());

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
                out.str(std::string(""));
                out << "[ERROR] Incorrect move for player " << std::to_string(activePlayer) 
                            << " given: " << Game::GameBoard::getMoveString(move) << std::endl;
                logger.log(Logging::SIMULATION_LOG, out.str());
                return 1;
            }

            // Print move and new board state
            out.str(std::string(""));
            out   << "Player " << std::to_string(activePlayer + 1)
                        << " Makes Move: " << Game::GameBoard::getMoveString(move) << std::endl
                        << "New board state:" << std::endl 
                        << gameBoard.getBoardStateString() << std::endl << std::endl;
            logger.log(Logging::SIMULATION_LOG, out.str());
            

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
        out.str(std::string(""));
        switch(gameResult)
        {
            
            case Game::GAME_OVER_PLAYER1_WIN:
                out << "Player 1 wins!" << std::endl;
                ++p1wins;
                break;
            case Game::GAME_OVER_PLAYER2_WIN:
                out << "Player 2 wins!" << std::endl;
                ++p2wins;
                break;
            case Game::GAME_OVER_TIE:
                out << "Game ended in a tie!" << std::endl;
                ++ties;
                break;
        }
        logger.log(Logging::SIMULATION_LOG, out.str());

        // Print final boardstate
        out.str(std::string(""));
        out << "Final board state: " << std::endl
                << gameBoard.getBoardStateString() << std::endl;
        logger.log(Logging::SIMULATION_LOG, out.str());
    }
    out.str(std::string(""));
    out << std::endl << "Number of Rounds Played: " << numRounds << std::endl;
    out << "Player 1 Wins: " << p1wins << std::endl;
    out << "Player 2 Wins: " << p2wins << std::endl;
    out << "Ties: " << ties << std::endl;

    out << std::endl << "Player 1 Performance Data:" << std::endl;
    out << playerManager.getPerformanceDataString(Player::PLAYER_NUMBER_1);
    out << std::endl << "Player 2 Performance Data:" << std::endl;
    out << playerManager.getPerformanceDataString(Player::PLAYER_NUMBER_2);
    logger.log(Logging::PERFORMANCE_LOG, out.str());

    return 0;
} // end main