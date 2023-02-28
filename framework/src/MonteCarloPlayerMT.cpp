#include <thread>
#include <functional>

#include "MonteCarloPlayerMT.h"
#include "GameTypes.h"
#include "RandomPlayer.h"

namespace Player {

// Run the algorithm for specified number of iterations
void MonteCarloPlayerMT::runSearch() {
    for(int i = 0; i < ITERATION_COUNT; ++i) {
        selection();
        expansion();
        simulation();
        backpropagation();
    }
}

// run a single simulation from the selected node
void MonteCarloPlayerMT::simulation() {

    m_endStatesFound = 0;
    m_endStatesFound = 0;

    // start threads
    std::vector<std::thread> threads;
    for(unsigned int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(&MonteCarloPlayerMT::simulationThread, this);
    }

    // sync threads
    for(auto& thread : threads) {
        thread.join();
    }

    std::cout << m_winStatesFound << std::endl;

    double avgWins = m_winStatesFound / m_endStatesFound;

    //std::cout << avgWins << std::endl;

    m_selectedNode->numWins += avgWins;
    m_selectedNode->simulated = true;

}

// propagates simulation results back to the gop of the tree
void MonteCarloPlayerMT::backpropagation() {
    // numWins at this point should only be 0 or 1 for m_selectedNode
    // It is possible if a leaf node is simulated more than once
    // for numWins to be greater than 1, but that breaks the tree's win/loss ratios
    // so I handle that case with the conditional below
    unsigned int backPropValue = (m_selectedNode->numWins > 1) ? 1 : m_selectedNode->numWins;
    MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, EXPLORATION_PARAM);
    while(m_selectedNode->parentNode != nullptr) {
        m_selectedNode = m_selectedNode->parentNode;
        m_selectedNode->numWins += backPropValue;
        MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, EXPLORATION_PARAM);
    }
}

void MonteCarloPlayerMT::simulationThread() {

    while(m_endStatesFound < NUM_END_STATES_DESIRED) {
        ++m_endStatesFound;

        // Declare two random players to duke it out
        RandomPlayer player;

        Game::GameBoard gameBoard = m_selectedNode->boardState;
        playernum_t playerTurn = m_selectedNode->playerNum;

        Game::boardresult_t result = gameBoard.getBoardResult(playerTurn);

        while(result == Game::GAME_ACTIVE) {

            Game::move_t selectedMove = player.selectMove(gameBoard, playerTurn);
            Game::moveresult_t moveResult = gameBoard.executeMove(selectedMove, playerTurn);

            if (moveResult == Game::MOVE_SUCCESS) {
                if (playerTurn == PLAYER_NUMBER_2) {
                    playerTurn = PLAYER_NUMBER_1;
                } else {
                    playerTurn = PLAYER_NUMBER_2;
                }
            } else if (moveResult == Game::MOVE_INVALID) {
                std::cout << "Invalid Move" << std::endl;
            }
            
            result = gameBoard.getBoardResult(playerTurn);
        }
        
        if(GameUtils::getPlayerFromBoardResult(result) == m_rootNode->playerNum) {
            ++m_winStatesFound;
        }
    }
}

}