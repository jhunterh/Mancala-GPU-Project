#include <sstream>

#include "MonteCarloPlayerMT.h"
#include "GameTypes.h"
#include "RandomPlayer.h"

namespace Player {

MonteCarloPlayerMT::MonteCarloPlayerMT() {

    m_numIterations = 250;

    m_explorationParam = 0;

    m_gameFinishFlag.store(false);

    m_waitingThreads.store(0);

    unsigned int numThreads = std::thread::hardware_concurrency();
    numThreads = (numThreads > MAX_NUM_THREADS) ? MAX_NUM_THREADS : numThreads; 
    if(numThreads > 0) {
        std::stringstream out("");
        out << getDescription() << ": Creating " << numThreads << " Simulation Threads" << std::endl;
        m_logger.log(Logging::PERFORMANCE_LOG, out.str());
        for(unsigned int i = 0; i < numThreads; ++i) {
            m_threads.emplace_back(&MonteCarloPlayerMT::simulationThread, this);
        }
    } else {
        m_logger.log(Logging::PERFORMANCE_LOG, "UNABLE TO DETECT NUMBER OF CORES ON SYSTEM!");
    }
    
}

MonteCarloPlayerMT::~MonteCarloPlayerMT() {
    m_gameFinishFlag.store(true);
    while(m_waitingThreads.load() < 4);
    m_simulationCondition.notify_all();
    for(auto& thread : m_threads) thread.join();
}

// run a single simulation from the selected node
unsigned int MonteCarloPlayerMT::simulation() {

    m_endStatesFound.store(0);
    m_winStatesFound.store(0);
    m_numMovesSimulated.store(0);
    m_simulationDoneFlag.store(false);
    while(m_waitingThreads.load() < 4);
    m_simulationCondition.notify_all();

    while(!m_simulationDoneFlag);

    double avgWins = ((double) m_winStatesFound) / NUM_END_STATES_DESIRED;

    m_selectedNode->numWins += avgWins;
    m_selectedNode->simulated = true;

    return m_numMovesSimulated.load();
}

void MonteCarloPlayerMT::simulationThread() {
    std::unique_lock<std::mutex> lck(m_simulationMutex);
    while(1) {

        ++m_waitingThreads;
        m_simulationCondition.wait(lck);
        --m_waitingThreads;
        if(m_gameFinishFlag.load()) {
            return;
        }
        
        while(m_endStatesFound++ < NUM_END_STATES_DESIRED) {

            Game::GameBoard gameBoard = m_selectedNode->boardState;
            playernum_t playerTurn = m_selectedNode->playerNum;

            Game::boardresult_t result = gameBoard.getBoardResult(playerTurn);

            unsigned int localNumMovesSimulated = 0;

            while(result == Game::GAME_ACTIVE) {

                Game::move_t selectedMove = m_randomPlayer->selectMove(gameBoard, playerTurn);
                Game::moveresult_t moveResult = gameBoard.executeMove(selectedMove, playerTurn);
                ++localNumMovesSimulated;

                if (moveResult == Game::MOVE_SUCCESS) {
                    if (playerTurn == PLAYER_NUMBER_2) {
                        playerTurn = PLAYER_NUMBER_1;
                    } else {
                        playerTurn = PLAYER_NUMBER_2;
                    }
                } else if (moveResult == Game::MOVE_INVALID) {
                    m_logger.log(Logging::SIMULATION_LOG, "Invalid Move!");
                }
                
                result = gameBoard.getBoardResult(playerTurn);
            }
            m_numMovesSimulated += localNumMovesSimulated;
            
            if(GameUtils::getPlayerFromBoardResult(result) == m_rootNode->playerNum) {
                ++m_winStatesFound;
            }
        }
        m_simulationDoneFlag.store(true);
    }
}

}