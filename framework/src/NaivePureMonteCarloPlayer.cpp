#include <sstream>
#include <thread>

#include "NaivePureMonteCarloPlayer.h"
#include "GameTypes.h"
#include "Timer.h"

namespace Player {

// Default Constructor
NaivePureMonteCarloPlayer::NaivePureMonteCarloPlayer()
{
    m_randomPlayer = std::shared_ptr<RandomPlayer>(new RandomPlayer());
}

// Select a move from the given boardstate
Game::move_t NaivePureMonteCarloPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    m_rootNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    m_rootNode->boardState = board;
    m_rootNode->playerNum = playerNum;
    m_rootNode->simulated = true;

    // get available moves
    Game::movelist_t moveList;
    Game::movecount_t count = board.getMoves(moveList, playerNum);

    std::vector<unsigned int> simulationResults;
    std::vector<unsigned int> simulationNumMoves;

    Timer timer;
    timer.start();

    // Expand the selected node
    for(int i = 0; i < count; ++i) {
        Game::GameBoard newState = board;
        Game::moveresult_t result = newState.executeMove(moveList[i], playerNum);
        m_rootNode->childNodes.push_back(std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode()));
        m_rootNode->childNodes[i]->boardState = newState;

        if (result == Game::MOVE_SUCCESS) {
            if (playerNum == PLAYER_NUMBER_2) {
                m_rootNode->childNodes[i]->playerNum = PLAYER_NUMBER_1;
            } else {
                m_rootNode->childNodes[i]->playerNum = PLAYER_NUMBER_2;
            }
        } else if (result == Game::MOVE_SUCCESS_GO_AGAIN) {
            m_rootNode->childNodes[i]->playerNum = playerNum;
        } else {
            m_logger.log(Logging::SIMULATION_LOG,"Invalid Move!");
        }

        m_rootNode->childNodes[i]->parentNode = m_rootNode;

        unsigned int results = 0;
        unsigned int numMoves = 0;

        runSimulation(results, numMoves, i);

        simulationResults.push_back(results);
        simulationNumMoves.push_back(numMoves);
    }
    timer.stop();

    unsigned int numMoves = 0;
    for(auto& count : simulationNumMoves) {
        numMoves += count;
    }

    MonteCarlo::SimulationPerformanceReport simReport;
    simReport.numMovesSimulated = numMoves;
    simReport.executionTime = timer.elapsedTime_ms();

    m_simulationReports.push_back(simReport);

    int max = 0;
    for(int i = 0; i < count; ++i) {
        if(simulationResults[i] > simulationResults[max]) {
            max = i;
        }
    }

    return moveList[max];
}

void NaivePureMonteCarloPlayer::runSimulation(unsigned int& simulationResults, unsigned int& simulationNumMoves, int moveNum) {
    Game::GameBoard boardState = m_rootNode->childNodes[moveNum]->boardState;
    playernum_t playerTurn = m_rootNode->childNodes[moveNum]->playerNum;
    Game::boardresult_t result = boardState.getBoardResult(playerTurn);

    while(result == Game::GAME_ACTIVE) {

        Game::move_t selectedMove = m_randomPlayer->selectMove(boardState, playerTurn);
        Game::moveresult_t moveResult = boardState.executeMove(selectedMove, playerTurn);
        ++simulationNumMoves;

        if (moveResult == Game::MOVE_SUCCESS) {
            if (playerTurn == PLAYER_NUMBER_2) {
                playerTurn = PLAYER_NUMBER_1;
            } else {
                playerTurn = PLAYER_NUMBER_2;
            }
        } else if (moveResult == Game::MOVE_INVALID) {
            m_logger.log(Logging::SIMULATION_LOG,"Invalid Move!");
        }
        
        result = boardState.getBoardResult(playerTurn);
    }
    
    if(GameUtils::getPlayerFromBoardResult(result) == m_rootNode->playerNum) {
        ++simulationResults;
    }
}

// Get performance data string
std::string NaivePureMonteCarloPlayer::getPerformanceDataString() {
    std::stringstream out("");
    out << getDescription() << ":" << std::endl;
    unsigned int numSimulations = 0;
    double executionTimeAggregate = 0.0f;
    unsigned int numMovesSimulatedAggregate = 0;
    for(auto report : m_simulationReports) {
        ++numSimulations;
        executionTimeAggregate += report.executionTime;
        numMovesSimulatedAggregate += report.numMovesSimulated;
    }
    double averageExecutionTime = executionTimeAggregate / numSimulations;
    double movesPerSecond = numMovesSimulatedAggregate / (executionTimeAggregate / 1000);

    out << "\tAverage Execution Time (For Simulation Step) - " << averageExecutionTime << std::endl;
    out << "\tMoves Simulated Per Second - " << movesPerSecond << std::endl;

    return out.str();
}

}