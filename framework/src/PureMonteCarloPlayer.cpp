#include <sstream>
#include <thread>

#include "PureMonteCarloPlayer.h"
#include "GameTypes.h"
#include "Timer.h"

namespace Player {

// Default Constructor
PureMonteCarloPlayer::PureMonteCarloPlayer()
{
    setDeterministic(false, 0);
    curandInit();
}

// Select a move from the given boardstate
Game::move_t PureMonteCarloPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    Timer turnTimer;
    turnTimer.start();

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

        simulationResults.push_back(0);
        simulationNumMoves.push_back(0);

        // launch thread
        simulateMove(i, simulationResults, simulationNumMoves);
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

    turnTimer.stop();
    m_executionTimes.push_back(turnTimer.elapsedTime_ms());
    
    return moveList[max];
}

void PureMonteCarloPlayer::simulateMove(int moveNum, std::vector<unsigned int>& simulationResults, std::vector<unsigned int>& simulationNumMoves) {
    // Start simulation
    gpu_result gpuResult;
    simulationGPU(&gpuResult, m_rootNode->childNodes[moveNum]->boardState, m_rootNode->childNodes[moveNum]->playerNum, m_deterministicData);

    simulationNumMoves[moveNum] = gpuResult.numMovesSimulated;

    // Make sure playcount is not equal to 0
    if(gpuResult.playCount == 0)
    {
        m_logger.log(Logging::SIMULATION_LOG, "[ERROR] Playcount is equal to 0!");
    }
    else
    {
        simulationResults[moveNum] = gpuResult.winCount[m_rootNode->playerNum];
    }
}

// Get performance data string
std::string PureMonteCarloPlayer::getPerformanceDataString() {
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

    double turnTimesAggregate = 0;
    for(auto turnTime : m_executionTimes)
    {
        turnTimesAggregate += (turnTime / 1000);
    }
    double avgTurnTime = turnTimesAggregate / m_executionTimes.size();

    out << "\tAverage Turn Execution Time - " << avgTurnTime << std::endl;
    out << "\tAverage Execution Time (For Simulation Step) - " << averageExecutionTime << std::endl;
    out << "\tMoves Simulated Per Second - " << movesPerSecond << std::endl;

    return out.str();
}

}