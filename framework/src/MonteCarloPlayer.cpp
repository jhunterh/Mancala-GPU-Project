#include <sstream>

#include "MonteCarloPlayer.h"
#include "GameTypes.h"
#include "Timer.h"

namespace Player {

// default constructor
MonteCarloPlayer::MonteCarloPlayer()
{
    m_randomPlayer = std::shared_ptr<RandomPlayer>(new RandomPlayer());
}

// Select a move from the given boardstate
Game::move_t MonteCarloPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    m_rootNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    m_rootNode->boardState = board;
    m_rootNode->playerNum = playerNum;
    m_rootNode->simulated = true; // no use in simulating this node
    m_selectedNode = m_rootNode;

    runSearch();

    Game::movelist_t moveList;
    board.getMoves(moveList, playerNum);

    int maxNode = MonteCarlo::getMaxNode(m_rootNode->childNodes);

    return moveList[maxNode];
}

// Run the algorithm for specified number of iterations
void MonteCarloPlayer::runSearch() {
    Timer timer;
    for(size_t i = 0; i < m_numIterations; ++i) {
        selection();
        expansion();
        timer.start();
        unsigned int numMovesSimulated = simulation();
        timer.stop();
        MonteCarlo::SimulationPerformanceReport simReport;
        simReport.executionTime = timer.elapsedTime_ms();
        simReport.numMovesSimulated = numMovesSimulated;
        m_simulationReports.push_back(simReport);
        backpropagation();
    }
}

// selectd a node to expand
void MonteCarloPlayer::selection() {
    ++m_selectedNode->numTimesVisited;
    while(!MonteCarlo::isLeafNode(m_selectedNode)) {
        int nextNode = MonteCarlo::selectLeafNode(m_selectedNode->childNodes);
        m_selectedNode = m_selectedNode->childNodes[nextNode];
        ++m_selectedNode->numTimesVisited;
    }
}

// expands selected node if it is eligible for expansion
void MonteCarloPlayer::expansion() {

    // If this node hasn't been simulated,
    // Then we don't want to expand it yet
    if(!m_selectedNode->simulated) {
        return;
    }

    // Get list of possible moves
    Game::movelist_t moveList;
    Game::movecount_t count = m_selectedNode->boardState.getMoves(moveList, m_selectedNode->playerNum);

    // Expand the selected node
    for(int i = 0; i < count; ++i) {
        Game::GameBoard newState = m_selectedNode->boardState;
        Game::moveresult_t result = newState.executeMove(moveList[i], m_selectedNode->playerNum);
        m_selectedNode->childNodes.push_back(std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode()));
        m_selectedNode->childNodes[i]->boardState = newState;

        if (result == Game::MOVE_SUCCESS) {
            if (m_selectedNode->playerNum == PLAYER_NUMBER_2) {
                m_selectedNode->childNodes[i]->playerNum = PLAYER_NUMBER_1;
            } else {
                m_selectedNode->childNodes[i]->playerNum = PLAYER_NUMBER_2;
            }
        } else if (result == Game::MOVE_SUCCESS_GO_AGAIN) {
            m_selectedNode->childNodes[i]->playerNum = m_selectedNode->playerNum;
        } else {
            m_logger.log(Logging::SIMULATION_LOG,"Invalid Move!");
        }

        m_selectedNode->childNodes[i]->parentNode = m_selectedNode;
    }

    // Select first new child node for simulation
    if (count > 0) {
        m_selectedNode = m_selectedNode->childNodes[0];
        ++m_selectedNode->numTimesVisited;
    }

}

// run single-threaded simulations from the selected node
unsigned int MonteCarloPlayer::simulation() {
    int numWins = 0;

    unsigned int numMovesSimulated = 0;

    for(int i = 0; i < m_numSimulations; ++i) {
        Game::GameBoard gameBoard = m_selectedNode->boardState;
        playernum_t playerTurn = m_selectedNode->playerNum;

        Game::boardresult_t result = gameBoard.getBoardResult(playerTurn);

        while(result == Game::GAME_ACTIVE) {

            Game::move_t selectedMove = m_randomPlayer->selectMove(gameBoard, playerTurn);
            Game::moveresult_t moveResult = gameBoard.executeMove(selectedMove, playerTurn);
            ++numMovesSimulated;

            if (moveResult == Game::MOVE_SUCCESS) {
                if (playerTurn == PLAYER_NUMBER_2) {
                    playerTurn = PLAYER_NUMBER_1;
                } else {
                    playerTurn = PLAYER_NUMBER_2;
                }
            } else if (moveResult == Game::MOVE_INVALID) {
                m_logger.log(Logging::SIMULATION_LOG,"Invalid Move!");
            }
            
            result = gameBoard.getBoardResult(playerTurn);
        }
        
        if(GameUtils::getPlayerFromBoardResult(result) == m_rootNode->playerNum) {
            ++numWins;
        }
    }

    m_selectedNode->numWins += ((double) numWins) / m_numSimulations;

    m_selectedNode->simulated = true;

    return numMovesSimulated;
}

// propagates simulation results back to the gop of the tree
void MonteCarloPlayer::backpropagation() {
    // numWins at this point should only be 0 or 1 for m_selectedNode
    // It is possible if a leaf node is simulated more than once
    // for numWins to be greater than 1, but that breaks the tree's win/loss ratios
    // so I handle that case with the conditional below
    unsigned int backPropValue = (m_selectedNode->numWins > 1) ? 1 : m_selectedNode->numWins;
    MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, m_explorationParam);
    while(m_selectedNode->parentNode != nullptr) {
        m_selectedNode = m_selectedNode->parentNode;
        m_selectedNode->numWins += backPropValue;
        MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, m_explorationParam);
    }
}

// Get performance data string
std::string MonteCarloPlayer::getPerformanceDataString() {
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