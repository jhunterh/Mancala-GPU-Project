#include "MonteCarloPlayer.h"
#include "RandomPlayer.h"

namespace Player {

// Select a move from the given boardstate
Game::move_t MonteCarloPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    m_rootNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    m_rootNode->boardState = board;
    m_rootNode->playerNum = playerNum;
    m_rootNode->simulated = true; // no use in simulating this node
    m_selectedNode = m_rootNode;

    runSearch(1000); // TODO: udpate way for number of iterations to be set

    Game::movelist_t moveList;
    board.getMoves(moveList, playerNum);

    int maxNode = MonteCarlo::getMaxNode(m_rootNode->childNodes);

    return moveList[maxNode];
}

// Run the algorithm for specified number of iterations
void MonteCarloPlayer::runSearch(int numIterations) {
    for(int i = 0; i < numIterations; ++i) {
        selection();
        expansion();
        simulation();
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
        // TODO: Need to find a way to better get the next player's turn
        // Suggestion: Replace moveresult_t with playernum_t
        m_selectedNode->childNodes.push_back(std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode()));
        m_selectedNode->childNodes[i]->boardState = newState;
        if (result == 1) {
            if (m_selectedNode->playerNum == 1) {
                m_selectedNode->childNodes[i]->playerNum = 0;
            } else {
                m_selectedNode->childNodes[i]->playerNum = 1;
            }
        } else if (result == 2) {
            m_selectedNode->childNodes[i]->playerNum = m_selectedNode->playerNum;
        } else {
            std::cout << "Invalid Move" << std::endl;
        }
        m_selectedNode->childNodes[i]->parentNode = m_selectedNode;
    }

    // Select first new child node for simulation
    if (count > 0) {
        m_selectedNode = m_selectedNode->childNodes[0];
        ++m_selectedNode->numTimesVisited;
    }
    
}

// run a single simulation from the selected node
void MonteCarloPlayer::simulation() {
    // Declare two random players to duke it out
    RandomPlayer player;

    Game::GameBoard gameBoard = m_selectedNode->boardState;
    playernum_t playerTurn = m_selectedNode->playerNum;

    Game::boardresult_t result = gameBoard.getBoardResult(playerTurn);

    while(result == 0) { // TODO: refactor

        Game::move_t selectedMove = player.selectMove(gameBoard, playerTurn);
        Game::moveresult_t moveResult = gameBoard.executeMove(selectedMove, playerTurn);

        if (moveResult == 1) {
            if (playerTurn == 1) {
                playerTurn = 0;
            } else {
                playerTurn = 1;
            }
        } else if (moveResult == 0) {
            std::cout << "Invalid Move" << std::endl;
        }
        
        result = gameBoard.getBoardResult(playerTurn);
    }
    
    if((result-1) == m_rootNode->playerNum) {
        ++m_selectedNode->numWins;
    }
    m_selectedNode->simulated = true;
}

// propagates simulation results back to the gop of the tree
void MonteCarloPlayer::backpropagation() {
    // numWins at this point should only be 0 or 1 for m_selectedNode
    // It is possible if a leaf node is simulated more than once
    // for numWins to be greater than 1, but that breaks the tree's win/loss ratios
    // so I handle that case with the conditional below
    unsigned int backPropValue = (m_selectedNode->numWins > 1) ? 1 : m_selectedNode->numWins;
    MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, 2); // TODO: set exploration param
    while(m_selectedNode->parentNode != nullptr) {
        m_selectedNode = m_selectedNode->parentNode;
        m_selectedNode->numWins += backPropValue;
        MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, 2);
    }
}

}