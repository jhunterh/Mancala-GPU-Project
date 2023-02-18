#include "MonteCarloPlayer.h"

namespace Player {

// MonteCarloPlayer constructor
MonteCarloPlayer::MonteCarloPlayer()
{

}

// Select a move from the given boardstate
Game::move_t MonteCarloPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    m_rootNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    m_rootNode->boardState = board;
    m_rootNode->playerNum = playerNum;
    m_selectedNode = m_rootNode;

    runSearch(25); // TODO: udpate way for number of iterations to be set

    Game::movelist_t moveList;
    board.getMoves(moveList, playerNum);

    int maxNode = MonteCarlo::getMaxNode(rootNode->childNodes);

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

void MonteCarloPlayer::selection() {
    while(!MonteCarlo::isLeafNode(m_selectedNode)) {
        int nextNode = MonteCarlo::getMaxNode(m_selectedNode->childNodes);
        m_selectedNode = m_selectedNode->childNodes[nextNode];
    }
}

void MonteCarloPlayer::expansion() {
    if(m_selectedNode->numTimesVisited != 0) {
        // Get list of possible moves
        Game::movelist_t moveList;
        Game::movecount_t count = m_selectedNode->boardState.getMoves(moveList, m_selectedNode->playerNum);

        for(int i = 0; i < count; ++i) {
            m_selectedNode->childNodes.push_back(std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode()));
            m_selectedNode->childNodes[i]->boardState = moveList[i];
            m_selectedNode->childNodes[i]->playerNum = 1; // TODO: Determine playerNum
            m_selectedNode->childNodes[i]->parentNode = m_selectedNode;
        }
    }
}

void MonteCarloPlayer::simulation() {
    // TODO: simulation
}

void MonteCarloPlayer::backpropagation() {
    // TODO: backpropagation
}

}