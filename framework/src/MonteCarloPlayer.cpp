#include "MonteCarloPlayer.h"
#include "RandomPlayer.h"

namespace Player {

// Select a move from the given boardstate
Game::move_t MonteCarloPlayer::selectMove(Game::GameBoard& board, playernum_t playerNum)
{
    m_rootNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    m_rootNode->boardState = board;
    m_rootNode->playerNum = playerNum;
    m_selectedNode = m_rootNode;
    m_playerNum = playerNum;

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
    ++m_selectedNode->numTimesVisited;
    while(!MonteCarlo::isLeafNode(m_selectedNode)) {
        int nextNode = MonteCarlo::getMaxNode(m_selectedNode->childNodes);
        m_selectedNode = m_selectedNode->childNodes[nextNode];
        ++m_selectedNode->numTimesVisited;
    }
}

void MonteCarloPlayer::expansion() {
    if(m_selectedNode->numTimesVisited != 0) {
        // Get list of possible moves
        Game::movelist_t moveList;
        Game::movecount_t count = m_selectedNode->boardState.getMoves(moveList, m_selectedNode->playerNum);

        // Expand the selected node
        for(int i = 0; i < count; ++i) {
            Game::GameBoard newState = m_selectedNode->boardState;
            moveresult_t result = newState.executeMove(moveList[i], m_selectedNode->playerNum);
            // TODO: Need to find a way to better get the next player's turn
            if (result <= 0) {
                std::cout << "INVALID MOVE" << std::endl;
            }
            m_selectedNode->childNodes.push_back(std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode()));
            m_selectedNode->childNodes[i]->boardState = newState;
            m_selectedNode->childNodes[i]->playerNum = result; // This is stinky and needs a refactor
            m_selectedNode->childNodes[i]->parentNode = m_selectedNode;
        }

        // Select first new child node for simulation
        if (count > 0) {
            m_selectedNode = m_selectedNode->childNodes[0];
        }
    }
}

void MonteCarloPlayer::simulation() {
    // Declare two random players to duke it out
    std::shared_ptr<Player> player = std::shared_ptr<Player>(new RandomPlayer());

    Game::GameBoard gameBoard = m_selectedNode->boardState;
    playernum_t playerTurn = m_selectedNode->playerNum;

    boardresult_t result = gameBoard.getBoardResult();

    while(result == 0) { // TODO: refactor
        
        Game::move_t selectedMove = player->selectMove(gameBoard, playerTurn);
        moveresult_t moveResult = gameBoard.executeMove(selectedMove, playerTurn);
        if(moveResult <= 0) {
            std::cout << "INVALID MOVE" << std::endl;
        }
        
        result = gameBoard.getBoardResult();
    }
    
    if(result == m_playerNum) {
        ++m_selectedNode->numWins;
    }

}

void MonteCarloPlayer::backpropagation() {
    unsigned int backPropValue = m_selectedNode->numWins;
    MonteCarlo::getUCT(m_selectedNode, m_rootNode->numTimesVisited, 2); // TODO: set exploration param
    while(m_selectedNode->parentNode != nullptr) {
        m_selectedNode = m_selectedNode->parentNode;
        m_selectedNode->numWins += backPropValue;
        MonteCarlo::getUCT(m_selectedNode, m_rootNode->numTimesVisited, 2);
    }
}

}