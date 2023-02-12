#include <ctime>
#include <cstdlib>
#include <iostream>

#include "MonteCarloPlayer.h"
#include "MancalaStatic.h"

#define EXPLORATION_PARAMETER 0.25
#define ITERATIONS 25

namespace Mancala {

    void MonteCarloPlayer::setSimulationPolicy(std::shared_ptr<SimulationPolicy> policy) {
        m_simPolicy = policy;
    } // end method setSimulationPolicy

    int MonteCarloPlayer::makeMove(std::vector<int> board) {

        int selectedMove = -1;
        m_totalSimulationsRan = 0;
        
        m_rootNode = std::make_shared<TreeNode>(new TreeNode());
        m_rootNode->gameState = board;
        m_rootNode->playerTurn = getPlayerNumber();
        m_rootNode->timesVisited = 1;

        for(int i = 0; i < ITERATIONS; ++i) {
            selection();
            expansion();
            simulation();
        } // end for

        return selectedMove;
    } // end method makeMove

    void MonteCarloPlayer::selection() {
        m_selectedNode = m_rootNode;
        
        while(!m_selectedNode.isLeaf) {
            std::shared_ptr<TreeNode> maxChild = m_selectedNode->children[0];

            for(auto child : m_selectedNode->children) {
                if(child->value > maxChild->value) {
                    maxChild = child;
                } // end if
            } // end for
            
            m_selectedNode = maxChild;
        } // end while
    } // end method selection

    void MonteCarloPlayer::expansion() {

        if(timesVisited == 0) {
            std::vector<int> validMoves = getValidMoves();

            for(auto move : validMoves) {
                std::shared_ptr<TreeNode> newNode(new TreeNode());
                newNode->parent = m_selectedNode;
                // TODO: set playerTurn and gameState
                m_selectedNode->children.push_back();
            } // end for

            if(validMoves.size() > 0) {
                m_selectedNode->isLeaf = false;
                m_selectedNode = m_selectedNode->children[0];
            } // end if
        } // end if

    } // end method expansion

    void MonteCarloPlayer::simulation() {
        m_selectedNode->value = m_simPolicy->runSimulation(m_selectedNode->gameState, m_selectedNode->playerTurn);
    } // end method simulation

    void MonteCarloPlayer::backpropagation() {
        // TODO: backpropagation
    } // end method backpropagation

} // end namespace Mancala