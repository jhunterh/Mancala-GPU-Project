#include "MonteCarloHybridPlayer.h"
#include "GameTypes.h"
#include "RandomPlayer.h"
#include "MonteCarloUtility.h"

#include <cstdio>

namespace Player {

// Constructor for MonteCarloHybridPlayer
MonteCarloHybridPlayer::MonteCarloHybridPlayer()
{
    // Init curand values
    curandInit();
}

// Run the algorithm for specified number of iterations
void MonteCarloHybridPlayer::runSearch() {
    for(size_t i = 0; i < ITERATION_COUNT_HYBRID; ++i) {
        selection();
        expansion();
        simulation();
        backpropagation();
    }
}

// run a single simulation from the selected node
void MonteCarloHybridPlayer::simulation()
{
    // Start simulation
    gpu_result gpuResult;
    simulationGPU(&gpuResult, m_selectedNode->boardState, m_selectedNode->playerNum);

    // Make sure playcount is not equal to 0
    if(gpuResult.playCount == 0)
    {
        printf("[ERROR] Playcount is equal to 0!\n");
        exit(1);
    }

    // Calculate average player wins
    double playerWinCount = static_cast<double>(gpuResult.winCount[m_rootNode->playerNum]);
    double totalWinCount = static_cast<double>(gpuResult.playCount);
    double avgWins = playerWinCount / totalWinCount;

    // Save wins to node
    m_selectedNode->numWins += avgWins;

    // Mark node as simulated
    m_selectedNode->simulated = true;
}

// propagates simulation results back to the gop of the tree
void MonteCarloHybridPlayer::backpropagation() {
    // numWins at this point should only be 0 or 1 for m_selectedNode
    // It is possible if a leaf node is simulated more than once
    // for numWins to be greater than 1, but that breaks the tree's win/loss ratios
    // so I handle that case with the conditional below
    double backPropValue = (m_selectedNode->numWins > 1) ? 1 : m_selectedNode->numWins;
    MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, EXPLORATION_PARAM_HYBRID);
    while(m_selectedNode->parentNode != nullptr) {
        m_selectedNode = m_selectedNode->parentNode;
        m_selectedNode->numWins += backPropValue;
        MonteCarlo::calculateValue(m_selectedNode, m_rootNode->numTimesVisited, EXPLORATION_PARAM_HYBRID);
    }
}

}