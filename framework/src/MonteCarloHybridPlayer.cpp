#include "MonteCarloHybridPlayer.h"
#include "GameTypes.h"
#include "RandomPlayer.h"

#include <cstdio>

namespace Player {

// Constructor for MonteCarloHybridPlayer
MonteCarloHybridPlayer::MonteCarloHybridPlayer()
{
    m_numIterations = 250;

    // Init curand values
    m_explorationParam = 0;
    setDeterministic(false, 0);
}

// set random value to pre-determined value (for unit testing)
void MonteCarloHybridPlayer::setDeterministic(bool isPreDetermined, int value)
{
    m_deterministicData.isPreDetermined = isPreDetermined;
    m_deterministicData.value = value;
}

// run a single simulation from the selected node
unsigned int MonteCarloHybridPlayer::simulation()
{
    // Start simulation
    gpu_result gpuResult;
    simulationGPU(&gpuResult, m_selectedNode->boardState, m_selectedNode->playerNum, m_deterministicData);

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

    return gpuResult.numMovesSimulated;
}

}