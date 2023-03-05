#include "MonteCarloHybridPlayer.h"
#include "GameTypes.h"
#include "RandomPlayer.h"

#include <ctime>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define BLOCK_SIZE 1024
#define MAX_THREADS_PER_CURAND 256

#define CURAND_SIZE ((BLOCK_SIZE + MAX_THREADS_PER_CURAND - 1) / BLOCK_SIZE)

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("%s at %s:%d\n",cudaGetErrorString(x),__FILE__,__LINE__); \
    exit(x);}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
    printf("%d at %s:%d\n",x,__FILE__,__LINE__); \
    exit(x);}} while(0)

struct gpu_result
{
    gpu_count_t winCount[2] = { 0, 0 };
    gpu_count_t playCount = 0;
};

#define RESULT_SIZE (sizeof(gpu_result)/sizeof(gpu_count_t))

__device__ __constant__ Player::playernum_t initPlayerTurn;
__device__ __constant__ Game::GameBoard initalGameBoard;

__global__ void simulationKernel(gpu_count_t* gpu_result_out, curandStateMtgp32_t* curandState)
{
    // Initialize result memory (to be copied out later)
    // First two are players, third is total play count
    __shared__ gpu_count_t resultCount[RESULT_SIZE];
    __shared__ curandStateMtgp32_t curandStateShared[CURAND_SIZE];

    // Init result counts to 0
    if(threadIdx.x < RESULT_SIZE)
        resultCount[threadIdx.x] = 0;
    __syncthreads();

    // Grab curand state 
    unsigned int curandStateIdx = threadIdx.x/MAX_THREADS_PER_CURAND;
    if(threadIdx.x == curandStateIdx)
        curandStateShared[curandStateIdx] = curandState[curandStateIdx];
    __syncthreads();

    // Init search states
    Player::playernum_t currentPlayerTurn = initPlayerTurn;
    Game::GameBoard currentBoardState = initalGameBoard;
    Game::boardresult_t currentBoardResult = Game::GAME_ACTIVE;

    // Do simulations until threshold reached
    while(resultCount[2] < PLAYCOUNT_THRESHOLD_HYBRID)
    {
        // Pick random move and execute

        // Get list of possible moves
        Game::movelist_t moveList;
        Game::movecount_t moveCount = currentBoardState.getMoves(moveList, currentPlayerTurn);
        
        // Prevent floating point exceptions
        if(moveCount > 0)
        {   
            // Select random move
            Game::move_t selectedMove = moveList[curand(&curandStateShared[curandStateIdx]) % moveCount];

            // Execute random move
            Game::moveresult_t moveResult = currentBoardState.executeMove(selectedMove, currentPlayerTurn);

            // Check Move
            // TODO: Add check to make sure move is not invalid
            if (moveResult == Game::MOVE_SUCCESS) {
                if (currentPlayerTurn == Player::PLAYER_NUMBER_2) {
                    currentPlayerTurn = Player::PLAYER_NUMBER_1;
                } else {
                    currentPlayerTurn = Player::PLAYER_NUMBER_2;
                }
            }
        }
        
        // Check if game has ended
        currentBoardResult = currentBoardState.getBoardResult(currentPlayerTurn);
        if(currentBoardResult != Game::GAME_ACTIVE)
        {
            // Game has ended, make sure board result has clear winner
            if(currentBoardResult == Game::GAME_OVER_PLAYER1_WIN || 
                currentBoardResult == Game::GAME_OVER_PLAYER2_WIN)
            {
                // If winner, count player win
                atomicAdd(&resultCount[currentBoardResult - 1], 1);
            }

            // Count playout
            atomicAdd(&resultCount[2], 1);

            // Reset search board states
            currentPlayerTurn = initPlayerTurn;
            currentBoardState = initalGameBoard;
        }

        // Sync threads before next stage
        // Prevents reading and writing at same time
        __syncthreads();
    }

    // Copy gpu result out
    if(threadIdx.x < RESULT_SIZE)
        gpu_result_out[threadIdx.x] = resultCount[threadIdx.x];
    __syncthreads();
    
    // Save rng state
    if(threadIdx.x == curandStateIdx)
        curandState[curandStateIdx] = curandStateShared[curandStateIdx];
    __syncthreads();
}

namespace Player {

// Constructor for MonteCarloHybridPlayer
MonteCarloHybridPlayer::MonteCarloHybridPlayer()
{
    // Allocate space for prng states on device
    CUDA_CALL(cudaMalloc(&devMTGPStates, CURAND_SIZE * sizeof(curandStateMtgp32)));

    /* Allocate space for MTGP kernel parameters */
    CUDA_CALL(cudaMalloc(&devKernelParams, sizeof(mtgp32_kernel_params)));

    // Create memory for GPU result out
    CUDA_CALL(cudaMalloc(&gpu_result_dev, sizeof(gpu_result)));

    /* Reformat from predefined parameter sets to kernel format, */
    /* and copy kernel parameters to device memory               */
    CURAND_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams));

    /* Initialize one state per thread block */
    CURAND_CALL(curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, 
                                CURAND_SIZE, time(NULL)));
}


// Destructor for MonteCarloHybridPlayer
MonteCarloHybridPlayer::~MonteCarloHybridPlayer()
{
    cudaFree(devKernelParams);
    cudaFree(devMTGPStates);
    cudaFree(gpu_result_dev);
}

// Run the algorithm for specified number of iterations
void MonteCarloHybridPlayer::runSearch() {

    cudaSearchInit();

    for(size_t i = 0; i < ITERATION_COUNT_HYBRID; ++i) {
        selection();
        expansion();
        simulation();
        backpropagation();
    }
}

void MonteCarloHybridPlayer::cudaSearchInit()
{
    // Get current board, turn, and result
    Game::GameBoard gameBoard = m_selectedNode->boardState;
    playernum_t playerTurn = m_selectedNode->playerNum;

    // Copy information to constant memory
    cudaMemcpyToSymbol(initPlayerTurn, &playerTurn, sizeof(playernum_t));
    cudaMemcpyToSymbol(initalGameBoard, &gameBoard, sizeof(Game::GameBoard));
}

// run a single simulation from the selected node
void MonteCarloHybridPlayer::simulation()
{
    // Launch kernel
    simulationKernel<<<1, BLOCK_SIZE>>>(gpu_result_dev, devMTGPStates);

    // Copy result back
    gpu_result gpu_result_host;
    cudaMemcpy(&gpu_result_host, gpu_result_dev, sizeof(gpu_result), cudaMemcpyDeviceToHost);

    // Calculate average player wins
    double playerWinCount = static_cast<double>(gpu_result_host.winCount[m_rootNode->playerNum]);
    double totalWinCount = static_cast<double>(gpu_result_host.playCount);
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