#include "MonteCarloHybridPlayer.h"
#include "GameTypes.h"
#include "RandomPlayer.h"

#include <ctime>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>

#define PLAYCOUNT_THRESHOLD 1000

typedef unsigned int gpu_count_t;
struct gpu_result
{
    gpu_count_t winCount[2] = { 0, 0 };
    gpu_count_t playCount = 0;
};

#define RESULT_SIZE (sizeof(gpu_result)/sizeof(gpu_count_t))

__device__ __constant__ Player::playernum_t initPlayerTurn;
__device__ __constant__ Game::GameBoard initalGameBoard;
__device__ __constant__ curandStateMtgp32_t* curandStateConstant;

__global__ void simulationKernel(gpu_count_t* gpu_result_out)
{
    // Initialize result memory (to be copied out later)
    // First two are players, third is total play count
    __shared__ gpu_count_t resultCount[RESULT_SIZE];

    // Init result counts to 0
    if(threadIdx.x < RESULT_SIZE)
        gpu_result_out[threadIdx.x] = 0;
    __syncthreads();

    // Grab curand state 
    curandStateMtgp32_t curandStateLocal = curandStateConstant[threadIdx.x];

    // Init search states
    Player::playernum_t currentPlayerTurn = initPlayerTurn;
    Game::GameBoard currentBoardState = initalGameBoard;
    Game::boardresult_t currentBoardResult = Game::GAME_ACTIVE;

    // Do simulations until threshold reached
    while(resultCount[2] < PLAYCOUNT_THRESHOLD)
    {
        // Pick random move and execute
        // TODO: Use CURAND

        // Get list of possible moves
        Game::movelist_t moveList;
        Game::movecount_t moveCount = currentBoardState.getMoves(moveList, currentPlayerTurn);
        
        // Prevent floating point exceptions
        if(moveCount > 0)
        {   
            // Select random move
            Game::move_t selectedMove = moveList[curand(&curandStateLocal) % moveCount];

            // Execute random move
            Game::moveresult_t moveResult = currentBoardState.executeMove(selectedMove, currentPlayerTurn);

            // Check Move
            // TODO: Add move result checking
            if (moveResult == Game::MOVE_SUCCESS) {
                if (currentPlayerTurn == Player::PLAYER_NUMBER_2) {
                    currentPlayerTurn = Player::PLAYER_NUMBER_1;
                } else {
                    currentPlayerTurn = Player::PLAYER_NUMBER_2;
                }
            }
        }
        __syncthreads();
        
        // Check if game has ended
        currentBoardResult = currentBoardState.getBoardResult(currentPlayerTurn);
        if(currentBoardResult != Game::GAME_ACTIVE)
        {
            // Game has ended, make sure board result has clear winner
            if(currentBoardResult == Game::GAME_OVER_PLAYER1_WIN || 
                currentBoardResult == Game::GAME_OVER_PLAYER2_WIN)
            {
                // If winner, count player win
                atomicInc(&resultCount[currentBoardResult - 1], 1);
            }

            // Count playout
            atomicInc(&resultCount[2], 1);

            // Reset search board states
            currentPlayerTurn = initPlayerTurn;
            currentBoardState = initalGameBoard;
        }

        // Sync threads before next stage
        __syncthreads();
    }

    // Copy gpu result out
    if(threadIdx.x < RESULT_SIZE)
        gpu_result_out[threadIdx.x] = resultCount[threadIdx.x];
    __syncthreads();
    
    // Save rng state
    curandStateConstant[threadIdx.x] = curandStateLocal;
}

namespace Player {

MonteCarloHybridPlayer::MonteCarloHybridPlayer()
{
    // Allocate space for prng states on device
    cudaMalloc(&devMTGPStates, sizeof(curandStateMtgp32));

    /* Allocate space for MTGP kernel parameters */
    cudaMalloc(&devKernelParams, sizeof(mtgp32_kernel_params));

    /* Reformat from predefined parameter sets to kernel format, */
    /* and copy kernel parameters to device memory               */
    curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);

    /* Initialize one state per thread block */
    curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, 1024, time(NULL));

    // Save curand state pointer to constant memory
    cudaMemcpyToSymbol(curandStateConstant, devMTGPStates, sizeof(devMTGPStates));
}

MonteCarloHybridPlayer::~MonteCarloHybridPlayer()
{
    cudaFree(devKernelParams);
    cudaFree(devMTGPStates);
}

// run a single simulation from the selected node
void MonteCarloHybridPlayer::simulation() {

    printf("EEEEEE\n");

    Game::GameBoard gameBoard = m_selectedNode->boardState;
    playernum_t playerTurn = m_selectedNode->playerNum;

    Game::boardresult_t result = gameBoard.getBoardResult(playerTurn);

    // 
    gpu_count_t* gpu_result_dev;
    cudaMalloc(&gpu_result_dev, sizeof(gpu_result));

    // Copy information to constant memory
    cudaMemcpyToSymbol(initPlayerTurn, &playerTurn, sizeof(playernum_t));
    cudaMemcpyToSymbol(initalGameBoard, &gameBoard, sizeof(Game::GameBoard));

    // Launch kernel
    simulationKernel<<<1, 1024>>>(gpu_result_dev);

    // Copy result back
    gpu_result gpu_result_host;
    cudaMemcpy(&gpu_result_host, gpu_result_dev, sizeof(gpu_result), cudaMemcpyDeviceToHost);

    // Count wins from kernel
    m_selectedNode->numWins += gpu_result_host.winCount[m_rootNode->playerNum];

    // Mark node as simulated
    m_selectedNode->simulated = true;
}

}