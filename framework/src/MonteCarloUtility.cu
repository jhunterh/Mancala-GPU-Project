#include "MonteCarloUtility.h"

#include <ctime>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

__device__ __constant__ Player::playernum_t initPlayerTurn;
__device__ __constant__ Game::GameBoard initalGameBoard;
__device__ gpu_result gpuResultGlobal;
__device__ curandState_t curandStatesGlobal[BLOCK_SIZE];

// Kernel to init curand
__global__ void curandInitKernel(unsigned long long seed)
{
    // Init curand states
    curand_init(seed, threadIdx.x, 0, &curandStatesGlobal[threadIdx.x]);
}

__global__ void simulationKernel()
{
    // Initialize result memory (to be copied out later)
    // First two are players, third is total play count
    __shared__ gpu_result gpuResultLocal;

    // Init result to 0
    if(threadIdx.x == 0)
        gpuResultLocal = {};
    __syncthreads();

    // Grab curand state 
    curandState_t curandStateLocal = curandStatesGlobal[threadIdx.x];

    // Init search states
    Player::playernum_t currentPlayerTurn = initPlayerTurn;
    Game::GameBoard currentBoardState = initalGameBoard;
    Game::boardresult_t currentBoardResult = Game::GAME_ACTIVE;

    // Do simulations until threshold reached
    while(gpuResultLocal.playCount < PLAYCOUNT_THRESHOLD_GPU)
    {
        // Pick random move and execute

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
                gpuResultLocal.winCount[currentBoardResult - 1]++;
            }

            // Count playout
            gpuResultLocal.playCount++;

            // Reset search board states
            currentPlayerTurn = initPlayerTurn;
            currentBoardState = initalGameBoard;
        }

        // Sync threads before next stage
        // Prevents reading and writing at same time
        __syncthreads();
    }

    // Save rng state
    curandStatesGlobal[threadIdx.x] = curandStateLocal;

    // Copy gpu result out
    if(threadIdx.x == 0)
        gpuResultGlobal = gpuResultLocal;
    __syncthreads();
}

void curandInit()
{
    // Init curand
    curandInitKernel<<<1, BLOCK_SIZE>>>(time(NULL));
    checkCudaErrors(cudaGetLastError());
}

void simulationGPU(gpu_result* gpu_result_out, Game::GameBoard gameBoard, Player::playernum_t playerTurn)
{
    // Copy information to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(initalGameBoard, &gameBoard, sizeof(Game::GameBoard)));
    checkCudaErrors(cudaMemcpyToSymbol(initPlayerTurn, &playerTurn, sizeof(Player::playernum_t)));

    // Launch kernel
    simulationKernel<<<1, BLOCK_SIZE>>>();
    checkCudaErrors(cudaGetLastError());

    // Copy result back
    checkCudaErrors(cudaMemcpyFromSymbol(gpu_result_out, gpuResultGlobal, sizeof(gpu_result)));
}