#include "MonteCarloUtility.h"

#include <ctime>
#include <iostream>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_cuda.h>

__device__ __constant__ Player::playernum_t initPlayerTurn;
__device__ __constant__ Game::GameBoard initalGameBoard;
__device__ gpu_result gpuResultGlobal;
__device__ __constant__ deterministic_data deterministicData;

__global__ void simulationKernel(unsigned long long seed)
{
    // Initialize result memory (to be copied out later)
    // First two are players, third is total play count
    __shared__ gpu_result gpuResultLocal;

    curandState_t curandState;
    unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
    seed += idx;
    // Init curand state
    curand_init(seed, 0, 0, &curandState);

    // number of moves simulated by this thread
    unsigned int numMovesSimulated = 0;

    // Init result to 0
    if(threadIdx.x == 0)
        gpuResultLocal = {};

    __syncthreads();

    // Init search states
    Player::playernum_t currentPlayerTurn = initPlayerTurn;
    Game::GameBoard currentBoardState = initalGameBoard;
    Game::boardresult_t currentBoardResult = currentBoardState.getBoardResult(currentPlayerTurn);
    deterministic_data deterministicDataReg = deterministicData;

    // Pick random move and execute
    while(currentBoardResult == Game::GAME_ACTIVE) 
    {
        // Get list of possible moves
        Game::movelist_t moveList;
        Game::movecount_t moveCount = currentBoardState.getMoves(moveList, currentPlayerTurn);
        
        // Prevent floating point exceptions
        if(moveCount > 0)
        {   
            // Select random move
            // NOTE: No need to worry about control divergence here since
            //       all threads will take the same path
            Game::move_t selectedMove;
            if(deterministicDataReg.isPreDetermined)
            {
                selectedMove = moveList[deterministicDataReg.value];
            }
            else
            {
                selectedMove = moveList[curand(&curandState) % moveCount];
            }
                
            // Execute random move
            Game::moveresult_t moveResult = currentBoardState.executeMove(selectedMove, currentPlayerTurn);
            ++numMovesSimulated;

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

        // Game has ended, make sure board result has clear winner
        if(currentBoardResult == Game::GAME_OVER_PLAYER1_WIN || 
            currentBoardResult == Game::GAME_OVER_PLAYER2_WIN)
        {
            // If winner, count player win
            atomicAdd(&gpuResultLocal.winCount[currentBoardResult - 1], 1);
        }
    }

    // Count playout
    atomicAdd(&gpuResultLocal.playCount, 1);

    atomicAdd(&gpuResultLocal.numMovesSimulated, numMovesSimulated);
    __syncthreads();

    // Copy gpu result out
    if(threadIdx.x == 0)
    {
        atomicAdd(&gpuResultGlobal.winCount[0], gpuResultLocal.winCount[0]);
        atomicAdd(&gpuResultGlobal.winCount[1], gpuResultLocal.winCount[1]);
        atomicAdd(&gpuResultGlobal.playCount, gpuResultLocal.playCount);
        atomicAdd(&gpuResultGlobal.numMovesSimulated, gpuResultLocal.numMovesSimulated);
    }
}

void simulationGPU(gpu_result* gpu_result_out, Game::GameBoard gameBoard, Player::playernum_t playerTurn, deterministic_data deterministicDataHost)
{
    // Copy information to constant memory
    checkCudaErrors(cudaMemcpyToSymbol(initalGameBoard, &gameBoard, sizeof(Game::GameBoard)));
    checkCudaErrors(cudaMemcpyToSymbol(initPlayerTurn, &playerTurn, sizeof(Player::playernum_t)));
    checkCudaErrors(cudaMemcpyToSymbol(deterministicData, &deterministicDataHost, sizeof(deterministic_data)));
    gpu_result gpuResult = {};
    checkCudaErrors(cudaMemcpyToSymbol(gpuResultGlobal, &gpuResult, sizeof(gpu_result)));

    // Launch kernel
    simulationKernel<<<GRID_SIZE, BLOCK_SIZE>>>(time(NULL));
    checkCudaErrors(cudaGetLastError());

    // Copy result back
    checkCudaErrors(cudaMemcpyFromSymbol(gpu_result_out, gpuResultGlobal, sizeof(gpu_result)));
}