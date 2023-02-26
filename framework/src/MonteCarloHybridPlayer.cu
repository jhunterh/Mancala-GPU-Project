#include "MonteCarloHybridPlayer.h"
#include "GameTypes.h"
#include "RandomPlayer.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define PLAYCOUNT_THRESHOLD 1000

struct gpu_result
{
    unsigned int player1WinCount = 0;
    unsigned int player2WinCount = 0;
    unsigned int playCount = 0;
}

__device__ __constant__ Player::playernum_t initPlayerTurn;
__device__ __constant__ Game::GameBoard initalGameBoard;

__global__ void simulationKernel(gpu_result* gpu_result_out)
{
    // Initialize result memory (to be copied out)
    __shared__ gpu_result gpuResult();

    // Init search states
    Player::playernum_t currentPlayerTurn = initPlayerTurn;
    Game::GameBoard currentBoardState = initalGameBoard;
    Game::boardresult_t currentBoardResult = Game::GAME_ACTIVE;

    // Do simulations until threshold reached
    while(gpuResult.playCount < PLAYCOUNT_THRESHOLD)
    {
        // Pick random move and execute
        // TODO: Use CURAND
        Game::move_t selectedMove; // = player.selectMove(currentBoardState, currentPlayerTurn);
        Game::moveresult_t moveResult = currentBoardState.executeMove(selectedMove, currentPlayerTurn);

        // Check Move
        // TODO: Add move result checking
        if (moveResult == Game::MOVE_SUCCESS) {
            if (currentPlayerTurn == Player::PLAYER_NUMBER_2) {
                currentPlayerTurn = Player::PLAYER_NUMBER_1;
            } else {
                currentPlayerTurn = Player::PLAYER_NUMBER_2;
            }
        } /*else if (moveResult == Game::MOVE_INVALID) {
            std::cout << "Invalid Move" << std::endl;
        }*/

        // Check if game has ended
        currentBoardResult = currentBoardState.getBoardResult(currentPlayerTurn);
        if(currentBoardResult != Game::GAME_ACTIVE)
        {
            // Game has ended, check who won
            if(currentBoardResult = Game::GAME_OVER_PLAYER1_WIN)
            {
                // Count player 1 win
                atomicInc(gpuResult.player1WinCount);
            }
            else if(currentBoardResult = Game::GAME_OVER_PLAYER2_WIN)
            {
                // Count player 2 win
                atomicInc(gpuResult.player2WinCount);
            }

            // Count playout
            atomicInc(gpuResult.playCount);

            // Reset search board states
            currentPlayerTurn = initPlayerTurn;
            Game::GameBoard currentBoardState = initalGameBoard;
            Game::boardresult_t currentBoardResult = Game::GAME_ACTIVE;
        }

        // Sync threads before next stage
        __syncthreads();
    }

    // Copy gpu result out
    if(threadIdx.x == 0)
        gpu_result_out = gpuResult;
    __syncthreads();
}

namespace Player {

// run a single simulation from the selected node
void MonteCarloHybridPlayer::simulation() {
    // Declare two random players to duke it out
    RandomPlayer player;

    Game::GameBoard gameBoard = m_selectedNode->boardState;
    playernum_t playerTurn = m_selectedNode->playerNum;

    Game::boardresult_t result = gameBoard.getBoardResult(playerTurn);

    // CUDA SECTION

    // CUDA SECTION END
    
    if(GameUtils::getPlayerFromBoardResult(result) == m_rootNode->playerNum) {
        ++m_selectedNode->numWins;
    }

    m_selectedNode->simulated = true;
}

}