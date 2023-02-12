#include <ctime>
#include <cstdlib>
#include <iostream>

#include "SimulationPolicyCPU.h"
#include "MancalaStatic.h"

namespace Mancala {

    float SimulationPolicyCPU::runSimulation(std::vector<int> gameState, int playerTurn) {
        float winPercentage = 0.0f;
        m_gameState = gameState;
        m_playerTurn = playerTurn;

        int winner = -1; // no winner yet

        while(winner == -1) {

            // Get move from player
            int move = -1;
            if(m_playerTurn == 1) {
                move = m_player1.makeMove(m_gameState);
            } else {
                move = m_player2.makeMove(m_gameState);
            } // end if

            // Make move on board
            bool repeatTurn = makeMoveOnBoard(move);

            // Are we in an end state?
            winner = determineWinner();
            if(winner != -1) {
                break;
            } // end if

            if(!repeatTurn) {
                if(m_playerTurn == 1) {
                    ++m_playerTurn;
                } else {
                    --m_playerTurn;
                } // end if
            } // end if

        } // end while

        int simulationPlayer = (playerTurn == 1) ? 2 : 1;

        return winPercentage = (simulationPlayer == winner) ? 1.0f : 0.0f;
    } // end method runSimulation

    bool SimulationPolicyCPU::makeMoveOnBoard(int move) {

    bool repeatTurn = false;

    int idx = move;

    int stones = m_gameState[idx];
    m_gameState[idx] = 0;

    // distribute stones
    while(stones > 0) {
        idx = (idx == P2GOAL) ? 0 : (idx + 1);

        if((m_playerTurn == 1) && (idx == P2GOAL)) {
            // skip
        } else if ((m_playerTurn == 2) && (idx == P1GOAL)) {
            // skip
        } else {
            ++m_gameState[idx];
            --stones;
        } // end if
    } // end while

    // check for special cases
    if((idx == P1GOAL) && (m_playerTurn == 1)) {
        repeatTurn = true;
    } else if((idx == P2GOAL) && (m_playerTurn == 2)) {
        repeatTurn = true;
    } // end if

    if((idx < P1GOAL) && (m_playerTurn == 1) && (m_gameState[idx] == 1)) {
        int mirrorIdx = 12-idx;
        if(m_gameState[mirrorIdx] > 0) {
            m_gameState[P1GOAL] += m_gameState[mirrorIdx];
            m_gameState[mirrorIdx] = 0;
            ++m_gameState[P1GOAL];
            m_gameState[idx] = 0;
        } // end if
    } else if((idx > P1GOAL) && (idx < P2GOAL) && (m_playerTurn == 2) && (m_gameState[idx] == 1)) {
        int mirrorIdx = 12-idx;
        if(m_gameState[mirrorIdx] > 0) {
            m_gameState[P2GOAL] += m_gameState[mirrorIdx];
            m_gameState[mirrorIdx] = 0;
            ++m_gameState[P2GOAL];
            m_gameState[idx] = 0;
        } // end if
    } // end if

    return repeatTurn;

} // end method makeMoveOnBoard

int SimulationPolicyCPU::determineWinner() {
    int winner = -1;

    bool player1OutOfMoves = true;
    for(int i = 0; i < P1GOAL; ++i) {
        if(m_gameState[i] > 0) {
            player1OutOfMoves = false;
            break;
        } // end if
    } // end for

    bool player2OutOfMoves = true;
    for(int i = P1GOAL+1; i < P2GOAL; ++i) {
        if(m_gameState[i] > 0) {
            player2OutOfMoves = false;
            break;
        } // end if
    } // end for

    if(player1OutOfMoves) {
        for(int i = P1GOAL+1; i < P2GOAL; ++i) {
            m_gameState[P2GOAL] += m_gameState[i];
            m_gameState[i] = 0;
        } // end for
        winner = (m_gameState[P1GOAL] > m_gameState[P2GOAL]) ? 1 : 2;
    } else if(player2OutOfMoves) {
        for(int i = 0; i < P1GOAL; ++i) {
            m_gameState[P1GOAL] += m_gameState[i];
            m_gameState[i] = 0;
        } // end for
        winner = (m_gameState[P1GOAL] > m_gameState[P2GOAL]) ? 1 : 2;
    } // end if

    return winner;
} // end method determineWinner

} // end namespace Mancala