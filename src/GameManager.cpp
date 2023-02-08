#include <iostream>

#include "GameManager.h"

namespace Mancala {

GameManager::GameManager() {

} // end default constructor

void GameManager::initGame(std::shared_ptr<Player> player1, std::shared_ptr<Player> player2) {
    m_player1 = player1;
    m_player1->setPlayerNumber(1);
    m_player2 = player2;
    m_player2->setPlayerNumber(2);

    m_gameBoard.player1Goal = 0;
    m_gameBoard.player2Goal = 0;

    for(int i = 0; i < m_gameBoard.pits.size(); ++i) {
        m_gameBoard.pits[i] = 4;
    } // end for
} // end method initGame

void GameManager::startGame() {

    std::cout << "GAME STARTING" << std::endl;

    int winner = -1; // no winner yet

    while(winner == -1) {

        // Get move from player
        int move = -1;
        if(m_playerTurn == 1) {
            move = m_player1->makeMove(m_gameBoard);
        } else {
            move = m_player2->makeMove(m_gameBoard);
        } // end if

        // Make move on board
        bool repeatTurn = makeMoveOnBoard(move);

        // Are we in an end state?
        if(isEndState()) {
            winner = (m_gameBoard.player1Goal > m_gameBoard.player2Goal) ? 1 : 2;

            if (m_gameBoard.player1Goal == m_gameBoard.player2Goal) {
                std::cout << "TIE GAME?" << std::endl;
            } // end if
        } // end if

        if(!repeatTurn) {
            if(m_playerTurn == 1) {
                ++m_playerTurn;
            } else {
                --m_playerTurn;
            } // end if
        } // end if

    } // end while

    std::cout << "Winner Is: " << winner << std::endl;
} // end method startGame

bool GameManager::makeMoveOnBoard(int move) {
    int idx = move;
    if(idx >= 6) ++idx;
    int stones = m_gameBoard.pits[idx];
    m_gameBoard.pits[idx] = 0;

    bool repeatMove = false;

    while (stones > 0) {
        if(idx < 6) {
            ++m_gameBoard.pits[idx];
            repeatMove = false;
        } else if(idx == 6) {
            ++m_gameBoard.player1Goal;
            repeatMove = (m_playerTurn == 2);
        } else if(idx < 13) {
            ++m_gameBoard.pits[idx-1];
            repeatMove = false;
        } else {
            ++m_gameBoard.player2Goal;
            idx = 0;
            repeatMove = (m_playerTurn == 1);
        } // end if
        --stones;

        if(stones == 0) {
            // do nothing
        } else if(idx > 12) {
            idx = 0;
        } else {
            ++idx;
        } // end if
    } // end while

    int pitIdx = -1;
    if(idx < 6) {
        pitIdx = idx;
    } else if ((idx > 6) && (idx < 13)) {
        pitIdx = idx-1;
    } // end if

    if(pitIdx != -1) {
        if((m_playerTurn == 1) && (m_gameBoard.pits[pitIdx] == 1) && (pitIdx < 6)) {
            int mirrorIdx = 11 - pitIdx;
            m_gameBoard.player1Goal += m_gameBoard.pits[mirrorIdx];
            m_gameBoard.pits[mirrorIdx] = 0;
        } else if((m_playerTurn == 2) && (m_gameBoard.pits[pitIdx] == 1) && (pitIdx >= 6)) {
            int mirrorIdx = 11 - pitIdx;
            m_gameBoard.player2Goal += m_gameBoard.pits[mirrorIdx];
            m_gameBoard.pits[mirrorIdx] = 0;
        } // end if
    } // end if

    return repeatMove;
} // end method makeMoveOnBoard

bool GameManager::isEndState() {

    bool player1OutOfMoves = true;
    for(int i = 0; i < 6; ++i) {
        if(m_gameBoard.pits[i] > 0) {
            player1OutOfMoves = false;
            break;
        }
    } // end for

    bool player2OutOfMoves = true;
    for(int i = 6; i < 12; ++i) {
        if(m_gameBoard.pits[i] > 0) {
            player2OutOfMoves = false;
            break;
        }
    } // end for

    return (player1OutOfMoves || player2OutOfMoves);
} // end method isEndState

} // end namespace Mancala