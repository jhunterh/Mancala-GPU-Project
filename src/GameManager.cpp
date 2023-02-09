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

    // initialize game vector
    std::vector<int> gameVec;
    for(int i = 0; i < 14; ++i) {
        if(i < 6) {
            gameVec.push_back(m_gameBoard.pits[i]);
        } else if(i == 6) {
            gameVec.push_back(m_gameBoard.player1Goal);
        } else if(i < 13) {
            gameVec.push_back(m_gameBoard.pits[i-1]);
        } else {
            gameVec.push_back(m_gameBoard.player2Goal);
        } // end if
    } // end for

    bool repeatTurn = false;

    int idx = (move < 6) ? move : move+1;

    int stones = gameVec[idx];
    gameVec[idx] = 0;

    // distribute stones
    while(stones > 0) {
        idx = (idx == 13) ? 0 : (idx + 1);
        ++gameVec[idx];
        --stones;
    } // end while

    // check for special cases
    if((idx == 6) && (m_playerTurn == 1)) {
        repeatTurn = true;
    } else if((idx == 13) && (m_playerTurn == 2)) {
        repeatTurn = true;
    } // end if

    if((idx < 6) && (m_playerTurn == 1) && (gameVec[idx] == 1)) {
        int mirrorIdx = 12-idx;
        gameVec[6] += gameVec[mirrorIdx];
        gameVec[mirrorIdx] = 0;
    } else if((idx > 6) && (idx < 13) && (m_playerTurn == 2) && (gameVec[idx] == 1)) {
        int mirrorIdx = 12-idx;
        gameVec[13] += gameVec[mirrorIdx];
        gameVec[mirrorIdx] = 0;
    } // end if

    // convert game vector back to board struct
    for(int i = 0; i < 14; ++i) {
        if(i < 6) {
            m_gameBoard.pits[i] = gameVec[i];
        } else if(i == 6) {
            m_gameBoard.player1Goal = gameVec[i];
        } else if(i < 13) {
            m_gameBoard.pits[i-1] = gameVec[i];
        } else {
            m_gameBoard.player2Goal = gameVec[i];
        } // end if
    } // end for

    return repeatTurn;

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