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

    m_gameBoard[P1GOAL] = 0;
    m_gameBoard[P2GOAL] = 0;
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

    std::cout << "Winner Is: Player " << winner << std::endl;
    printBoard(m_gameBoard);
} // end method startGame

bool GameManager::makeMoveOnBoard(int move) {

    printBoard(m_gameBoard);

    std::cout << "Player " << m_playerTurn << " Makes Move: " << move << std::endl << std::endl;

    bool repeatTurn = false;

    int idx = move;

    int stones = m_gameBoard[idx];
    m_gameBoard[idx] = 0;

    // distribute stones
    while(stones > 0) {
        idx = (idx == P2GOAL) ? 0 : (idx + 1);

        if((m_playerTurn == 1) && (idx == P2GOAL)) {
            // skip
        } else if ((m_playerTurn == 2) && (idx == P1GOAL)) {
            // skip
        } else {
            ++m_gameBoard[idx];
            --stones;
        } // end if
    } // end while

    // check for special cases
    if((idx == P1GOAL) && (m_playerTurn == 1)) {
        repeatTurn = true;
    } else if((idx == P2GOAL) && (m_playerTurn == 2)) {
        repeatTurn = true;
    } // end if

    if((idx < P1GOAL) && (m_playerTurn == 1) && (m_gameBoard[idx] == 1)) {
        int mirrorIdx = 12-idx;
        if(m_gameBoard[mirrorIdx] > 0) {
            m_gameBoard[P1GOAL] += m_gameBoard[mirrorIdx];
            m_gameBoard[mirrorIdx] = 0;
            ++m_gameBoard[P1GOAL];
            m_gameBoard[idx] = 0;
        } // end if
    } else if((idx > P1GOAL) && (idx < P2GOAL) && (m_playerTurn == 2) && (m_gameBoard[idx] == 1)) {
        int mirrorIdx = 12-idx;
        if(m_gameBoard[mirrorIdx] > 0) {
            m_gameBoard[P2GOAL] += m_gameBoard[mirrorIdx];
            m_gameBoard[mirrorIdx] = 0;
            ++m_gameBoard[P2GOAL];
            m_gameBoard[idx] = 0;
        } // end if
    } // end if

    return repeatTurn;

} // end method makeMoveOnBoard

int GameManager::determineWinner() {
    int winner = -1;

    bool player1OutOfMoves = true;
    for(int i = 0; i < P1GOAL; ++i) {
        if(m_gameBoard[i] > 0) {
            player1OutOfMoves = false;
            break;
        } // end if
    } // end for

    bool player2OutOfMoves = true;
    for(int i = P1GOAL+1; i < P2GOAL; ++i) {
        if(m_gameBoard[i] > 0) {
            player2OutOfMoves = false;
            break;
        } // end if
    } // end for

    if(player1OutOfMoves) {
        for(int i = P1GOAL+1; i < P2GOAL; ++i) {
            m_gameBoard[P2GOAL] += m_gameBoard[i];
            m_gameBoard[i] = 0;
        } // end for
        winner = (m_gameBoard[P1GOAL] > m_gameBoard[P2GOAL]) ? 1 : 2;
    } else if(player2OutOfMoves) {
        for(int i = 0; i < P1GOAL; ++i) {
            m_gameBoard[P1GOAL] += m_gameBoard[i];
            m_gameBoard[i] = 0;
        } // end for
        winner = (m_gameBoard[P1GOAL] > m_gameBoard[P2GOAL]) ? 1 : 2;
    } // end if

    if ((m_gameBoard[P1GOAL] == m_gameBoard[P2GOAL]) && (player1OutOfMoves || player2OutOfMoves)) {
        std::cout << "TIE GAME?" << std::endl;
    } // end if

    return winner;
} // end method determineWinner

} // end namespace Mancala