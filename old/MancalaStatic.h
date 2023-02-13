#ifndef _MANCALATYPES_H
#define _MANCALATYPES_H

#include <string>
#include <map>
#include <vector>
#include <iomanip>

namespace Mancala {

static const int P1GOAL = 6;
static const int P2GOAL = 13;

static std::vector<int> getValidMoves(std::vector<int> board, int playerTurn) {
    std::vector<int> validMoves;
    
    if (playerTurn == 1) {
        for(int i = 0; i < P1GOAL; ++i) {
            if(board[i] > 0) validMoves.push_back(i);
        } // end for
    } else {
        for(int i = P1GOAL+1; i < P2GOAL; ++i) {
            if(board[i] > 0) validMoves.push_back(i);
        } // end for
    } // end if

    return validMoves;
} // end function getValidMoves

enum PlayerType {
    RANDOM,
    MONTE_CPU,
    MONTE_GPU
}; // end enum PlayerType

static std::string getPlayerString(PlayerType type) {
    std::map<PlayerType,std::string> m;
    m[PlayerType::RANDOM] = "RANDOM";
    m[PlayerType::MONTE_CPU] = "MONTE_CPU";
    m[PlayerType::MONTE_GPU] = "MONTE_GPU";

    auto it = m.find(type);
    if(it == m.end()) {
        return "Unknown Type"; // TODO: Logger
    } else {
        return it->second;
    } // end if
} // end function getPlayerString

static void printBoard(std::vector<int> board) {
    const std::string SPACE = " ";

    for(int i = P2GOAL-1; i > P1GOAL; --i) {
        std::cout << SPACE << std::setw(2) << board[i];
    } // end for

    std::cout << std::endl << board[P2GOAL] << std::setw(19) << board[P1GOAL] << std::endl;

    for(int i = 0; i < P1GOAL; ++i) {
        std::cout << SPACE << std::setw(2) << board[i];
    } // end for

    std::cout << std::endl;
} // end function printBoard

} // end namespace Mancala

#endif