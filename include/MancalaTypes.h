#ifndef _MANCALATYPES_H
#define _MANCALATYPES_H

#include <string>
#include <map>
#include <vector>

namespace Mancala {

struct MancalaBoard {
    std::vector<int> pits{std::vector<int>(12,0)};
    int player1Goal = 0;
    int player2Goal = 0;
}; // end struct MancalaBoard

std::vector<int> getValidMoves(MancalaBoard *board, int playerTurn) {
    std::vector<int> validMoves;
    
    if (playerTurn == 1) {
        for(int i = 0; i < 6; ++i) {
            if(board->pits[i] > 0) validMoves.push_back(i);
        } // end for
    } else {
        for(int i = 6; i < 12; ++i) {
            if(board->pits[i] > 0) validMoves.push_back(i);
        } // end for
    } // end if

    return validMoves;
} // end function getValidMoves

enum PlayerType {
    RANDOM,
    MONTE_CPU,
    MONTE_GPU
}; // end enum PlayerType

std::string getPlayerString(PlayerType type) {
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

} // end namespace Mancala

#endif