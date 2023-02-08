#include <ctime>
#include <cstdlib>
#include <iostream>

#include "RandomPlayer.h"
#include "MancalaTypes.h"

namespace Mancala {

    int RandomPlayer::makeMove(MancalaBoard board) {

        std::cout << "my turn: " << getPlayerNumber() << std::endl;

        int playerNum = getPlayerNumber();

        std::vector<int> validMoves = getValidMoves(board, playerNum);

        srand(time(NULL));

        int randomValue = (rand() % validMoves.size());

        std::cout << "I choose: " << randomValue << " " << validMoves[randomValue] << std::endl;

        return validMoves[randomValue];
    } // end method makeMove

} // end namespace Mancala