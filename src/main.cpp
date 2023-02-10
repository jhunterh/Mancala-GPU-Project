#include <fstream>
#include <iostream>

#include "MancalaStatic.h"
#include "GameManager.h"
#include "RandomPlayer.h"

int main(int argc, char **argv) {

    if(argc != 3) {
        std::cout << "USAGE: ./mancala <PlayerType> <PlayerType>" << std::endl;
        return 1;
    }

    Mancala::PlayerType player1Type = (Mancala::PlayerType)atoi(argv[1]);
    Mancala::PlayerType player2Type = (Mancala::PlayerType)atoi(argv[2]);

    std::cout << "Player 1: " << Mancala::getPlayerString(player1Type) << std::endl;
    std::cout << "Player 2: " << Mancala::getPlayerString(player2Type) << std::endl;

    std::shared_ptr<Mancala::Player> player1(new Mancala::RandomPlayer());
    std::shared_ptr<Mancala::Player> player2(new Mancala::RandomPlayer());

    Mancala::GameManager gm;
    gm.initGame(player1, player2);

    gm.startGame();

    return 0;
} // end main