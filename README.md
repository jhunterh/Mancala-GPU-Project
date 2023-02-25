# Mancala-GPU-Project
This is the repository for the Mancala Monte Carlo Tree Search Implementation for General Purpose GPU Programming.

## Game Simulation Framework
This includes a framework for simulating other games as well. To add a new game, copy the example folder under games/ to the name of your desired game, define the necessary types in game.h, and implement the GameBoard functions in GameBoard.cpp. Then run the following command:
```
$ make game=example
```

where example is the name of your custom game.

## Game Guidlines
Any game that is made for this framework must follow these rules:
1. The game must have two players
2. The game board must be able to be described with a simple vector or array
3. Each move during a game can only result in one of the two players getting marked to have the next turn.
4. Each game must have a finished state in which one of two players is a winner or the result is a tie.

## Example Simulation
To run the included example simulation, run the follwing command:
```
$ ./build/bin/simulation Player1Type Player2Type
```
where Player1Type and Player2Type are the desired AI types in numerical format (ex. 0). You can find out the current implemented types by running the simulation executable without arguments.

Note: the example simulation only currently supports a two person game.