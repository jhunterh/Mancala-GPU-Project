# Mancala-GPU-Project
This is the repository for the Mancala Monte Carlo Tree Search Implementation for General Purpose GPU Programming.

## Game Simulation Framework
This includes a framework for simulating other games as well. To add a new game, copy the example folder under games/ to the name of your desired game, define the necessary types in game.h, and implement the GameBoard functions in GameBoard.cpp. Then run the following command:
```
$ make game=example
```

where example is the name of your custom game.

## Example Simulation
To run the included example simulation, run the follwing command:
```
$ ./build/bin/simulation Player1Type Player2Type
```
where Player1Type and Player2Type are the desired AI types in numerical format (ex. 0). You can find out the current implemented types by running the simulation executable without arguments.

Note: the example simulation only currently supports a two person game.