#include <iostream>
#include <memory>
#include <ctime>
#include <cstdlib>
#include <random>

#include "game.h"
#include "MonteCarloPlayer.h"
#include "MonteCarloPlayerMT.h"
#include "MonteCarloHybridPlayer.h"
#include "MonteCarloTypes.h"
#include "GameBoard.h"

std::shared_ptr<MonteCarlo::TreeNode> generateSelectionTree()
{
    std::shared_ptr<MonteCarlo::TreeNode> root = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> selected = root;
    srand(time(NULL));

    // generates a tree 100 layers deep with a random number
    // of nodes per layer and each node having a random value
    for(int i = 0; i < 100; ++i) {
        int numChildNodes = (rand() % 10) + 1;
        for(int j = 0; j < numChildNodes; ++j) {
            std::shared_ptr<MonteCarlo::TreeNode> newNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
            newNode->parentNode = selected;
            newNode->value = ((double)(rand() % 100)) / 100;
            newNode->numTimesVisited = 99 - i;
            selected->childNodes.push_back(newNode);
        }
        selected = selected->childNodes[0];
    }

    return root;
}

void selectionTest()
{
    std::shared_ptr<MonteCarlo::TreeNode> rootNode = generateSelectionTree();

    Player::MonteCarloPlayer reference; // single threaded CPU
    Player::MonteCarloPlayerMT uutMT; // multi-threaded CPU
    Player::MonteCarloHybridPlayer uutGPU; // GPU

    reference.setRootNode(rootNode);
    reference.setSelectedNode(rootNode);
    reference.selection();

    uutMT.setRootNode(rootNode);
    uutMT.setSelectedNode(rootNode);
    uutMT.selection();

    uutGPU.setRootNode(rootNode);
    uutGPU.setSelectedNode(rootNode);
    uutGPU.selection();

    std::cout << std::endl;

    if(reference.getSelectedNode() == uutMT.getSelectedNode())
    {
        std::cout << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
    }

    if(reference.getSelectedNode() == uutGPU.getSelectedNode())
    {
        std::cout << __PRETTY_FUNCTION__ << " Hybrid Passed" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " Hybrid Failed" << std::endl;
    }
}

void expansionTest()
{
    Player::MonteCarloPlayer reference; // single threaded CPU
    Player::MonteCarloPlayerMT uutMT; // multi-threaded CPU
    Player::MonteCarloHybridPlayer uutGPU; // GPU

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    referenceNode->boardState.initBoard();
    referenceNode->playerNum = Player::PLAYER_NUMBER_1;
    referenceNode->simulated = true;

    std::shared_ptr<MonteCarlo::TreeNode> uutMTNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutMTNode->boardState.initBoard();
    uutMTNode->playerNum = Player::PLAYER_NUMBER_1;
    uutMTNode->simulated = true;

    std::shared_ptr<MonteCarlo::TreeNode> uutGPUNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutGPUNode->boardState.initBoard();
    uutGPUNode->playerNum = Player::PLAYER_NUMBER_1;
    uutGPUNode->simulated = true;

    reference.setSelectedNode(referenceNode);
    reference.expansion();

    uutMT.setSelectedNode(uutMTNode);
    uutMT.expansion();

    uutGPU.setSelectedNode(uutGPUNode);
    uutGPU.expansion();

    std::cout << std::endl;

    if(referenceNode->childNodes.size() != uutMTNode->childNodes.size())
    {
        std::cout << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
    }
    else
    {
        bool pass = true;
        for(int i = 0; i < referenceNode->childNodes.size(); ++i)
        {
            if(!MonteCarlo::nodeCompare(referenceNode->childNodes[i], uutMTNode->childNodes[i]))
            {
                pass = false;
                break;
            }
        }

        if(pass)
        {
            std::cout << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
        }
        else
        {
            std::cout << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
        }
    }

    if(referenceNode->childNodes.size() != uutGPUNode->childNodes.size())
    {
        std::cout << __PRETTY_FUNCTION__ << " Hybrid Failed" << std::endl;
    }
    else
    {
        bool pass = true;
        for(int i = 0; i < referenceNode->childNodes.size(); ++i)
        {
            if(!MonteCarlo::nodeCompare(referenceNode->childNodes[i], uutGPUNode->childNodes[i]))
            {
                pass = false;
                break;
            }
        }

        if(pass)
        {
            std::cout << __PRETTY_FUNCTION__ << " Hybrid Passed" << std::endl;
        }
        else
        {
            std::cout << __PRETTY_FUNCTION__ << " Hybrid Failed" << std::endl;
        }
    }
}

void simulationTest()
{
    Player::MonteCarloPlayer reference; // single threaded CPU
    Player::MonteCarloPlayerMT uutMT; // multi-threaded CPU
    Player::MonteCarloHybridPlayer uutGPU; // GPU

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    Game::GameBoard gameBoard;
    gameBoard.scramble();
    referenceNode->boardState = gameBoard;
    referenceNode->playerNum = Player::PLAYER_NUMBER_1;

    std::shared_ptr<MonteCarlo::TreeNode> uutMTNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutMTNode->boardState = gameBoard;
    uutMTNode->playerNum = Player::PLAYER_NUMBER_1;

    std::shared_ptr<MonteCarlo::TreeNode> uutGPUNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutGPUNode->boardState = gameBoard;
    uutGPUNode->playerNum = Player::PLAYER_NUMBER_1;

    reference.setRootNode(referenceNode);
    uutMT.setRootNode(uutMTNode);
    uutGPU.setRootNode(uutGPUNode);

    reference.setSelectedNode(referenceNode);
    uutMT.setSelectedNode(uutMTNode);
    uutGPU.setSelectedNode(uutGPUNode);

    reference.setDeterministic(true, 0);
    uutMT.setDeterministic(true, 0);
    uutGPU.setDeterministic(true, 0);

    reference.simulation();
    uutMT.simulation();
    uutGPU.simulation();

    std::cout << std::endl;

    if(referenceNode->numWins == uutMTNode->numWins) 
    {
        std::cout << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
    }

    if(referenceNode->numWins == uutGPUNode->numWins) 
    {
        std::cout << __PRETTY_FUNCTION__ << " Hybrid Passed" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " Hybrid Failed" << std::endl;
    }
}

void backpropagationTest()
{
    std::uniform_real_distribution<double> dist1(0, 1);
    std::uniform_real_distribution<double> dist2(0, 50);

    std::default_random_engine generator;

    srand(time(NULL));

    Player::MonteCarloPlayer reference; // single threaded CPU
    Player::MonteCarloPlayerMT uutMT; // multi-threaded CPU
    Player::MonteCarloHybridPlayer uutGPU; // GPU

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> uutMTNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> uutGPUNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());

    double numWins = dist2(generator);
    int numTimesVisited = rand() % 101;

    referenceNode->numWins = numWins;
    referenceNode->numTimesVisited = numTimesVisited;

    uutMTNode->numWins = numWins;
    uutMTNode->numTimesVisited = numTimesVisited;

    uutGPUNode->numWins = numWins;
    uutGPUNode->numTimesVisited = numTimesVisited;

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode2 = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> uutMTNode2 = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> uutGPUNode2 = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());

    numWins = dist1(generator);
    numTimesVisited = 1;

    referenceNode2->numWins = numWins;
    referenceNode2->numTimesVisited = numTimesVisited;

    uutMTNode2->numWins = numWins;
    uutMTNode2->numTimesVisited = numTimesVisited;

    uutGPUNode2->numWins = numWins;
    uutGPUNode2->numTimesVisited = numTimesVisited;

    referenceNode2->parentNode = referenceNode;
    uutMTNode2->parentNode = uutMTNode;
    uutGPUNode2->parentNode = uutGPUNode;

    reference.setRootNode(referenceNode);
    reference.setSelectedNode(referenceNode2);
    reference.setExplorationParam(1);

    uutMT.setRootNode(uutMTNode);
    uutMT.setSelectedNode(uutMTNode2);
    uutMT.setExplorationParam(1);

    uutGPU.setRootNode(uutGPUNode);
    uutGPU.setSelectedNode(uutGPUNode2);
    uutGPU.setExplorationParam(1);

    reference.backpropagation();
    uutMT.backpropagation();
    uutGPU.backpropagation();

    double relErrMT = (referenceNode->value - uutMTNode->value)*(referenceNode->value - uutMTNode->value) 
                        + (referenceNode2->value - uutMTNode2->value)*(referenceNode2->value - uutMTNode2->value);
    double relErrGPU = (referenceNode->value - uutGPUNode->value)*(referenceNode->value - uutGPUNode->value) 
                        + (referenceNode2->value - uutGPUNode2->value)*(referenceNode2->value - uutGPUNode2->value);

    std::cout << std::endl;

    if(relErrMT < 0.01)
    {
        std::cout << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
    }

    if(relErrGPU < 0.01)
    {
        std::cout << __PRETTY_FUNCTION__ << " Hybrid Passed" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " Hybrid Failed" << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    selectionTest();
    expansionTest();
    simulationTest();
    backpropagationTest();
    return 0;
}