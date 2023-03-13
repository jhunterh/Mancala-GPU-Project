#include <iostream>
#include <memory>
#include <ctime>
#include <cstdlib>
#include <random>

#include "game.h"
#include "MonteCarloPlayer.h"
#include "MonteCarloPlayerMT.h"
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

    Player::MonteCarloPlayer reference;
    Player::MonteCarloPlayerMT uut;

    reference.setRootNode(rootNode);
    reference.setSelectedNode(rootNode);
    reference.selection();

    uut.setRootNode(rootNode);
    uut.setSelectedNode(rootNode);
    uut.selection();

    if(reference.getSelectedNode() == uut.getSelectedNode())
    {
        std::cout << __PRETTY_FUNCTION__ << " PASSED" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " FAILED" << std::endl;
    }
}

void expansionTest()
{
    Player::MonteCarloPlayer reference;
    Player::MonteCarloPlayerMT uut;

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    referenceNode->boardState.initBoard();
    referenceNode->playerNum = Player::PLAYER_NUMBER_1;
    referenceNode->simulated = true;

    std::shared_ptr<MonteCarlo::TreeNode> uutNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutNode->boardState.initBoard();
    uutNode->playerNum = Player::PLAYER_NUMBER_1;
    uutNode->simulated = true;

    reference.setSelectedNode(referenceNode);
    reference.expansion();

    uut.setSelectedNode(uutNode);
    uut.expansion();

    if(referenceNode->childNodes.size() != uutNode->childNodes.size())
    {
        std::cout << __PRETTY_FUNCTION__ << " FAILED" << std::endl;
    }
    else
    {
        bool pass = true;
        for(int i = 0; i < referenceNode->childNodes.size(); ++i)
        {
            if(!MonteCarlo::nodeCompare(referenceNode->childNodes[i], uutNode->childNodes[i]))
            {
                pass = false;
                break;
            }
        }

        if(pass)
        {
            std::cout << __PRETTY_FUNCTION__ << " PASSED" << std::endl;
        }
        else
        {
            std::cout << __PRETTY_FUNCTION__ << " FAILED" << std::endl;
        }
    }
}

void simulationTest()
{
    Player::MonteCarloPlayer reference;
    Player::MonteCarloPlayerMT uut;

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    Game::GameBoard gameBoard;
    gameBoard.scramble();
    referenceNode->boardState = gameBoard;
    referenceNode->playerNum = Player::PLAYER_NUMBER_1;

    std::shared_ptr<MonteCarlo::TreeNode> uutNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutNode->boardState = gameBoard;
    uutNode->playerNum = Player::PLAYER_NUMBER_1;

    reference.setRootNode(referenceNode);
    uut.setRootNode(uutNode);

    reference.setSelectedNode(referenceNode);
    uut.setSelectedNode(uutNode);

    reference.setDeterministic(true, 0);
    uut.setDeterministic(true, 0);

    reference.simulation();
    uut.simulation();

    if(referenceNode->numWins == uutNode->numWins) 
    {
        std::cout << __PRETTY_FUNCTION__ << " PASSED" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " FAILED" << std::endl;
    }
}

void backpropagationTest()
{
    std::cout << "ENTERING BACKPROP" << std::endl;
    std::uniform_real_distribution<double> dist1(0, 1);
    std::uniform_real_distribution<double> dist2(0, 50);

    std::default_random_engine generator;

    srand(time(NULL));

    Player::MonteCarloPlayer reference;
    Player::MonteCarloPlayerMT uut;

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> uutNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());

    double numWins = dist2(generator);
    int numTimesVisited = rand() % 101;

    referenceNode->numWins = numWins;
    referenceNode->numTimesVisited = numTimesVisited;

    uutNode->numWins = numWins;
    uutNode->numTimesVisited = numTimesVisited;

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode2 = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> uutNode2 = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());

    numWins = dist1(generator);
    numTimesVisited = 1;

    referenceNode2->numWins = numWins;
    referenceNode2->numTimesVisited = numTimesVisited;

    uutNode2->numWins = numWins;
    uutNode2->numTimesVisited = numTimesVisited;

    referenceNode2->parentNode = referenceNode;
    uutNode2->parentNode = uutNode;

    reference.setRootNode(referenceNode);
    reference.setSelectedNode(referenceNode2);
    reference.setExplorationParam(1);

    uut.setRootNode(uutNode);
    uut.setSelectedNode(uutNode2);
    uut.setExplorationParam(1);

    reference.backpropagation();
    uut.backpropagation();

    double referenceSum = (referenceNode->value * referenceNode->value) + (referenceNode2->value * referenceNode2->value);
    double uutSum = (uutNode->value * uutNode->value) + (uutNode2->value * uutNode2->value);

    double relErr = referenceSum - uutSum;

    std::cout << referenceNode->value << std::endl;
    std::cout << uutNode->value << std::endl;

    if(relErr < 0.01)
    {
        std::cout << __PRETTY_FUNCTION__ << " PASSED" << std::endl;
    }
    else
    {
        std::cout << __PRETTY_FUNCTION__ << " FAILED" << std::endl;
    }
}

int main()
{
    selectionTest();
    expansionTest();
    simulationTest();
    backpropagationTest();
    return 0;
}