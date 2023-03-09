#include <iostream>
#include <memory>
#include <ctime>
#include <cstdlib>

#include "game.h"
#include "MonteCarloPlayer.h"
#include "MonteCarloPlayerMT.h"
#include "MonteCarloTypes.h"

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
    referenceNode->playerNum = 0;
    referenceNode->simulated = true;

    std::shared_ptr<MonteCarlo::TreeNode> uutNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutNode->boardState.initBoard();
    uutNode->playerNum = 0;
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

}

void backpropagationTest()
{

}

int main()
{
    selectionTest();
    expansionTest();
    simulationTest();
    backpropagationTest();
    return 0;
}