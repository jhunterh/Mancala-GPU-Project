#include <iostream>
#include <memory>
#include <ctime>
#include <cstdlib>
#include <random>
#include <sstream>

#include "game.h"
#include "MonteCarloPlayer.h"
#include "MonteCarloPlayerMT.h"
#include "MonteCarloHybridPlayer.h"
#include "NaivePureMonteCarloPlayer.h"
#include "PureMonteCarloPlayer.h"
#include "MonteCarloTypes.h"
#include "GameBoard.h"
#include "Logger.h"
#include "MonteCarloUtility.h"

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
    Logging::Logger& logger = Logging::Logger::getInstance();

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

    std::stringstream out("");

    out << std::endl;

    if(reference.getSelectedNode() == uutMT.getSelectedNode())
    {
        out << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
    }
    else
    {
        out << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
    }

    if(reference.getSelectedNode() == uutGPU.getSelectedNode())
    {
        out << __PRETTY_FUNCTION__ << " Hybrid PASSED" << std::endl;
    }
    else
    {
        out << __PRETTY_FUNCTION__ << " Hybrid FAILED" << std::endl;
    }

    logger.log(Logging::TEST_LOG, out.str());
}

void expansionTest()
{
    Logging::Logger& logger = Logging::Logger::getInstance();

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

    std::stringstream out("");

    out << std::endl;

    if(referenceNode->childNodes.size() != uutMTNode->childNodes.size())
    {
        out << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
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
            out << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
        }
        else
        {
            out << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
        }
    }

    if(referenceNode->childNodes.size() != uutGPUNode->childNodes.size())
    {
        out << __PRETTY_FUNCTION__ << " Hybrid FAILED" << std::endl;
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
            out << __PRETTY_FUNCTION__ << " Hybrid PASSED" << std::endl;
        }
        else
        {
            out << __PRETTY_FUNCTION__ << " Hybrid FAILED" << std::endl;
        }
    }

    logger.log(Logging::TEST_LOG, out.str());
}

void simulationTest()
{
    Logging::Logger& logger = Logging::Logger::getInstance();

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

    std::stringstream out("");

    out << std::endl;

    if(referenceNode->numWins == uutMTNode->numWins) 
    {
        out << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
    }
    else
    {
        out << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
    }

    if(referenceNode->numWins == uutGPUNode->numWins) 
    {
        out << __PRETTY_FUNCTION__ << " Hybrid PASSED" << std::endl;
    }
    else
    {
        out << __PRETTY_FUNCTION__ << " Hybrid FAILED" << std::endl;
    }

    logger.log(Logging::TEST_LOG, out.str());
}

void backpropagationTest()
{
    Logging::Logger& logger = Logging::Logger::getInstance();

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

    double referenceNormSquared = referenceNode->value * referenceNode->value;
    double diffMTNormSquared = (referenceNode->value - uutMTNode->value)*(referenceNode->value - uutMTNode->value) 
                                + (referenceNode2->value - uutMTNode2->value)*(referenceNode2->value - uutMTNode2->value);
    double diffGPUNormSquared = (referenceNode->value - uutGPUNode->value)*(referenceNode->value - uutGPUNode->value) 
                                + (referenceNode2->value - uutGPUNode2->value)*(referenceNode2->value - uutGPUNode2->value);

    double relErrMT = sqrt(diffMTNormSquared / referenceNormSquared);
    double relErrGPU = sqrt(diffGPUNormSquared / referenceNormSquared);

    std::stringstream out("");

    out << std::endl;

    if(relErrMT < 0.01)
    {
        out << __PRETTY_FUNCTION__ << " MultiThreaded PASSED" << std::endl;
    }
    else
    {
        out << __PRETTY_FUNCTION__ << " MultiThreaded FAILED" << std::endl;
    }

    if(relErrGPU < 0.01)
    {
        out << __PRETTY_FUNCTION__ << " Hybrid PASSED" << std::endl;
    }
    else
    {
        out << __PRETTY_FUNCTION__ << " Hybrid FAILED" << std::endl;
    }
    out << std::endl;

    logger.log(Logging::TEST_LOG, out.str());
}

void pureMonteCarloTest()
{
    Logging::Logger& logger = Logging::Logger::getInstance();

    Player::NaivePureMonteCarloPlayer reference;
    Player::PureMonteCarloPlayer uut;

    reference.setDeterministic(true, 0);
    uut.setDeterministic(true, 0);

    std::shared_ptr<MonteCarlo::TreeNode> referenceNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    std::shared_ptr<MonteCarlo::TreeNode> childNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    Game::GameBoard gameBoard;
    gameBoard.scramble();
    childNode->boardState = gameBoard;
    childNode->playerNum = Player::PLAYER_NUMBER_2;
    referenceNode->playerNum = Player::PLAYER_NUMBER_1;

    referenceNode->childNodes.push_back(childNode);

    std::shared_ptr<MonteCarlo::TreeNode> uutNode = std::shared_ptr<MonteCarlo::TreeNode>(new MonteCarlo::TreeNode());
    uutNode->playerNum = Player::PLAYER_NUMBER_1;

    uutNode->childNodes.push_back(childNode);

    reference.setRootNode(referenceNode);
    uut.setRootNode(uutNode);

    unsigned int simulationResults_ref = 0;
    unsigned int simulationNumMoves_ref = 0;
    unsigned int moveNum = 0;

    std::vector<unsigned int> simulationResults_uut(1,0);
    std::vector<unsigned int> simulationNumMoves_uut(1,0);

    reference.runSimulation(simulationResults_ref, simulationNumMoves_ref, moveNum);
    uut.simulateMove(moveNum, simulationResults_uut, simulationNumMoves_uut);

    bool pass = false;
    if(simulationResults_ref*PLAYCOUNT_THRESHOLD_GPU == simulationResults_uut[0])
    {
        pass = true;
    }

    std::stringstream out("");
    std::string statusString = (pass ? " PASSED" : " FAILED");
    out << __PRETTY_FUNCTION__ << statusString << std::endl;

    logger.log(Logging::TEST_LOG, out.str());
}

int main()
{
    selectionTest();
    expansionTest();
    simulationTest();
    backpropagationTest();
    pureMonteCarloTest();
    return 0;
}