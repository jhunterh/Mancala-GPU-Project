#include <iostream>

#include "Logger.h"

namespace Logging 
{

void Logger::log(std::string logName, std::string message) 
{
    std::lock_guard<std::mutex> lg(m_loggingMutex);
    if(m_outStreams.find(logName) == m_outStreams.end())
    {
        std::cout << "Creating log: " << logName << std::endl;
        std::string logFileName = logName+".txt";
        std::shared_ptr<std::ofstream> logFile = std::shared_ptr<std::ofstream>(new std::ofstream());
        logFile->open(logFileName, std::ios::out);
        m_outStreams.insert(std::pair<std::string, std::shared_ptr<std::ofstream>>(logName, logFile));
    }

    (*m_outStreams[logName]) << message << std::endl;
}

Logger::~Logger()
{
    for(auto const& [key, val] : m_outStreams)
    {
        val->close();
    }
}

}