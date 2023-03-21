#ifndef _LOGGER_H
#define _LOGGER_H

#include <fstream>
#include <map>
#include <memory>
#include <mutex>

namespace Logging 
{

static const std::string PERFORMANCE_LOG = "Performance_Log";
static const std::string SIMULATION_LOG = "Simulation_Log";
static const std::string TEST_LOG = "Test_Log";

// Definition of the Logger class
class Logger 
{
public:
    Logger(Logger const&) = delete;
    void operator=(Logger const&) = delete;

    // Logger Instance
    static Logger& getInstance() 
    {
        static Logger instance;
        return instance;
    }

    void log(std::string logName, std::string message);

private:
    Logger() = default;
    ~Logger();
    std::map<std::string, std::shared_ptr<std::ofstream>> m_outStreams;
    std::mutex m_loggingMutex;
};

}

#endif