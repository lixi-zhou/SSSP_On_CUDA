#ifndef ARGUMENT_PARSER_HPP
#define ARGUMENT_PARSER_HPP

#include "global.hpp"



class ArgumentParser {
    private:

    public:
        string inputFilePath;
        bool runOnCPU;
        int sourceNode;
        bool hasSourceNode;


    ArgumentParser(int argc, char **argv);


};




#endif