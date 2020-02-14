#include "argument_parser.hpp"


ArgumentParser::ArgumentParser(int argc, char **argv) {
    if (argc == 1) {
        std::cout << "Please specify the input file" << std::endl;
        exit(0);
    }

    for (int i = 1; i < argc - 1; i = i + 2) {
        if (strcmp(argv[i], "--input") == 0) {
            this->inputFilePath = string(argv[i + 1]);
        }
    }
    
}
