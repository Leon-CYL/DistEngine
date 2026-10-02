#include "engine/version.h"

#include <iostream>
#include <string>

int main() {
    const std::string actual = distengine::version();

    if (actual != "0.1.0") {
        std::cerr << "Expected 0.1.0, got " << actual << '\n';
        return 1;
    }

    return 0;
}