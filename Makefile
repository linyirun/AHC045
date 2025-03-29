CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall
CXXDEBUG = $(CXXFLAGS) -DDEBUG
TARGET = main
SRC = main.cpp

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

debug: $(SRC)
	$(CXX) $(CXXDEBUG) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)