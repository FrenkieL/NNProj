#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up CNN project environment...${NC}"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed${NC}"
    exit 1
fi

# Create venv in src folder
VENV_PATH="./venv"

if [ -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_PATH${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
    else
        echo -e "${YELLOW}Using existing virtual environment${NC}"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python -m venv "$VENV_PATH"
fi

# Activate venv
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
if [ -f "requirements.txt" ]; then
    echo -e "${GREEN}Installing requirements from requirements.txt...${NC}"
    pip install -r requirements.txt
else
    echo -e "${YELLOW}Warning: requirements.txt not found${NC}"
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To activate the environment, run:${NC}"
echo -e "  source src/venv/bin/activate"
echo -e "${YELLOW}To deactivate the environment, run:${NC}"
echo -e "  deactivate"