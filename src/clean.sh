# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
VENV_PATH=./venv

echo -e "${RED}!!! THIS WILL DEACTIVATE VIRTUAL ENVIROMENT AND DELETE VENV DIRECTORY !!!${NC}"
read -p "CONTINUE (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    deactivate
    rm -rf "$VENV_PATH"
else
    echo -e "${YELLOW}venv${NC} will not be deleted"
fi