cd /tmp
curl -Lk 'https://api2.cursor.sh/updates/download-latest?os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz

# Install VSCode Extensions
./cursor tunnel --accept-server-license-terms --install-extension ms-python.python   # Python extension
./cursor tunnel --accept-server-license-terms --install-extension ms-python.debugpy   # Python Debug Adapter
./cursor tunnel --accept-server-license-terms --install-extension ms-toolsai.jupyter  # Jupyter extension

# Authenticate and start the tunnel
./cursor tunnel user login --provider github 
./cursor tunnel --accept-server-license-terms --random-name &
