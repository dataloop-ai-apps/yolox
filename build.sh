cd /tmp
curl -Lk 'https://api2.cursor.sh/updates/download-latest?os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
./cursor tunnel user login --provider github 
./cursor tunnel --accept-server-license-terms --random-name