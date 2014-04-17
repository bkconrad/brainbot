#!/bin/bash
REMOTE_HOST=23.94.5.154
REMOTE_BRAINBOT_DIR=/home/kaen/brainbot/
REMOTE_RESOURCE_DIR=/home/kaen/bitfighter-hg/exe/

git push origin master

ssh_commands=`cat <<EOF
cd $REMOTE_BRAINBOT_DIR
git pull
cd $REMOTE_RESOURCE_DIR
pkill test
rm screenshots/*
nohup ./test --gtest_filter=*Botfight* > /dev/null < /dev/null &
disown %1
exit
EOF`

ssh $REMOTE_HOST "$ssh_commands"