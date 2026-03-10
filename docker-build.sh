#!/usr/bin/env bash

TAG=`date '+%Y-%m-%d-%H-%M-%S'`
if [ -z "${1+x}" ]; then
    echo "No version tag supplied, using timestamp: ${TAG}"
else
    TAG=$1
    echo "Using supplied version tag: ${TAG}"
fi

sudo docker build . -t ghcr.io/munger1985/kbot:${TAG}
sudo docker tag ghcr.io/munger1985/kbot:${TAG} ghcr.io/munger1985/kbot:latest

sudo docker push ghcr.io/munger1985/kbot:${TAG}
sudo docker push ghcr.io/munger1985/kbot:latest
