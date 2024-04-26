#!/usr/bin/env bash
# LINK=https://drive.google.com/uc?export=download&id=1y2IfbcoJmH3s0OwuQyJfCfTxHiO1xLU4 
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='1y2IfbcoJmH3s0OwuQyJfCfTxHiO1xLU4 '
ggURL='https://drive.google.com/uc?export=download'
TARGET=dataset.zip 
SHA256SUM=a5b4b525a0f832443f3dde55b39ca8a0004bce4a4a216fd9fb58b2a2a9326b30

cd "$(dirname ${BASH_SOURCE[0]})/.."

# # Automatic downloading script adopted from https://stackoverflow.com/a/38937732
# filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
# getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
# curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${TARGET}"

if ! command -v gdown &> /dev/null
then
    read -p "Prerequisite package gdown is not installed. Press any key to install (pip install --upgrade gdown)"
    pip install --upgrade gdown
fi
gdown -O "${TARGET}" ${ggID}

echo "$SHA256SUM  $TARGET" | sha256sum -c
test $? -eq 0 || read -p "Failed to verify SHA256 checksum. Press any key to continue anyway." -n 1 -r

# please run these commands in the terminal to unzip and clean the dataset
cd core
unzip dataset.zip 
rm -rf __MACOSX/
mv tapedataset dataset
rm -rf dataset.zip 
