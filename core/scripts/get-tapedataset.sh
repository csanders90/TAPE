#!/usr/bin/env bash
# LINK=https://drive.google.com/uc?export=download&id=1vb0aNUyM06ol_9ZzEKMJi8y5y-wc7Qi1
# If automatic downloading isn't working, the file can be downloaded manually with the above link.
ggID='1vb0aNUyM06ol_9ZzEKMJi8y5y-wc7Qi1'
ggURL='https://drive.google.com/uc?export=download'
TARGET=dataset.tar.gz
SHA256SUM=c58ab10792a8f350177a684afe67a817be9fdccaedba33f8502e884a663e1cf5

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
unzip dataset.tar.gz
mv tapedataset dataset
rm -rf dataset.tar.gz
