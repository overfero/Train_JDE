#!/bin/bash
set -e

cd /kaggle/working/Train_JDE/CrowdHuman/

mkdir -p images/train images/val

echo "Extracting train01..."
if [ -f CrowdHuman_train01.zip ]; then
    unzip -q CrowdHuman_train01.zip -d images/train/
    rm CrowdHuman_train01.zip
    find images/train/Images/ -type f -exec mv -t images/train/ {} +
    rm -rf images/train/Images/
fi

echo "Extracting train02..."
if [ -f CrowdHuman_train02.zip ]; then
    unzip -q CrowdHuman_train02.zip -d images/train/
    rm CrowdHuman_train02.zip
    find images/train/Images/ -type f -exec mv -t images/train/ {} +
    rm -rf images/train/Images/
fi

echo "Extracting train03..."
if [ -f CrowdHuman_train03.zip ]; then
    unzip -q CrowdHuman_train03.zip -d images/train/
    rm CrowdHuman_train03.zip
    find images/train/Images/ -type f -exec mv -t images/train/ {} +
    rm -rf images/train/Images/
fi

echo "Extracting val..."
if [ -f CrowdHuman_val.zip ]; then
    unzip -q CrowdHuman_val.zip -d images/val/
    rm CrowdHuman_val.zip
    find images/val/Images/ -type f -exec mv -t images/val/ {} +
    rm -rf images/val/Images/
fi

echo "Extraction finished."
