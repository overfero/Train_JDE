#!/bin/bash

# Folder tujuan
DEST_DIR="/kaggle/working/Train_JDE"
cd $DEST_DIR

echo "Mulai mengunduh MOT17..."
wget -c https://motchallenge.net/data/MOT17.zip
echo "Mengekstrak MOT17..."
unzip -q MOT17.zip
echo "Menghapus file zip MOT17..."
rm MOT17.zip
echo "MOT17 Selesai!"

echo "Mulai mengunduh MOT20..."
wget -c https://motchallenge.net/data/MOT20.zip
echo "Mengekstrak MOT20..."
unzip -q MOT20.zip
echo "Menghapus file zip MOT20..."
rm MOT20.zip
echo "MOT20 Selesai!"

echo "Semua dataset telah berhasil diunduh dan diekstrak di $DEST_DIR"
