#!/bin/bash -x
if [ $# -eq 0 ]
  then
    echo "No png file supplied"
    exit 1
fi

filein=$1
justfile=$(basename $1)
dirout=${justfile%.*}

rm -rf ${dirout}.iconset ${dirout}.icns
	
mkdir -p ${dirout}.iconset

convert -resize 16x16     ${filein} ${dirout}.iconset/icon_16x16.png
convert -resize 32x32     ${filein} ${dirout}.iconset/icon_16x16@2x.png
convert -resize 32x32     ${filein} ${dirout}.iconset/icon_32x32.png
convert -resize 64x64     ${filein} ${dirout}.iconset/icon_32x32@2x.png
convert -resize 64x64     ${filein} ${dirout}.iconset/icon_64x64.png
convert -resize 128x128   ${filein} ${dirout}.iconset/icon_64x64@2x.png
convert -resize 128x128   ${filein} ${dirout}.iconset/icon_128x128.png
convert -resize 256x256   ${filein} ${dirout}.iconset/icon_128x128@2x.png
convert -resize 256x256   ${filein} ${dirout}.iconset/icon_256x256.png
convert -resize 512x512   ${filein} ${dirout}.iconset/icon_256x256@2x.png
convert -resize 512x512   ${filein} ${dirout}.iconset/icon_512x512.png
convert -resize 1024x1024 ${filein} ${dirout}.iconset/icon_512x512@2x.png
convert -resize 1024x1024 ${filein} ${dirout}.iconset/icon_1024x1024.png
convert -resize 2048x2048 ${filein} ${dirout}.iconset/icon_1024x1024@2x.png
convert -resize 2048x2048 ${filein} ${dirout}.iconset/icon_2048x2048.png

iconutil -c icns ${dirout}.iconset