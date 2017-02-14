#!/bin/bash
mkdir -p dmg-icon.iconset
convert -resize 16x16     dmg-icon.png dmg-icon.iconset/icon_16x16.png
convert -resize 32x32     dmg-icon.png dmg-icon.iconset/icon_16x16@2x.png
convert -resize 32x32     dmg-icon.png dmg-icon.iconset/icon_32x32.png
convert -resize 64x64     dmg-icon.png dmg-icon.iconset/icon_32x32@2x.png
convert -resize 128x128   dmg-icon.png dmg-icon.iconset/icon_128x128.png
convert -resize 256x256   dmg-icon.png dmg-icon.iconset/icon_128x128@2x.png
convert -resize 256x256   dmg-icon.png dmg-icon.iconset/icon_256x256.png
convert -resize 512x512   dmg-icon.png dmg-icon.iconset/icon_256x256@2x.png
convert -resize 512x512   dmg-icon.png dmg-icon.iconset/icon_512x512.png
convert -resize 1024x1024 dmg-icon.png dmg-icon.iconset/icon_1024x1024.png
convert -resize 2048x2048 dmg-icon.png dmg-icon.iconset/icon_1024x1024@2x.png
iconutil -c icns dmg-icon.iconset