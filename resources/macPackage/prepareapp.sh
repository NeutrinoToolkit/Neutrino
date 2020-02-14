#! /bin/bash -x -e

rm -rf Neutrino.dmg dmg_dir

mkdir dmg_dir

cp -r Neutrino.app dmg_dir

/usr/local/opt/qt5/bin/macdeployqt dmg_dir/Neutrino.app

# python ../macdeployqtfix/macdeployqtfix.py dmg_dir/Neutrino.app/Contents/MacOS/Neutrino /usr/local 
# rm -rf dmg_dir/macdeployqtfix*

../resources/macPackage/createdmg.sh --icon-size 96 --volname Neutrino --volicon ../resources/macPackage/dmg-icon.icns --background ../resources/macPackage/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-MacOS.dmg dmg_dir

