#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino.dmg dmg_dir

mkdir dmg_dir

cp -r Neutrino.app dmg_dir

/usr/local/opt/qt5/bin/macdeployqt dmg_dir/Neutrino.app

rm -rf macdeployqtfix*

git clone https://github.com/iltommi/macdeployqtfix.git

python macdeployqtfix/macdeployqtfix.py dmg_dir/Neutrino.app/Contents/MacOS/Neutrino /usr/local
 
$DIR/createdmg.sh --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-MacOS.dmg dmg_dir

