#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino.dmg dmg_dir

mkdir dmg_dir

cp -r Neutrino.app dmg_dir

ln -sf /usr/local/opt/python@3.9/Frameworks/Python.framework/ /usr/local/opt/python@3.9/lib/Python.framework

/usr/local/opt/qt5/bin/macdeployqt dmg_dir/Neutrino.app

python3 $DIR/macdeployqtfix.py dmg_dir/Neutrino.app/Contents/MacOS/Neutrino /usr/local

install_name_tool -change /usr/local/lib/gcc/11/libgcc_s.1.dylib @executable_path/../Frameworks/libgcc_s.1.dylib dmg_dir/Neutrino.app/Contents/MacOS/Neutrino
for i in dmg_dir/Neutrino.app/Contents/Resources/plugins/lib*; do install_name_tool -change /usr/local/lib/gcc/11/libgcc_s.1.dylib @executable_path/../Frameworks/libgcc_s.1.dylib $i; done

 
$DIR/createdmg.sh --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-MacOS.dmg dmg_dir

