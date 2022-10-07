#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino.dmg dmg_dir

mkdir dmg_dir

cp -r Neutrino.app dmg_dir

#ln -sf /usr/local/opt/python@3.9/Frameworks/Python.framework/ /usr/local/opt/python@3.9/lib/Python.framework

`brew --prefix qt`/bin/macdeployqt dmg_dir/Neutrino.app

python $DIR/macdeployqtfix.py dmg_dir/Neutrino.app/Contents/MacOS/Neutrino `brew --prefix qt`

cp `brew --prefix hdf4`/lib/libxdr.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks

install_name_tool -change `brew --prefix`/Cellar/hdf4/4.2.15_4/lib/libxdr.4.dylib @executable_path/../Frameworks/libxdr.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib
install_name_tool -change `brew --prefix`/Cellar/hdf4/4.2.15_4/lib/libdf.4.dylib @executable_path/../Frameworks/libdf.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib

codesign --force --deep --sign - dmg_dir/Neutrino.app

# ditto -c -k --sequesterRsrc --keepParent dmg_dir/Neutrino.app Neutrino-`uname -s`-`uname -m`.zip

$DIR/createdmg.sh --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-`uname -s`-`uname -m`.dmg dmg_dir
