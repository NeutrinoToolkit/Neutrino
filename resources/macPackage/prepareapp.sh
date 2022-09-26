#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino.dmg dmg_dir

mkdir dmg_dir

cp -r Neutrino.app dmg_dir

#ln -sf /usr/local/opt/python@3.9/Frameworks/Python.framework/ /usr/local/opt/python@3.9/lib/Python.framework

`brew --prefix qt@5`/bin/macdeployqt dmg_dir/Neutrino.app

python3 $DIR/macdeployqtfix.py dmg_dir/Neutrino.app/Contents/MacOS/Neutrino `brew --prefix`

cp `brew --prefix hdf4`/lib/libxdr.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks

if [[ `uname -m` == x86_64 ]]
then
  install_name_tool -change @rpath/libxdr.4.dylib @executable_path/../Frameworks/libxdr.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib
  install_name_tool -change @rpath/libdf.4.dylib @executable_path/../Frameworks/libdf.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib
else
  install_name_tool -change /opt/homebrew/Cellar/hdf4/4.2.15_4/lib/libxdr.4.dylib @executable_path/../Frameworks/libxdr.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib
  install_name_tool -change /opt/homebrew/Cellar/hdf4/4.2.15_4/lib/libdf.4.dylib  @executable_path/../Frameworks/libdf.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib
fi


# install_name_tool -change /usr/local/Cellar/hdf4/4.2.15_4/lib/libdf.4.dylib @executable_path/../Frameworks/libdf.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib
# install_name_tool -change /usr/local/Cellar/hdf4/4.2.15_4/lib/libxdr.4.dylib @executable_path/../Frameworks/libdf.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib

codesign --force --deep --sign - dmg_dir/Neutrino.app

$DIR/createdmg.sh --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-`uname -s`-`uname -m`.dmg dmg_dir
