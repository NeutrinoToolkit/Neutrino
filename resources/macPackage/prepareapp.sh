#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino.dmg dmg_dir

mkdir dmg_dir

cp -r Neutrino.app dmg_dir

#ln -sf /usr/local/opt/python@3.9/Frameworks/Python.framework/ /usr/local/opt/python@3.9/lib/Python.framework

/usr/local/opt/qt@5/bin/macdeployqt dmg_dir/Neutrino.app

python3 $DIR/macdeployqtfix.py dmg_dir/Neutrino.app/Contents/MacOS/Neutrino /usr/local

cp /usr/local/lib/libxdr.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks
install_name_tool -change @rpath/libxdr.4.dylib @executable_path/../Frameworks/libxdr.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib
install_name_tool -change @rpath/libdf.4.dylib @executable_path/../Frameworks/libdf.4.dylib dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib

# install_name_tool -change /usr/local/lib/gcc/11/libgcc_s.1.dylib @executable_path/../Frameworks/libgcc_s.1.dylib dmg_dir/Neutrino.app/Contents/MacOS/Neutrino
# 
# for i in dmg_dir/Neutrino.app/Contents/Resources/plugins/*.dylib dmg_dir/Neutrino.app/Contents/Frameworks/*.dylib; do
# echo "$i"
# install_name_tool -change /usr/local/lib/gcc/11/libgcc_s.1.dylib @executable_path/../Frameworks/libgcc_s.1.dylib $i
# done
# 
# for i in dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib; do 
# install_name_tool -change /usr/local/Cellar/hdf4/4.2.15_4/lib/libxdr.4.dylib @executable_path/../Frameworks/libxdr.4.dylib $i
# install_name_tool -change /usr/local/Cellar/hdf4/4.2.15_4/lib/libdf.4.dylib @executable_path/../Frameworks/libxdr.4.dylib $i
# done
 
$DIR/createdmg.sh --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-MacOS.dmg dmg_dir


# dmg_dir/Neutrino.app/Contents/Frameworks/libmfhdf.4.dylib:
#         @executable_path/../Frameworks/libmfhdf.4.dylib (compatibility version 4.0.0, current version 4.15.2)
#         /usr/local/Cellar/hdf4/4.2.15_4/lib/libxdr.4.dylib (compatibility version 4.0.0, current version 4.15.2)
#         /usr/local/Cellar/hdf4/4.2.15_4/lib/libdf.4.dylib (compatibility version 4.0.0, current version 4.15.2)
#         /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1292.100.5)
