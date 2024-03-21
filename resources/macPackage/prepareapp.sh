#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino*.dmg Neutrino*.zip dmg_dir
mkdir dmg_dir
cp -r Neutrino.app dmg_dir


$(brew --prefix qt)/bin/macdeployqt  dmg_dir/Neutrino.app -always-overwrite -libpath=$(brew --prefix)/lib -libpath=$(brew --prefix brotli)/lib -libpath=$(brew --prefix hdf4) 

#fix pythonqt
# cp -r $(brew --prefix python)/Frameworks/Python.framework dmg_dir/Neutrino.app/Contents/Frameworks
# for filename in libPythonQt-Qt6.3.dylib libPythonQt_QtAll-Qt6.3.dylib;
# do
# install_name_tool -change $(brew --prefix)/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Python @executable_path/../Frameworks/Python.framework/Python  dmg_dir/Neutrino.app/Contents/Frameworks/$filename
# done

# cp -r $(brew --prefix python)/Frameworks/Python.framework/Versions/Current/Python dmg_dir/Neutrino.app/Contents/Frameworks
# for filename in libPythonQt-Qt6.3.dylib libPythonQt_QtAll-Qt6.3.dylib;
# do
# install_name_tool -change $(brew --prefix)/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Python @executable_path/../Frameworks/Python  dmg_dir/Neutrino.app/Contents/Frameworks/$filename
# done

codesign --force --deep --sign - dmg_dir/Neutrino.app

ditto -c -k --sequesterRsrc --keepParent dmg_dir/Neutrino.app Neutrino-$(uname -s)-$(uname -m).zip

create-dmg --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-$(uname -s)-$(uname -m).dmg dmg_dir

