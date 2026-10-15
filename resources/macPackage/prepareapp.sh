#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino*.dmg Neutrino*.zip dmg_dir
mkdir dmg_dir
cp -r Neutrino.app dmg_dir

APP=dmg_dir/Neutrino.app
PACKAGE_NAME=Neutrino-$(uname -s)-$(uname -m)

# $(brew --prefix qt)/bin/macdeployqt  dmg_dir/Neutrino.app -always-overwrite -libpath=$(brew --prefix)/lib -libpath=$(brew --prefix brotli)/lib -libpath=$(brew --prefix hdf4)/lib
$(brew --prefix qt)/bin/macdeployqt  ${APP} -always-overwrite -libpath=$(brew --prefix)/lib -libpath=$(brew --prefix brotli)/lib -libpath=$(brew --prefix hdf4)/lib

# remove Qt plugins whose @rpath dependencies were not deployed (e.g. qtvirtualkeyboard, qpdf)
for plugin in $(find ${APP}/Contents/PlugIns -name '*.dylib'); do
    for dep in $(otool -L ${plugin} | tail -n +2 | awk '{print $1}' | grep '^@rpath/' || true); do
        if [ ! -e "${APP}/Contents/Frameworks/${dep#@rpath/}" ]; then
            echo "removing ${plugin}: missing ${dep}"
            rm -f ${plugin}
            break
        fi
    done
done

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

# cp -aH /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework dmg_dir/Neutrino.app/Contents/Frameworks
# codesign --force --deep --sign - dmg_dir/Neutrino.app/Contents/Frameworks/Python3.framework
# Python3.framework (CommandLineTools) is needed only when PyShell is built and PythonQt links @rpath/Python3.framework
CLT_PYTHON=/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework
if [ -e ${APP}/Contents/Resources/plugins/libPyShell.dylib ]; then
    if otool -L ${APP}/Contents/Resources/plugins/libPyShell.dylib ${APP}/Contents/Frameworks/libPythonQt*.dylib | grep -q '@rpath/Python3.framework'; then
        if [ -d ${CLT_PYTHON} ]; then
            cp -aH ${CLT_PYTHON} ${APP}/Contents/Frameworks
            codesign --force --deep --sign - ${APP}/Contents/Frameworks/Python3.framework
        else
            echo "WARNING: PyShell needs Python3.framework but ${CLT_PYTHON} is missing"
        fi
    fi
fi

# codesign --force --deep --sign - dmg_dir/Neutrino.app
codesign --force --deep --sign - ${APP}

# ditto -c -k --sequesterRsrc --keepParent dmg_dir/Neutrino.app Neutrino-$(uname -s)-$(uname -m).zip
ditto -c -k --sequesterRsrc --keepParent ${APP} ${PACKAGE_NAME}.zip

# create-dmg --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-$(uname -s)-$(uname -m).dmg dmg_dir
ln -s /Applications dmg_dir/Applications
hdiutil create -volname Neutrino -srcfolder dmg_dir -fs HFS+ -format UDZO -ov ${PACKAGE_NAME}.dmg

