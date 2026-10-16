#! /bin/bash -x -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo $DIR
rm -rf Neutrino*.dmg Neutrino*.zip dmg_dir
mkdir dmg_dir
cp -r Neutrino.app dmg_dir

APP=dmg_dir/Neutrino.app
PACKAGE_NAME=Neutrino-$(uname -s)-$(uname -m)

# PyShell: ship the standard library of the python PythonQt was built with (pyshell_python.sh is written by cmake)
# PyShell::startup sets PYTHONHOME to Contents/Resources/python
EXTRA_EXECUTABLES=()
if [ -e ${APP}/Contents/Resources/plugins/libPyShell.dylib ] && [ -f pyshell_python.sh ]; then
    source ./pyshell_python.sh
    PY_DEST=${APP}/Contents/Resources/python/lib/python${PYSHELL_PYTHON_VERSION}
    mkdir -p ${PY_DEST}
    rsync -a --exclude __pycache__ --exclude 'test/' --exclude 'tests/' --exclude idlelib --exclude tkinter --exclude turtledemo --exclude ensurepip --exclude site-packages --exclude 'config-*' "${PYSHELL_PYTHON_STDLIB}/" ${PY_DEST}/
    # the compiled modules link brew libraries (openssl, xz, ...): let macdeployqt fix them too
    # for so in ${PY_DEST}/lib-dynload/*.so; do
    #     EXTRA_EXECUTABLES+=("-executable=${so}")
    # done
    # (only the few that need it: passing all of them makes macdeployqt very slow)
    for so in ${PY_DEST}/lib-dynload/*.so; do
        if otool -L ${so} | tail -n +2 | awk '{print $1}' | grep -qvE '^(/usr/lib|/System)'; then
            EXTRA_EXECUTABLES+=("-executable=${so}")
        fi
    done
    # libpython: macdeployqt looks for the brew Python.framework in the wrong place, so copy it here
    # and point every binary using it to the copy
    for bin in $(find ${APP} -type f \( -name '*.dylib' -o -perm +111 \)); do
        PY_LIB=$(otool -L ${bin} 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -E 'Python\.framework/Versions/[0-9.]+/Python$|libpython[0-9.]+\.dylib$' || true)
        if [ -n "${PY_LIB}" ]; then
            PY_LIB_NAME=libpython${PYSHELL_PYTHON_VERSION}.dylib
            if [ ! -e ${APP}/Contents/Frameworks/${PY_LIB_NAME} ]; then
                cp -L ${PY_LIB} ${APP}/Contents/Frameworks/${PY_LIB_NAME}
                chmod u+w ${APP}/Contents/Frameworks/${PY_LIB_NAME}
                install_name_tool -id @executable_path/../Frameworks/${PY_LIB_NAME} ${APP}/Contents/Frameworks/${PY_LIB_NAME}
            fi
            install_name_tool -change ${PY_LIB} @executable_path/../Frameworks/${PY_LIB_NAME} ${bin}
        fi
    done
fi

# $(brew --prefix qt)/bin/macdeployqt  dmg_dir/Neutrino.app -always-overwrite -libpath=$(brew --prefix)/lib -libpath=$(brew --prefix brotli)/lib -libpath=$(brew --prefix hdf4)/lib
$(brew --prefix qt)/bin/macdeployqt  ${APP} -always-overwrite -libpath=$(brew --prefix)/lib -libpath=$(brew --prefix brotli)/lib -libpath=$(brew --prefix hdf4)/lib "${EXTRA_EXECUTABLES[@]}"

# python modules shipped with PyShell (PYSHELL_PIP_PACKAGES, e.g. numpy scipy) with the same python:
# the wheels bring their own libraries (@loader_path), only the test suites are removed.
# Installed after macdeployqt, which does not need to see them (and gets very slow with them)
if [ -n "${PY_DEST}" ] && [ -n "${PYSHELL_PIP_PACKAGES}" ]; then
    ${PYSHELL_PYTHON_EXECUTABLE} -m pip install --only-binary=:all: --no-compile --disable-pip-version-check --target ${PY_DEST}/site-packages ${PYSHELL_PIP_PACKAGES}
    find ${PY_DEST}/site-packages -type d -name tests -prune -exec rm -rf {} +
    rm -rf ${PY_DEST}/site-packages/bin
fi

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
# CLT_PYTHON=/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework
# if [ -e ${APP}/Contents/Resources/plugins/libPyShell.dylib ]; then
#     if otool -L ${APP}/Contents/Resources/plugins/libPyShell.dylib ${APP}/Contents/Frameworks/libPythonQt*.dylib | grep -q '@rpath/Python3.framework'; then
#         if [ -d ${CLT_PYTHON} ]; then
#             cp -aH ${CLT_PYTHON} ${APP}/Contents/Frameworks
#             codesign --force --deep --sign - ${APP}/Contents/Frameworks/Python3.framework
#         else
#             echo "WARNING: PyShell needs Python3.framework but ${CLT_PYTHON} is missing"
#         fi
#     fi
# fi
# PythonQt is now built with the brew python: its library is deployed by macdeployqt and its standard library copied above

# codesign --deep does not sign the code in Contents/Resources (plugins, python modules modified by macdeployqt):
# an invalid signature there makes macOS kill the app when the library is loaded
find ${APP}/Contents/Resources -type f \( -name '*.dylib' -o -name '*.so' \) -exec codesign --force --sign - {} \;

# codesign --force --deep --sign - dmg_dir/Neutrino.app
codesign --force --deep --sign - ${APP}

# ditto -c -k --sequesterRsrc --keepParent dmg_dir/Neutrino.app Neutrino-$(uname -s)-$(uname -m).zip
ditto -c -k --sequesterRsrc --keepParent ${APP} ${PACKAGE_NAME}.zip

# create-dmg --icon-size 96 --volname Neutrino --volicon $DIR/dmg-icon.icns --background $DIR/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino-$(uname -s)-$(uname -m).dmg dmg_dir
ln -s /Applications dmg_dir/Applications
hdiutil create -volname Neutrino -srcfolder dmg_dir -fs HFS+ -format UDZO -ov ${PACKAGE_NAME}.dmg

