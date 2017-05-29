#! /bin/bash

rm -rf package_App

mkdir package_App

cp -r $1/Neutrino.app package_App

cd package_App

/usr/local/opt/qt5/bin/macdeployqt Neutrino.app
git clone https://github.com/iltommi/macdeployqtfix.git
python macdeployqtfix/macdeployqtfix.py Neutrino.app/Contents/MacOS/Neutrino /usr/local
/usr/libexec/PlistBuddy -c "Add NSHighResolutionCapable bool True" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes array" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0 dict" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeName string Neutrino session" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeRole string Viewer" Neutrino.app/Contents/Info.plist
cp ../../resources/macPackage/filetype.icns Neutrino.app/Contents/Resources
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeIconFile string filetype.icns" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions array" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:0 string neus" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:1 string neu" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:2 string fits" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:3 string tiff" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:4 string hdf" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:5 string img" Neutrino.app/Contents/Info.plist
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:0 string sif" Neutrino.app/Contents/Info.plist
mkdir dmg
mv Neutrino.app dmg
../../resources/macPackage/createdmg.sh --icon-size 96 --volname Neutrino --volicon ../../resources/macPackage/dmg-icon.icns --background ../../resources/macPackage/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino.dmg dmg

cp Neutrino.dmg ..
cd ..