
ifeq ($(OS),Windows_NT)
    UNAME_S := Windows_NT
else
    UNAME_S := $(shell uname -s)
endif

ifeq (,$(findstring debug,$(config)))
	CMAKEFLAGS += -DCMAKE_BUILD_TYPE=Debug
endif

all: $(UNAME_S)

colormaps:
	cd resources/colormaps && /usr/local/opt/qt5/bin/qmake -spec macx-g++-5 && make && ./colormaps

dmg::
	rm -rf Neutrino.app
	cp -r Darwin/Neutrino.app .
	/usr/local/opt/qt5/bin/macdeployqt Neutrino.app
	rm -rf macdeployqtfix
	git clone https://github.com/iltommi/macdeployqtfix.git
	python macdeployqtfix/macdeployqtfix.py Neutrino.app/Contents/MacOS/Neutrino /usr/local
	/usr/libexec/PlistBuddy -c "Add NSPrincipalClass string NSApplication" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add NSHighResolutionCapable bool True" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes array" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0 dict" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeName string Neutrino session" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeRole string Viewer" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeIconFile string icon.icns" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions array" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:0 string neus" Neutrino.app/Contents/Info.plist
	-diskutil eject /Volumes/Neutrino 2> /dev/null
	rm -rf Neutrino.dmg rw.Neutrino.dmg dmg
	mkdir -p dmg
	cp -r Neutrino.app dmg
	./resources/macPackage/createdmg.sh --icon-size 96 --volname Neutrino --volicon resources/icons/icon.icns --background resources/macPackage/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino.dmg dmg && rm -rf dmg

Darwin:: 
	cmake -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-6 -DQt5_DIR=/usr/local/opt/qt5/lib/cmake/Qt5 -B$@ -H.
	$(MAKE) -C $@
	rm -rf Neutrino.app
	cp -r $@/Neutrino.app .
	/usr/local/opt/qt5/bin/macdeployqt Neutrino.app
	rm -rf macdeployqtfix
	git clone https://github.com/iltommi/macdeployqtfix.git
	python macdeployqtfix/macdeployqtfix.py Neutrino.app/Contents/MacOS/Neutrino /usr/local
	/usr/libexec/PlistBuddy -c "Add NSPrincipalClass string NSApplication" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add NSHighResolutionCapable bool True" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes array" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0 dict" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeName string Neutrino session" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeRole string Viewer" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeIconFile string icon.icns" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions array" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:0 string neus" Neutrino.app/Contents/Info.plist
	
	-diskutil eject /Volumes/Neutrino 2> /dev/null
	rm -rf Neutrino.dmg rw.Neutrino.dmg dmg
	mkdir -p dmg
	cp -r Neutrino.app dmg
	./resources/macPackage/createdmg.sh --icon-size 96 --volname Neutrino --volicon resources/icons/icon.icns --background resources/macPackage/background.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino.dmg dmg && rm -rf dmg

cross::
	mkdir -p $@
	cd $@ && cmake .. -DCMAKE_TOOLCHAIN_FILE=../resources/cmake/Toolchain-i686-mingw32.cmake -DNEUTRINO_CROSS_ROOT=/home/neutrino/CROSS-SOURCES
	$(MAKE) -C $@
	$(MAKE) -C $@ package	

Linux::
	cmake -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-6 -DQt5_DIR=/usr/local/opt/qt5/lib/cmake/Qt5 -B$@ -H.
	$(MAKE) -C $@ package


appdir:: Linux
	mkdir -p Neutrino.AppDir/usr

	cp -r Linux/lib Linux/bin Neutrino.AppDir/usr
	cp resources/icons/icon.png resources/linuxPackage/*  Neutrino.AppDir

	cp -r /usr/lib/x86_64-linux-gnu/qt5/plugins Neutrino.AppDir/usr/bin

	# apt-get install pax-utils
	lddtree Neutrino.AppDir/usr/bin/Neutrino | grep "=>" | awk '{print $$3}' | xargs cp -t Neutrino.AppDir/usr/lib/

	rm -rf Neutrino
	~/AppImageKit/AppImageAssistant Neutrino.AppDir Neutrino

	zip -FSr Neutrino.zip Neutrino


.PHONY: doc 

doc:
	cd doc ; cat neutrino.dox| doxygen -; cd latex; pdflatex refman.tex; pdflatex refman.tex


	 
