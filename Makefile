
ifeq ($(OS),Windows_NT)
    UNAME_S := Windows_NT
else
    UNAME_S := $(shell uname -s)
endif

all: $(UNAME_S)

version_tag:=$(shell git describe --abbrev=0 --tags)
version_number:=$(shell git rev-list master ${version_tag}^..HEAD --count)

VERSION:=${version_tag}-${version_number}

colormap:
	cd resources/colormaps && /usr/local/opt/qt5/bin/qmake -spec macx-g++-5 && make && ./colormaps

debug::
	mkdir -p $@
	cd $@ && cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-5 -DQt5_DIR=/usr/local/opt/qt5/lib/cmake/Qt5 ..
	$(MAKE) -C $@ $(MAKECMDGOALS)
	@echo "Debug app is in $@"

Darwin::
	rm -rf $@ 
	mkdir -p $@
	# debug cmake command line
	# cd $@ && cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-5 -DQt5_DIR=/usr/local/opt/qt5/lib/cmake/Qt5 ..
	cd $@ && cmake -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-5 -DQt5_DIR=/usr/local/opt/qt5/lib/cmake/Qt5 ..
	$(MAKE) -C $@ $(MAKECMDGOALS)
	cp -r $@/Neutrino.app .
	/usr/local/opt/qt5/bin/macdeployqt Neutrino.app
	/usr/libexec/PlistBuddy -c "Set CFBundleShortVersionString ${VERSION}" Neutrino.app/Contents/Info.plist
ifneq ("$(findstring +,$VERSION)","+")
	/usr/libexec/PlistBuddy -c "Set CFBundleVersion ${VERSION}" Neutrino.app/Contents/Info.plist
endif
	rm -rf Neutrino.dmg rw.Neutrino.dmg dmg
	diskutil eject /Volumes/Neutrino 2> /dev/null
	mkdir dmg
	cp -r Neutrino.app dmg
	./resources/macPackage/createdmg.sh --icon-size 96 --volname Neutrino --volicon resources/icons/icon.icns --background resources/macPackage/sfondo.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino.dmg dmg && rm -rf dmg
	mv Neutrino.dmg Neutrino-${VERSION}-${@}.dmg

doc:
	cd doc ; (cat neutrino.dox; echo "PROJECT_NUMBER=${VERSION}") | doxygen -; cd latex; pdflatex refman.tex; pdflatex refman.tex


	 
