
ifeq ($(OS),Windows_NT)
    UNAME_S := Windows_NT
else
    UNAME_S := $(shell uname -s)
endif

ifeq (,$(findstring debug,$(config)))
	CMAKEFLAGS += -DCMAKE_BUILD_TYPE=Debug
endif


all: $(UNAME_S)

version_tag:=$(shell git describe --abbrev=0 --tags)
version_number:=$(shell git rev-list master ${version_tag}^..HEAD --count)
version_branch:=$(shell git name-rev --name-only HEAD)

VERSION:=${version_tag}-${version_number}

ifneq ($(version_branch),master)
	VERSION:=$(version_branch)-$(VERSION)
endif

colormap:
	cd resources/colormaps && /usr/local/opt/qt5/bin/qmake -spec macx-g++-5 && make && ./colormaps


Darwin:: 
	mkdir -p $@
	cd $@ && cmake -DCMAKE_CXX_COMPILER=/usr/local/bin/clang-omp++ -DQt5_DIR=/usr/local/opt/qt5/lib/cmake/Qt5 ..
	$(MAKE) -C $@
	rm -rf Neutrino.app
	cp -r $@/Neutrino.app .
	/usr/local/opt/qt5/bin/macdeployqt Neutrino.app
	rm -rf macdeployqtfix
	git clone git@github.com:iltommi/macdeployqtfix.git
	python macdeployqtfix/macdeployqtfix.py Neutrino.app/Contents/MacOS/Neutrino /usr/local/Cellar/qt5/5.6.1/
	install_name_tool -change /usr/local/lib/gcc/6/libgcc_s.1.dylib \@executable_path/../Frameworks/libgcc_s.1.dylib Neutrino.app/Contents/Frameworks/libfftw3.3.dylib
	install_name_tool -change /usr/local/Cellar/fftw/3.3.4_1/lib/libfftw3.3.dylib \@executable_path/../Frameworks/libfftw3.3.dylib Neutrino.app/Contents/Frameworks/libfftw3_threads.3.dylib
	install_name_tool -change /usr/local/lib/gcc/6/libgcc_s.1.dylib \@executable_path/../Frameworks/libgcc_s.1.dylib Neutrino.app/Contents/Frameworks/libfftw3_threads.3.dylib
	install_name_tool -change /usr/local/Cellar/fftw/3.3.4_1/lib/libfftw3.3.dylib \@executable_path/../Frameworks/libfftw3.3.dylib Neutrino.app/Contents/Frameworks/libfftw3_omp.3.dylib
	install_name_tool -change /usr/local/lib/gcc/6/libgcc_s.1.dylib \@executable_path/../Frameworks/libgcc_s.1.dylib Neutrino.app/Contents/Frameworks/libfftw3_omp.3.dylib
	install_name_tool -change /usr/local/Cellar/hdf4/4.2.11/lib/libdf.4.2.11.dylib \@executable_path/../Frameworks/libdf.4.2.11.dylib Neutrino.app/Contents/Frameworks/libmfhdf.4.2.11.dylib
	install_name_tool -change /usr/local/lib/libjpeg.8.dylib \@executable_path/../Frameworks/libjpeg.8.dylib Neutrino.app/Contents/Frameworks/libmfhdf.4.2.11.dylib
	install_name_tool -change /usr/local/lib/libsz.2.dylib \@executable_path/../Frameworks/libsz.2.dylib Neutrino.app/Contents/Frameworks/libmfhdf.4.2.11.dylib
	install_name_tool -change /usr/local/Cellar/hdf5/1.8.16_1/lib/libhdf5.10.dylib \@executable_path/../Frameworks/libhdf5.10.dylib Neutrino.app/Contents/Frameworks/libhdf5_hl.10.dylib
	install_name_tool -change /usr/local/lib/libjpeg.8.dylib \@executable_path/../Frameworks/libjpeg.8.dylib Neutrino.app/Contents/Frameworks/libdf.4.2.11.dylib
	install_name_tool -change /usr/local/lib/libsz.2.dylib \@executable_path/../Frameworks/libsz.2.dylib Neutrino.app/Contents/Frameworks/libdf.4.2.11.dylib
	install_name_tool -change /usr/local/lib/gcc/6/libgcc_s.1.dylib \@executable_path/../Frameworks/libgcc_s.1.dylib Neutrino.app/Contents/Frameworks/libgomp.1.dylib
	/usr/libexec/PlistBuddy -c "Set CFBundleShortVersionString ${VERSION}" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add NSPrincipalClass string NSApplication" Neutrino.app/Contents/Info.plist
	/usr/libexec/PlistBuddy -c "Add NSHighResolutionCapable bool True" Neutrino.app/Contents/Info.plist
ifneq ("$(findstring +,$VERSION)","+")
	/usr/libexec/PlistBuddy -c "Set CFBundleVersion ${VERSION}" Neutrino.app/Contents/Info.plist
endif
	-diskutil eject /Volumes/Neutrino 2> /dev/null
	rm -rf Neutrino.dmg rw.Neutrino.dmg dmg
	mkdir -p dmg
	cp -r Neutrino.app dmg
	./resources/macPackage/createdmg.sh --icon-size 96 --volname Neutrino --volicon resources/icons/icon.icns --background resources/macPackage/sfondo.png --window-size 420 400 --icon Neutrino.app 90 75 --app-drop-link 320 75 Neutrino.dmg dmg && rm -rf dmg
	mv Neutrino.dmg Neutrino-${VERSION}-${@}.dmg
	@echo "\nBuild $@ : Neutrino-${VERSION}-${@}.dmg"


cross::
	mkdir -p $@
	cd $@ && cmake .. -DCMAKE_TOOLCHAIN_FILE=../resources/cmake/Toolchain-i686-mingw32.cmake -DNEUTRINO_CROSS_ROOT=/home/neutrino/CROSS-SOURCES
	$(MAKE) -C $@
	$(MAKE) -C $@ package
	

Linux::
	mkdir -p $@	
	cd $@ && cmake ..
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

	zip -FSr Neutrino-${VERSION}-$(UNAME_S).zip Neutrino


.PHONY: doc smonta

doc:
	cd doc ; (cat neutrino.dox; echo "PROJECT_NUMBER=${VERSION}") | doxygen -; cd latex; pdflatex refman.tex; pdflatex refman.tex


	 
