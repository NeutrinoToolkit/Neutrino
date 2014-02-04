# define installer name
Name "Neutrino"

outFile "neutrino-installer.exe"
 
# set desktop as install directory
InstallDir $SMPROGRAMS\neutrino
 
# default section start
section

CreateShortCut "$DESKTOP\neutrino.lnk" "$INSTDIR\neutrino.exe" "" 

# define output path
setOutPath $INSTDIR

SetAutoClose true 
# specify file to go in output path
File neutrino.exe
File *.dll
File nPython.ui
File PyNeutrino.pyd
File neutrino.py
File /r plugins

SetFileAttributes "plugins" HIDDEN
SetFileAttributes "libfftw3-3.dll" HIDDEN
SetFileAttributes "libgcc_s_dw2-1.dll" HIDDEN
SetFileAttributes "libgsl.dll" HIDDEN
SetFileAttributes "libgslcblas.dll" HIDDEN
SetFileAttributes "libnetpbm10.dll" HIDDEN
SetFileAttributes "libnPhysImageF.dll" HIDDEN
SetFileAttributes "libnPhysUnwraps.dll" HIDDEN
SetFileAttributes "mingwm10.dll" HIDDEN
SetFileAttributes "QtCore4.dll" HIDDEN
SetFileAttributes "QtGui4.dll" HIDDEN
SetFileAttributes "QtSvg4.dll" HIDDEN
SetFileAttributes "QtXml4.dll" HIDDEN
SetFileAttributes "QtScript4.dll" HIDDEN
SetFileAttributes "qwt.dll" HIDDEN
SetFileAttributes "mfhdfdll.dll" HIDDEN
SetFileAttributes "hdfdll.dll" HIDDEN
SetFileAttributes "jpeg.dll" HIDDEN
SetFileAttributes "zlib.dll" HIDDEN
SetFileAttributes "zlib1.dll" HIDDEN
SetFileAttributes "xdrdll.dll" HIDDEN
SetFileAttributes "szip.dll" HIDDEN
SetFileAttributes "msvcp100.dll" HIDDEN
SetFileAttributes "msvcr100.dll" HIDDEN
SetFileAttributes "libtiff3.dll" HIDDEN
SetFileAttributes "jpeg62.dll" HIDDEN
SetFileAttributes "libcfitsio.dll" HIDDEN
SetFileAttributes "hdf5.dll" HIDDEN
SetFileAttributes "hdf5_hl.dll" HIDDEN
SetFileAttributes "libstdc++-6.dll" HIDDEN
SetFileAttributes "Neutrino.dll" HIDDEN
SetFileAttributes "PyNeutrino.pyd" HIDDEN
SetFileAttributes "nPython.ui" HIDDEN


# define uninstaller name
writeUninstaller $INSTDIR\neutrino-uninstaller.exe

# default section end
sectionEnd
 
# create a section to define what the uninstaller does.
# the section will always be named "Uninstall"
section "Uninstall"
 
# Always delete uninstaller first
delete "$INSTDIR\neutrino-uninstaller.exe"
rmDir /r "$INSTDIR"
Delete "$DESKTOP\neutrino.lnk"
sectionEnd
