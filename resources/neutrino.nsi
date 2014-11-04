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
#File nPython.ui
#File PyNeutrino.pyd
#File neutrino.py

SetFileAttributes "QtCore4.dll" HIDDEN
SetFileAttributes "QtGui4.dll" HIDDEN
SetFileAttributes "QtOpenGL4.dll" HIDDEN
SetFileAttributes "QtSvg4.dll" HIDDEN
SetFileAttributes "libcfitsio.dll" HIDDEN
SetFileAttributes "jpeg62.dll" HIDDEN
SetFileAttributes "libfftw3-3.dll" HIDDEN
SetFileAttributes "libgcc_s_dw2-1.dll" HIDDEN
SetFileAttributes "libgsl.dll" HIDDEN
SetFileAttributes "libgslcblas.dll" HIDDEN
SetFileAttributes "libnetpbm10.dll" HIDDEN
SetFileAttributes "libstdc++-6.dll" HIDDEN
SetFileAttributes "libtiff3.dll" HIDDEN
SetFileAttributes "libwinpthread-1.dll" HIDDEN
SetFileAttributes "qwt.dll" HIDDEN
SetFileAttributes "zlib1.dll" HIDDEN
SetFileAttributes "python27.dll" HIDDEN
SetFileAttributes "xdr.dll" HIDDEN
SetFileAttributes "hdf.dll" HIDDEN
SetFileAttributes "mfhdf.dll" HIDDEN

SetFileAttributes "libnPhysImageF.dll" HIDDEN
SetFileAttributes "Neutrino1.dll" HIDDEN


# define uninstaller name
writeUninstaller $INSTDIR\neutrino-uninstaller.exe

# default section end
sectionEnd
 
# create a section to define what the uninstaller does.
# the section will always be named "Uninstall"
section "Uninstall"
SetAutoClose true 

# Always delete uninstaller first
delete "$INSTDIR\neutrino-uninstaller.exe"
rmDir /r "$INSTDIR"
Delete "$DESKTOP\neutrino.lnk"
sectionEnd
