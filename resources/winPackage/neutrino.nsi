# define installer name
Name "Neutrino"

outFile "neutrino-installer.exe"
 
# set desktop as install directory
InstallDir $PROGRAMFILES\Neutrino
 
# default section start
section

CreateShortCut "$DESKTOP\neutrino.lnk" "$INSTDIR\Neutrino.exe" "" "$INSTDIR\Neutrino.ico"

CreateDirectory "$SMPROGRAMS\Neutrino"
CreateShortCut "$SMPROGRAMS\Neutrino\Neutrino.lnk" "$INSTDIR\Neutrino.exe" "" "$INSTDIR\Neutrino.ico"
CreateShortCut "$SMPROGRAMS\Neutrino\Uninstall.lnk" "$INSTDIR\neutrino-uninstaller.exe" "" 

# define output path
setOutPath $INSTDIR

SetAutoClose true 
# specify file to go in output path
File neutrino.exe
File neutrino.ico
File *.dll

setOutPath $INSTDIR\platforms
File platforms\*
#File nPython.ui
#File PyNeutrino.pyd
#File neutrino.py


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
rmdir /r "$SMPROGRAMS\Neutrino"
sectionEnd

SilentUnInstall silent
