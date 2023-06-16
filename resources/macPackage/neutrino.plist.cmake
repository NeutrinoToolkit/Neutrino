<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>

    <key>CFBundleName</key>
    <string>${MACOSX_BUNDLE_BUNDLE_NAME}</string>

    <key>CFBundleIdentifier</key>
    <string>com.neutrino</string>
    <key>CFBundleExecutable</key>
    <string>${MACOSX_BUNDLE_EXECUTABLE_NAME}</string>


    <key>CFBundleGetInfoString</key>
    <string>${MACOSX_BUNDLE_INFO_STRING}</string>
    <key>CFBundleIdentifier</key>
    <string>${MACOSX_BUNDLE_GUI_IDENTIFIER}</string>

    <key>CFBundleLongVersionString</key>
    <string>${MACOSX_BUNDLE_LONG_VERSION_STRING}</string>

    <key>CFBundleShortVersionString</key>
    <string>${MACOSX_BUNDLE_SHORT_VERSION_STRING}</string>
    <key>CFBundleVersion</key>
    <string>${MACOSX_BUNDLE_BUNDLE_VERSION}</string>

    <key>LSMinimumSystemVersion</key>
    <string>${CMAKE_OSX_DEPLOYMENT_TARGET}</string>

    <key>NSHumanReadableCopyright</key>
    <string>${MACOSX_BUNDLE_COPYRIGHT}</string>

    <key>CFBundleIconFile</key>
    <string>icon.icns</string>

    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>

    <key>NSPrincipalClass</key>
    <string>NSApplication</string>

    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>

    <key>NSCameraUsageDescription</key>
    <string>to grab images</string>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>Neutrino session</string>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>sif</string>
                <string>neus</string>
                <string>neu</string>
                <string>fits</string>
                <string>tiff</string>
                <string>tif</string>
                <string>hdf</string>
                <string>img</string>
            </array>
            <key>CFBundleTypeIconFile</key>
            <string>filetype.icns</string>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
            <key>LSHandlerRank</key>
            <string>Default</string>
        </dict>
    </array>

</dict>
</plist>
