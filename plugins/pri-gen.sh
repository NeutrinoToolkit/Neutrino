#!/bin/bash

bdir=`dirname $0`
cd $bdir


[ -d ui_ ] || mkdir ui_
rm -f ui_/*
cp ../src/ui_* ui_

cwd=$(pwd)

cat > neutrino-shared.pri << __EOF

INCLUDEPATH += $cwd/ui_
INCLUDEPATH += $(find ../nPhysImage -iname "*.h" -printf "%h\n" | sort | uniq | xargs -I __ss echo -n $cwd/__ss" ")
INCLUDEPATH += $(find ../src -iname "*.h" -printf "%h\n" | sort | uniq | xargs -I __ss echo -n $cwd/__ss" ")

__EOF


