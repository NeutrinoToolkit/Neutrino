SVGs = $(wildcard *.svg) 
PNGs := $(patsubst %.svg, %.png, $(SVGs))

INKSCAPE= /Applications/Inkscape.app/Contents/Resources/bin/inkscape

all: $(PNGs)

%.png:%.svg
	$(INKSCAPE) -f $< -e $@ 
	rm -f ../../build/qrc_neutrino.cpp

clean:
	rm -f $(PNGs) ../../build/qrc_neutrino.cpp
	