SVGs = $(wildcard *.svg) 
PNGs := $(patsubst %.svg, %.png, $(SVGs))

INKSCAPE= /Applications/Inkscape.app/Contents/MacOS/inkscape

all: $(PNGs)

%.png:%.svg
	$(INKSCAPE) $< -o $@ 
	rm -f ../../build/qrc_neutrino.cpp

clean:
	rm -f $(PNGs) ../../build/qrc_neutrino.cpp
	