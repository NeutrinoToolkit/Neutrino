SVGs = $(wildcard *.svg) 
PNGs := $(patsubst %.svg, %.png, $(SVGs))
# dark mode icons (selected at startup by nApp): generated from the light ones by darkify.py
DARK_SVGs := $(patsubst %.svg, dark/%.svg, $(SVGs))
DARK_PNGs := $(patsubst %.svg, dark/%.png, $(SVGs))

INKSCAPE= /Applications/Inkscape.app/Contents/MacOS/inkscape

# all: $(PNGs)
all: $(PNGs) $(DARK_PNGs)

%.png:%.svg
	$(INKSCAPE) $< -o $@ 
	rm -f ../../build/qrc_neutrino.cpp

dark/%.svg: %.svg darkify.py
	@mkdir -p dark
	python3 darkify.py $< $@

dark/%.png: dark/%.svg
	$(INKSCAPE) $< -o $@

# keep the generated dark svg (make would delete them as intermediate files)
.SECONDARY: $(DARK_SVGs)

clean:
	rm -f $(PNGs) ../../build/qrc_neutrino.cpp
	
clean-dark:
	rm -rf dark
