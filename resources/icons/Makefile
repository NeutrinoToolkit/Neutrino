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

# plugin icons (with an svg): plugins/<name>/dark/<icon>.png, compiled in icons_dark.qrc of Neutrino
PLUGIN_PNGs := $(wildcard ../../plugins/*/*.png ../../plugins/*/*/*.png)
PLUGIN_ICONS := $(foreach p,$(PLUGIN_PNGs),$(if $(wildcard $(p:.png=.svg)),$(p)))
PLUGIN_DARK_PNGs := $(foreach p,$(PLUGIN_ICONS),$(dir $(p))dark/$(notdir $(p)))

define DARK_PLUGIN_RULE
$(dir $(1))dark/$(notdir $(1)): $(1:.png=.svg) darkify.py
	@mkdir -p $$(@D)
	python3 darkify.py $$< $$(@:.png=.svg)
	$(INKSCAPE) $$(@:.png=.svg) -o $$@
endef
$(foreach p,$(PLUGIN_ICONS),$(eval $(call DARK_PLUGIN_RULE,$(p))))

all: $(PLUGIN_DARK_PNGs)

dark-plugins: $(PLUGIN_DARK_PNGs)

clean:
	rm -f $(PNGs) ../../build/qrc_neutrino.cpp
	
clean-dark:
	rm -rf dark $(foreach p,$(PLUGIN_ICONS),$(dir $(p))dark)
