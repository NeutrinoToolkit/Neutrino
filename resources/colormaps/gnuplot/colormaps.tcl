#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"


proc init {} {
	set fid [open "definitions.txt" r]
	set dataFile [read $fid]
	.editor insert end "$dataFile"
	close $fid
	global gnuplot
	set gnuplot [ open "| gnuplot" w+ ]
 	puts $gnuplot "unset border; set pm3d map; unset colorbox; unset tics; set bmargin at screen 0;set lmargin at screen 0;set rmargin at screen 1;set tmargin at screen 1;"
  	puts $gnuplot "unset mouse; set terminal x11 enhanced window \'[winfo id .gnuplot]\'"
}

proc displaySpin {} {
	global gnuplot data i j k model
	puts $gnuplot "set multiplot layout 5,1"
	set q 0
	set data "rgbformulae $i, $j, $k model $model"
	displayData
}

proc displayData {} {
	global gnuplot data i j k model
	puts $gnuplot "set palette $data; sp x"
	if {[scan $data "rgbformulae %d, %d, %d model %s" ai aj ak amodel] == 4 } {
		set i $ai
		set j $aj
		set k $ak
		set model $amodel
	}
	flush $gnuplot
}

proc add {} {
	global name data
	if {$name != ""} {
		.editor inser end "\n$name ; $data"	
	}
	.editor see end
}

proc save {} {
	set data [.editor get 1.0 end]
	if {$data!=""} {
		set fid [open "map-definitions.txt" w]
		puts -nonewline $fid $data
		close $fid
	}
}

proc updateSpin {} {
	global name data i j k model
	set linen [lindex [split "[.editor index insert]" .] 0]
	set line [.editor get $linen.0 $linen.end]
	if {[llength [split $line \;]] == 2} {
		set name [string trim [lindex [split $line \;] 0]]
		set data [string trim [lindex [split $line \;] 1]]
		if {[scan $data "rgbformulae %d, %d, %d model %s" ai aj ak amodel] == 4 } {
			set i $ai
			set j $aj
			set k $ak
			set model $amodel
			set data "rgbformulae $i, $j, $k model $model"
			displaySpin
		} else {
			displayData
		}
	}
}

set gnuplot 0

set i 0
set j 0 
set k 0
set model RGB

set data "rgbformulae $i, $j, $k model $model"

frame .spin
spinbox .spin.i -width 5 -from -36 -to 36 -textvariable i -command displaySpin
spinbox .spin.j -width 5 -from -36 -to 36 -textvariable j -command displaySpin
spinbox .spin.k -width 5 -from -36 -to 36 -textvariable k -command displaySpin
bind .spin.i <Key> {displaySpin}
bind .spin.j <Key> {displaySpin}
bind .spin.k <Key> {displaySpin}
pack .spin.i .spin.j .spin.k -side left -fill x -expand y
frame .radio

foreach mod "RGB HSV CMY YIQ XYZ" {
	radiobutton .radio.a$mod -variable model -text $mod -value $mod -command {displaySpin}
	pack .radio.a$mod -side left
}

frame .data
label .data.lab -text "Data" -width 4
entry .data.ent -textvariable data
button .data.show -text "Show" -width 4 -command displayData
pack .data.lab .data.ent .data.show -side left -fill x -expand y

frame .name
label .name.lab -text "Name" -width 4
entry .name.ent -textvariable name
button .name.add -text "Add" -width 4 -command add
pack .name.lab .name.ent .name.add -side left -fill x -expand y
frame .gnuplot -width 300 -height 300
pack .gnuplot -fill x
text .editor -width 50 -height 10
pack .spin .radio .data .name -fill x
pack .editor -fill both -expand y
frame .store
button .store.save -text "Save" -command save
button .store.make -text "Make" -command {save; eval exec make}
pack .store.save .store.make -side left -fill x -expand y
pack .store -fill x

bind .editor <Double-Button-1> {updateSpin}

init
displaySpin

