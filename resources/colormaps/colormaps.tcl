#!/bin/sh
# the next line restarts using wish \
exec wish "$0" "$@"

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
		set fid [open "definitions.txt" w]
		puts -nonewline $fid $data
		close $fid
	}
}

proc updateData {} {
	global name data i j k model
	set linen [lindex [split "[.editor index insert]" .] 0]
	set line [.editor get $linen.0 $linen.end]
	if {[llength [split $line \;]] == 2} {
		set name [string trim [lindex [split $line \;] 0]]
		set data [string trim [lindex [split $line \;] 1]]
		displayData
	}
}

set gnuplot 0

set i 0
set j 0 
set k 0
set model RGB

set data "rgbformulae $i, $j, $k model $model"

frame .data
label .data.lab -text "Data" -width 4
entry .data.ent -textvariable data
bind .data.ent <Return> {displayData}

button .data.show -text "Show" -width 4 -command displayData
pack .data.lab .data.show -side left
pack .data.ent -side right -fill x -expand y

frame .name
label .name.lab -text "Name" -width 4
entry .name.ent -textvariable name
button .name.add -text "Add" -width 4 -command add
pack .name.lab .name.add -side left
pack .name.ent -side right -fill x -expand y

text .editor -width 50 -height 10
pack .data .name -fill x
pack .editor -fill both -expand y
button .save -text "Save" -command save
pack .save -fill x

bind .editor <Double-Button-1> {updateData}

set fid [open "definitions.txt" r]
set dataFile [read $fid]
.editor insert end "$dataFile"
close $fid
global gnuplot
set gnuplot [ open "| gnuplot" w+ ]
puts $gnuplot "unset border; set pm3d map; unset colorbox; unset tics; set bmargin at screen 0;set lmargin at screen 0;set rmargin at screen 1;set tmargin at screen 1;"
puts $gnuplot "unset mouse;"

displayData

