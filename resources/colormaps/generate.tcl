#!/bin/sh
# the next line restarts using wish \
exec tclsh "$0" "$@"

set fin [open [file join [file dirname [info script]] "definitions.txt"] r]

seek $fin 0
set i 0
while {! [eof $fin]} {
	set linea [string trim [gets $fin]]
	if { [string index $linea 0] != "\#" && [string is alnum $linea] == 0} {
		set splitlinea [split $linea ";"]
		set name [string trim [lindex $splitlinea 0]]
		set cmap [string trim [lindex $splitlinea 1]]

	    set foutname [file join [file dirname [info script]] "cmaps" "$name"]
	    puts "$foutname $cmap"
	    set fout [open $foutname w ]
		incr i;
		regsub -all {"} $cmap {\"} cmap
 		set command [concat "gnuplot -e \"set palette" $cmap "; show palette palette 256\""]
  		set gnuplot [open "| $command 2>@stdout" w+]
		set linea [gets $gnuplot]
		while {! [eof $gnuplot]} {
			set linea [gets $gnuplot]
			if {[string is alnum $linea] == 0} {
				set splitlinea [lindex [split $linea "="] 3]
				set vals [scan $splitlinea "%3d %3d %3d" r g b]
				if {$vals == 3} {
					puts $fout "[format %03d $r] [format %03d $g] [format %03d $b]"
				}
			}
		}
	    close $fout
  	}
}
