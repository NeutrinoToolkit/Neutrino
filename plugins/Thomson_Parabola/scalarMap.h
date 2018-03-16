/*
 * Reads scalar maps from file and returns for given point, interpolated values. Holds scalar
 * quantities. public tribox? Mah...
 */

#include <iostream>
#include <string>
#include <vector>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#ifndef __scalarMap_h
#define __scalarMap_h

#include "tribox.h"
#include "tridimvec.h"
#include "datafile.h"

typedef tridimvec<double> fp;

enum maptype {constant_map, interpolation_map};

class scalarMap : public tribox {
public:
	scalarMap()
		: tribox()
	{ }

	scalarMap(fp vertex1, fp vertex2)
		: tribox(vertex1, vertex2), myfield_versor(), myfield_center()
	{ }


	// costruzione di mappa scalare a partire da file, dimensione definita, 
	// versore che indica la direzione di interpolazione e centro di traslazione.
	// le unita' devono essere indicate rispetto al sistema della scatola
	// (altrimenti si impazzisce...)
	void setScalarMap(const char *fileName, int dimensions, fp map_direction, fp center, fp field_direction);
	fp getField(fp look_point);

	datafile *my_datafile;
	fp myfield_versor, myfield_center;	// dati campo
	fp mymap_direction;			// interpolation axis

	double **map;	// holds scalar map
	double **axis;	// holds map axis definitions
	int n_datapoints;
	int map_dimensions;
	gsl_interp_accel *my_interp_accel;
	gsl_spline *my_spline_alloc;


	// metodi di compatibilita' per versione a campo costante
	void assignScalarQuantity(double value, fp direction)
	{ fieldValue = value; myfield_versor = direction; my_maptype = constant_map; }

	double fieldValue;
	enum maptype my_maptype;
};



#endif
