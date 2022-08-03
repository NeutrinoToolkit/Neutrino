#include "scalarMap.h"


fp
scalarMap::getField(fp look_point)
{
	// anche questo, in quanto a menosita'...
	
	if (isInside(look_point)) {
		if (my_maptype == constant_map) {
			return fieldValue*myfield_versor;
		} else if (my_maptype == interpolation_map) {

			// 1. look_point to local origin
			//fp LO_lookpoint = look_point-myvertex1; 
			fp LO_lookpoint = look_point-myfield_center;

			// 2. distance from field_center on interpolation direction
			double interpolation_distance = intp( LO_lookpoint, mymap_direction );
			double scalar_field = gsl_spline_eval(my_spline_alloc, interpolation_distance, my_interp_accel);

		//	std::cout<<"look point: "<<look_point<<std::endl;
		//	std::cout<<"LO_look_point: "<<LO_lookpoint<<std::endl;
		///	std::cout<<"map direction: "<<mymap_direction<<std::endl;
		//	std::cout<<"interpolation distance: "<<interpolation_distance<<std::endl;
		//	std::cout<<"scalar field: "<<scalar_field<<std::endl;
		//	std::cout<<std::endl;
		
			return scalar_field*myfield_versor;
		}
    }
    return fp(0.0,0.0,0.0);
}

// costruzione da vettore
//scalarMap::setScalarMap(const char *fileName, int dimensions, fp map_direction, fp center, fp field_direction)


// costruzione da file
void
scalarMap::setScalarMap(const char *fileName, int dimensions, fp map_direction, fp center, fp field_direction)
{
	// uffs, questo e' paccosissimo...
	// ATTENZIONE: il centro del campo e' in coordinate globali
	my_maptype = interpolation_map;
	if (dimensions == 2) {
		myfield_versor = field_direction.normVec();
		myfield_center = center;
		mymap_direction = map_direction.normVec();
		
		// 1. lettura file, due colonne, dimensional map e scalar value
		//datafile *my_datafile;
		my_datafile = new datafile(fileName, 2);
		my_datafile->readfile();


		// 2. scriviti i dati
		n_datapoints = my_datafile->nrows;
		map = new double * [2];
	} else {
		throw "Unimplemented!";
	}
		

//	for (int j=0; j<dimensions; j++) {
//		map[j] = new double [n_datapoints];
//		for (int i = 0; i<n_datapoints; i++) {
//			cout<<(void *)my_datafile->dataspace<<endl;
//			cout<<"--------------------------------------------n_datapoints="<<n_datapoints<<endl;
//			vector<double> *myvec;
//			myvec = (((vector<double> *)my_datafile->dataspace)+j);
//			map[j][i] = (*myvec)[i];
//		}
//	}


	for (int j=0; j<dimensions; j++)
		map[j] = (my_datafile->dataspace)[j];

	//for (int j=0; j<n_datapoints; j++) {
	//	cout<<map[0][j]<<"\t"<<map[1][j]<<endl;
	//}

	// set interpolation
	if (dimensions == 2) {
		my_interp_accel = gsl_interp_accel_alloc();
		my_spline_alloc = gsl_spline_alloc(gsl_interp_cspline, n_datapoints);
		gsl_spline_init(my_spline_alloc, map[0], map[1], n_datapoints);
	}
}
