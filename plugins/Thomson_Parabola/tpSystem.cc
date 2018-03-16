#include "tpSystem.h"


void
tpSystem::parseConfig(const char *conf_filename)
{
	// read/parse conf file
	Config *tpConfig;
	tpConfig = new Config();
	tpConfig->setAutoConvert(true);
	tpConfig->readFile(conf_filename);

	string read_string, efield_type, bfield_type;
	double Evalue, Bvalue;
	double xs[2], ys[2], zs[2], fsize[3];

	f3point Efield_vec, Bfield_vec;

//	tpConfig->lookupValue("dump-trajectory", traj_dump);

	// simulation box
	tpConfig->lookupValue("simulation-box.x-spawn-mm.init", xs[0]);
	tpConfig->lookupValue("simulation-box.x-spawn-mm.end", xs[1]);
	tpConfig->lookupValue("simulation-box.y-spawn-mm.init", ys[0]);
	tpConfig->lookupValue("simulation-box.y-spawn-mm.end", ys[1]);
	tpConfig->lookupValue("simulation-box.z-spawn-mm.init", zs[0]);
	tpConfig->lookupValue("simulation-box.z-spawn-mm.end", zs[1]);
	f3point sB_1(xs[0]*mm, ys[0]*mm, zs[0]*mm), sB_2(xs[1]*mm, ys[1]*mm, zs[1]*mm);	// sim boundaries
	sim_box->setBoundaries(sB_1, sB_2);


	// efield boundary field (got to change map version to include vertical dependence)
	tpConfig->lookupValue("thomson-parabola.efield.box-center-vec", read_string );
	tpConfig->lookupValue("thomson-parabola.efield.x-size-mm",fsize[0]);
	tpConfig->lookupValue("thomson-parabola.efield.y-size-mm",fsize[1]);
	tpConfig->lookupValue("thomson-parabola.efield.z-size-mm",fsize[2]);
	f3point Ef_1( mm*(f3point(read_string) - f3point(fsize[0], fsize[1], fsize[2])/2) );
	f3point Ef_2( mm*(f3point(read_string) + f3point(fsize[0], fsize[1], fsize[2])/2) );
	Efield->setBoundaries(Ef_1, Ef_2);

	// efield field
	tpConfig->lookupValue("thomson-parabola.efield.type",efield_type);
	if (efield_type == string("const")) {
		cerr<<"E field type: CONST"<<endl;
		tpConfig->lookupValue("thomson-parabola.efield.field-value", Evalue);
		tpConfig->lookupValue("thomson-parabola.efield.field-direction", read_string);
		Efield_vec = f3point(read_string);
		Efield_vec.norm();
		Efield->assignScalarQuantity(Evalue, Efield_vec);
	} else if (efield_type == string("scalarMap")) {
		cerr<<"E field type: SCALAR MAP"<<endl;
		f3point field_direction, field_center, map_direction;
		string ifilename;
		tpConfig->lookupValue("thomson-parabola.efield.mapfile", ifilename);
		tpConfig->lookupValue("thomson-parabola.efield.field-direction", read_string);
		field_direction = f3point(read_string);
		field_direction.norm();
		tpConfig->lookupValue("thomson-parabola.efield.field-center", read_string);
		field_center = f3point(read_string)*mm;
		tpConfig->lookupValue("thomson-parabola.efield.map-direction", read_string);
		map_direction = f3point(read_string);
		map_direction.norm();

		// TODO: cambiare numero dimensioni (2->1)
		Efield->setScalarMap(ifilename.c_str(), 2, map_direction, field_center, field_direction);

	}
	
	
	// bfield boundary and field
	tpConfig->lookupValue("thomson-parabola.bfield.box-center-vec", read_string );
	tpConfig->lookupValue("thomson-parabola.bfield.x-size-mm",fsize[0]);
	tpConfig->lookupValue("thomson-parabola.bfield.y-size-mm",fsize[1]);
	tpConfig->lookupValue("thomson-parabola.bfield.z-size-mm",fsize[2]);
	f3point Bf_1( mm*(f3point(read_string) - f3point(fsize[0], fsize[1], fsize[2])/2) );
	f3point Bf_2( mm*(f3point(read_string) + f3point(fsize[0], fsize[1], fsize[2])/2) );
	Bfield->setBoundaries(Bf_1, Bf_2);
	
	// bfield field
	tpConfig->lookupValue("thomson-parabola.bfield.type",bfield_type);
	if (bfield_type == string("const")) {
		cerr<<"B field type: CONST"<<endl;
		tpConfig->lookupValue("thomson-parabola.bfield.field-value", Bvalue);
		tpConfig->lookupValue("thomson-parabola.bfield.field-direction", read_string);
		Bfield_vec = f3point(read_string);
		Bfield_vec.norm();
		Bfield->assignScalarQuantity(Bvalue, Bfield_vec);
	} else if (bfield_type == string("scalarMap")) {
		cerr<<"B field type: SCALAR MAP"<<endl;
		f3point field_direction, field_center, map_direction;
		string ifilename;
		tpConfig->lookupValue("thomson-parabola.bfield.mapfile", ifilename);
		tpConfig->lookupValue("thomson-parabola.bfield.field-direction", read_string);
		field_direction = f3point(read_string);
		field_direction.norm();
		tpConfig->lookupValue("thomson-parabola.bfield.field-center", read_string);
		field_center = f3point(read_string)*mm;
		tpConfig->lookupValue("thomson-parabola.bfield.map-direction", read_string);
		map_direction = f3point(read_string);
		map_direction.norm();

		Bfield->setScalarMap(ifilename.c_str(), 2, map_direction, field_center, field_direction);

	}

	valid_system = true;
	valid_config = true;

	// mah...
	delete tpConfig;
}

void
tpSystem::writeConfig(const char *fname)
{
	Config cf;
	cf.setAutoConvert(true);

	Setting &root = cf.getRoot();
	//Setting &relative = root;

	// simulation box
	Setting &cf_sim = root.add("simulation-box", Setting::TypeGroup);

       	Setting& cf_sim_x = cf_sim.add("x-spawn-mm", Setting::TypeGroup);
	cf_sim_x.add("init", Setting::TypeFloat) = sim_box->myvertex1.x()/mm;
	cf_sim_x.add("end", Setting::TypeFloat) = sim_box->myvertex2.x()/mm;

       	Setting& cf_sim_y = cf_sim.add("y-spawn-mm", Setting::TypeGroup);
	cf_sim_y.add("init", Setting::TypeFloat) = sim_box->myvertex1.y()/mm;
	cf_sim_y.add("end", Setting::TypeFloat) = sim_box->myvertex2.y()/mm;

       	Setting& cf_sim_z = cf_sim.add("z-spawn-mm", Setting::TypeGroup);
	cf_sim_z.add("init", Setting::TypeFloat) = sim_box->myvertex1.z()/mm;
	cf_sim_z.add("end", Setting::TypeFloat) = sim_box->myvertex2.z()/mm;
	
	
	Setting &cf_tp = root.add("thomson-parabola", Setting::TypeGroup);
	Setting &cf_tp_B = cf_tp.add("bfield", Setting::TypeGroup);
	Setting &cf_tp_E = cf_tp.add("efield", Setting::TypeGroup);

	// efield
	cf_tp_E.add("type", Setting::TypeString) = std::string("const");
	cf_tp_E.add("box-center-vec", Setting::TypeString) = ((1/mm)*Efield->getCenter()).str();

	f3point si = Efield->getSize();
	cf_tp_E.add("x-size-mm", Setting::TypeFloat) = si.x()/mm;
	cf_tp_E.add("y-size-mm", Setting::TypeFloat) = si.y()/mm;
	cf_tp_E.add("z-size-mm", Setting::TypeFloat) = si.z()/mm;

	cf_tp_E.add("field-value", Setting::TypeFloat) = Efield->fieldValue;
	cf_tp_E.add("field-direction", Setting::TypeString) = Efield->myfield_versor.str();

	// bfield
	cf_tp_B.add("type", Setting::TypeString) = std::string("const");
	cf_tp_B.add("box-center-vec", Setting::TypeString) = ((1/mm)*Bfield->getCenter()).str();

	si = Bfield->getSize();
	cf_tp_B.add("x-size-mm", Setting::TypeFloat) = si.x()/mm;
	cf_tp_B.add("y-size-mm", Setting::TypeFloat) = si.y()/mm;
	cf_tp_B.add("z-size-mm", Setting::TypeFloat) = si.z()/mm;

	cf_tp_B.add("field-value", Setting::TypeFloat) = Bfield->fieldValue;
	cf_tp_B.add("field-direction", Setting::TypeString) = Bfield->myfield_versor.str();


	cf.writeFile(fname);

//
//	//# electric
//	lcread.setKey("thomson-parabola.efield.type=const","string")
//	lcread.setKey("thomson-parabola.efield.box-center-vec="+self.efield.box_center.str(),"vector")
//	lcread.setKey("thomson-parabola.efield.x-size-mm="+str(self.efield.box_widths.x),"number")
//	lcread.setKey("thomson-parabola.efield.y-size-mm="+str(self.efield.box_widths.y),"number")
//	lcread.setKey("thomson-parabola.efield.z-size-mm="+str(self.efield.box_widths.z),"number")
//	f_mod = self.efield.fieldVec.mod()
//	lcread.setKey("thomson-parabola.efield.field-value="+str(f_mod),"number")
//	lcread.setKey("thomson-parabola.efield.field-direction="+self.efield.fieldVec.scale(1./f_mod).str(),"vector")
//
//	//# magnetic
//	lcread.setKey("thomson-parabola.bfield.type=const","string")
//	lcread.setKey("thomson-parabola.bfield.box-center-vec="+self.bfield.box_center.str(),"vector")
//	lcread.setKey("thomson-parabola.bfield.x-size-mm="+str(self.bfield.box_widths.x),"number")
//	lcread.setKey("thomson-parabola.bfield.y-size-mm="+str(self.bfield.box_widths.y),"number")
//	lcread.setKey("thomson-parabola.bfield.z-size-mm="+str(self.bfield.box_widths.z),"number")
//	f_mod = self.bfield.fieldVec.mod()
//	lcread.setKey("thomson-parabola.bfield.field-value="+str(f_mod),"number")
//	lcread.setKey("thomson-parabola.bfield.field-direction="+self.bfield.fieldVec.scale(1./f_mod).str(),"vector")

}

void
tpSystem::getImpact(struct ionImpact *ion)
{
	
	// from this point everything should be MKSA
	
	my_ion = ion;	// questo deve venire dall'esterno

	double ion_energy = my_ion->energy;
	double ion_mass = my_ion->mass;
	double ion_charge = my_ion->charge;

	double ion_gamma = 1+ion_energy/(ion_mass*_cspeed*_cspeed);
	double ion_beta = sqrt(1-1/(ion_gamma*ion_gamma));
	my_ion->gamma = ion_gamma;
	
	double init_v;
	if (!relativistic) {
		init_v = sqrt(2*ion_energy/ion_mass);
	} else {	// relativistic calculation
		init_v = ion_beta*_cspeed;
	}
		
	double t_end = 3*(sim_box->myvertex1-sim_box->myvertex2).mod() / init_v;
	

	//cerr<<"\nStarting simulation:\n";
	//cerr<<"\tSimulation box: "<<*sim_box<<"\n";
	//cerr<<"\tE-field box: "<<*myTP->Efield<<" at field "<<myTP->Efield->scalarValue<<"\n";
	//cerr<<"\tB-field box: "<<*myTP->Bfield<<" at field "<<myTP->Bfield->scalarValue<<"\n";
	//cerr<<"energy[MeV]: "<<ion_energy/MeV<<", ion mass[uma]: "<<ion_mass/GSL_CONST_MKSA_UNIFIED_ATOMIC_MASS<<", ";
	//cerr<<"ionization state: +"<<ion_charge/GSL_CONST_MKSA_ELECTRON_CHARGE;
	//cerr<<"\tSimulation time is "<<t_end/ns<<"ns\n";
	//cerr<<endl;


	// ODE8 stepping
	const gsl_odeiv_step_type * T = gsl_odeiv_step_rk8pd;
     
	gsl_odeiv_step * s  = gsl_odeiv_step_alloc (T, 6);
	gsl_odeiv_control * c  = gsl_odeiv_control_y_new (1e-10, 0.0);
	gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (6);	// num dimensioni

	gsl_odeiv_system sys;
        if (!relativistic) {
		gsl_odeiv_system temp_sys = {lorentzStep, NULL, 6, (void *)this};
		memcpy((void*)&sys, (void*)&temp_sys, sizeof(gsl_odeiv_system));
	} else {
		gsl_odeiv_system temp_sys = {lorentzStepRelativistic, NULL, 6, (void *)this};
		memcpy((void*)&sys, (void*)&temp_sys, sizeof(gsl_odeiv_system));
	}

	double t = 0.0*ns, t1 = 100*ns;
	double h = 10*ps;
	double y[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, init_v };

	/* Problem here: h gets increasing adaptively. This is dangerous if small structures are
	 * present after a certain propagation distance.
	 * Solution (future): set dangerous points in the simulation and adaptively correct h accordingly
	 * Workaround (present): always correct h to ridiculous small value */

	while (t < t_end)
	{
		h = 100*ps;
		int status = gsl_odeiv_evolve_apply (e, c, s,
				&sys, 
				&t, t_end,
				&h, y);

		if (status != GSL_SUCCESS)
			break;

		if (ion->traj_dump)
			ion->traj.push_back(std::array<float, 7>{{(float) t, 
					(float) y[0], 
					(float) y[1], 
					(float) y[2], 
					(float) y[3], 
					(float) y[4], 
					(float) y[5]}});
			//printf ("%.5e %.5e %.5e %.5e %.5e %.5e %.5e\n", t, y[0], y[1], y[2], y[3], y[4], y[5]);
	}

//	if (!traj_dump)
	//	printf ("%.5e %.5e %.5e %.5e\n", ion_energy/MeV, y[0], y[1], y[2]);

	
	my_ion->impact = f3point(y[0], y[1], y[2]);


	gsl_odeiv_evolve_free (e);
	gsl_odeiv_control_free (c);
	gsl_odeiv_step_free (s);

}


// definizione stepping Lorenz in TP definita da struttura annessa
int lorentzStep(double t, const double *y, double *dy, void *params)
{
	// simulation status
	f3point cur_position = f3point(y[0],y[1],y[2]);
	bool sim_stat = ((class tpSystem *)params)->sim_box->isInside(cur_position);

	if (sim_stat) {

		// definizione campi
		f3point c_E = ((class tpSystem *)params)->Efield->getField(cur_position);
		f3point c_B = ((class tpSystem *)params)->Bfield->getField(cur_position);
		double c_q = ((class tpSystem *)params)->my_ion->charge;
		double c_m = ((class tpSystem *)params)->my_ion->mass;
		
		// derivazione
		dy[0] = y[3];
		dy[1] = y[4];
		dy[2] = y[5];
		dy[3] = (c_q/c_m)*(c_E.x() + y[4]*c_B.z() - y[5]*c_B.y());
		dy[4] = (c_q/c_m)*(c_E.y() + y[5]*c_B.x() - y[3]*c_B.z());
		dy[5] = (c_q/c_m)*(c_E.z() + y[3]*c_B.y() - y[4]*c_B.x());
	} else {
		// stop simulation outside sim_box
		dy[0] = 0;
		dy[1] = 0;
		dy[2] = 0;
		dy[3] = 0;
		dy[4] = 0;
		dy[5] = 0;
	}

	return GSL_SUCCESS;
}

// fuori dalle colonne d'Ercole
int lorentzStepRelativistic(double t, const double *y, double *dy, void *params)
{
	// simulation status
	f3point cur_position = f3point(y[0],y[1],y[2]);
	bool sim_stat = ((class tpSystem *)params)->sim_box->isInside(cur_position);

	if (sim_stat) {

		// definizione campi
		f3point c_E = ((class tpSystem *)params)->Efield->getField(cur_position);
		f3point c_B = ((class tpSystem *)params)->Bfield->getField(cur_position);
		double c_q = ((class tpSystem *)params)->my_ion->charge;
		double c_m = ((class tpSystem *)params)->my_ion->mass;
		double c_g = ((class tpSystem *)params)->my_ion->gamma;
		
		// derivazione
		dy[0] = y[3];
		dy[1] = y[4];
		dy[2] = y[5];
		dy[3] = (c_q/(c_m*c_g))*(c_E.x() + y[4]*c_B.z() - y[5]*c_B.y());
		dy[4] = (c_q/(c_m*c_g))*(c_E.y() + y[5]*c_B.x() - y[3]*c_B.z());
		dy[5] = (c_q/(c_m*c_g))*(c_E.z() + y[3]*c_B.y() - y[4]*c_B.x());
	} else {
		// stop simulation outside sim_box
		dy[0] = 0;
		dy[1] = 0;
		dy[2] = 0;
		dy[3] = 0;
		dy[4] = 0;
		dy[5] = 0;
	}

	return GSL_SUCCESS;
}
