/*
 * This file is NOT open source. 
 *
 * Any use is FORBIDDEN unless by written permission
 *
 * (C) Alessandro Flacco 2014-15
 *
 */

#include "Thomson_Parabola.h"

#include "neutrino.h"
#include "nSpectrum.h"

#include <QFileDialog>

// ------------------------------------------------------------------------------

Thomson_Parabola::Thomson_Parabola(neutrino *nparent)
    : nGenericPan(nparent)
{
	// here my pan creator
	
	// you probably want to instantiate the widget from Ui::
	my_w.setupUi(this);
	
	// create tpSystem
	my_tp = new tpSystem();

//	// initialize GL draw area
//    tpDraw = new tpGlDraw(this);
//    my_w.gridGl->addWidget(tpDraw);
//
//	// pass tpSystem boxes to GL for plotting
//	tpDraw->addTribox(my_tp->Efield);
//	tpDraw->addTribox(my_tp->Bfield);

    // moved to neutrino load/save system
    //connect(my_w.loadConfig_pb, SIGNAL(clicked()), this, SLOT(load_config()));
    //connect(my_w.saveConfig_pb, SIGNAL(clicked()), this, SLOT(save_config()));


    //connect(my_w.track_pb, SIGNAL(clicked()), this, SLOT(run_simulation()));
	//connect(my_w.run_pb, SIGNAL(clicked()), this, SLOT(run_simulation()));

	// table
	connect(my_w.addTrack_pb, SIGNAL(clicked()), this, SLOT(addTrack()));
	connect(my_w.trackTable, SIGNAL(cellChanged(int, int)), this, SLOT(updateSingleTrack(int, int)));

	// signals
	
	// vec inputs
	QList<QString> vecInputList;
	vecInputList << "boxV1" << "boxV2";
       	vecInputList << "Bcenter" << "Bbox" << "Bvec";
       	vecInputList << "Ecenter" << "Ebox" << "Evec";
	foreach (QString objn, vecInputList) {
		QVecInput *vi = findChild<QVecInput *>(objn);

		if (vi != 0) {
			connect(vi, SIGNAL(vecInput(f3point)), this, SLOT(vecInput(f3point)));
        } else DEBUG("CANNOT find "<<objn.toLatin1().constData());
	}

	// lines
	//my_line = new nSpectrum(nparent);

	// tweak nSpectrum
	//my_line->my_w.points->setColumnCount(5);
	//QStringList lbls; lbls<<"X"<<"Y"<<"K [MeV/nucl.]"<<"Q [e+]"<<"m [amu]";
	//my_line->my_w.points->setHorizontalHeaderLabels(lbls);
	// --> moved to nSpectrum constructor

    show();
}

void
Thomson_Parabola::vecInput(f3point p)
{
	QVecInput *s = (QVecInput *)sender();
    DEBUG(5,"got "<<p<<" from "<<s->objectName().toLatin1().constData());
	
	// questa e' palloserrima
	if (s->objectName() == "boxV1") {
		my_tp->sim_box->setV1(p);
	} else if (s->objectName() == "boxV2") {
		my_tp->sim_box->setV2(p);
	} else if (s->objectName() == "Bcenter") {
		my_tp->Bfield->setCenter(p);
	} else if (s->objectName() == "Bbox") {
		my_tp->Bfield->setSize(p);
	} else if (s->objectName() == "Bvec") {
		double val = p.mod();
		my_tp->Bfield->assignScalarQuantity(val, p.normVec());
	} else if (s->objectName() == "Ecenter") {
		my_tp->Efield->setCenter(p);
	} else if (s->objectName() == "Ebox") {
		my_tp->Efield->setSize(p);
	} else if (s->objectName() == "Evec") {
		double val = p.mod();
		my_tp->Efield->assignScalarQuantity(val, p.normVec());
	}

	//if (my_w.follow_cb->isChecked()) {
	//	my_w.run_pb->click();
	//}
}

//void
//Thomson_Parabola::load_config(void)
//{
//	QString confname = QFileDialog::getOpenFileName (this, QString("Select tp conf file"), "", QString("*.conf"));
//	if (! confname.isEmpty()) {
//		my_tp->parseConfig(confname.toStdString().c_str());

//		// sync to widget
//		if (my_tp->isValid()) {
//			findChild<QVecInput *>("boxV1")->setText(my_tp->sim_box->myvertex1.str().c_str());
//			findChild<QVecInput *>("boxV2")->setText(my_tp->sim_box->myvertex2.str().c_str());

//			findChild<QVecInput *>("Ecenter")->setText(my_tp->Efield->getCenter().str().c_str());
//			findChild<QVecInput *>("Ebox")->setText(my_tp->Efield->getSize().str().c_str());
//			findChild<QVecInput *>("Evec")->setText((my_tp->Efield->fieldValue*my_tp->Efield->myfield_versor).str().c_str());

//			findChild<QVecInput *>("Bcenter")->setText(my_tp->Bfield->getCenter().str().c_str());
//			findChild<QVecInput *>("Bbox")->setText(my_tp->Bfield->getSize().str().c_str());
//			findChild<QVecInput *>("Bvec")->setText((my_tp->Bfield->fieldValue*my_tp->Bfield->myfield_versor).str().c_str());
//		}
//	}


//}

//void
//Thomson_Parabola::save_config(void)
//{
//	QString confname = QFileDialog::getSaveFileName (this, QString("Select tp conf file"), "", QString("*.conf"));
//	if (! confname.isEmpty()) {
//		my_tp->writeConfig(confname.toStdString().c_str());
//	}
       	
//}


void
Thomson_Parabola::run_simulation(void)
{
	// whoami
	bool tracking = false;
	//QPushButton *pb = (QPushButton *)sender();
	//if (pb->objectName() == "track_pb") {
	//	std::cout<<"tracking simulation"<<std::endl;
	//	tracking = true;
	//}

	ionImpact ion;

	// adding dependence for protons and heavier ions
	ion.charge = sp.Z*GSL_CONST_MKSA_ELECTRON_CHARGE;
	ion.mass = sp.A*PROTON_ION_MASS;

	float iE = sp.iE_mev*MeV;
	float eE = sp.eE_mev*MeV;
	int nSim = sp.n_points;
	float stepE = (eE-iE)/(nSim-1);

	nSpectrum *my_line = sp.nsp;
	if (my_line == NULL) {
		DEBUG("something went very wrong with nLine allocation");
		return;

		// should throw an exception, though
	}

	if (tracking)
		ion.traj_dump = true;

	int traj_size_hint = 1000;

	vec2f orig(0,0), scale(1,1);
    if (currentBuffer) {
        orig = currentBuffer->get_origin();
        scale = currentBuffer->get_scale();
        DEBUG(5,"img scale: "<<scale<<", img orig: "<<orig);
	}

	// TODO: I should derive class nSpectrum to support in a more general way line representation with
	// point description (i.e. to associate to each point, apart from its x/y coordinates, other data, like
	// energy, mass, charge, etc.
	//
	// WORKAROUND: for the moment I just explicitely set values in the QTabWid. Energies are held in an std::vector

	QPolygonF lp;
	std::vector<float> ens;
	ens.reserve(nSim);
	for (int ii=0; ii<nSim; ii++) {
		//ion.energy = iE+ii*stepE;
		// try K^(3/2) progression
		ion.energy = iE+std::pow(.001*(float)ii,3./2)*(eE-iE)/(std::pow(.001*(float)(nSim-1), 3./2));
		ens.push_back(ion.energy/MeV);

		if (tracking) {
			ion.traj.clear();
			ion.traj.reserve(traj_size_hint);
		}

		my_tp->getImpact(&ion);

		if (tracking) {
			//std::cout<<"tracking sim run in "<<ion.traj.size()<<" steps; corrected hint"<<std::endl;
			traj_size_hint = ion.traj.size();
		}

		vec2f d_impact = vec2f(ion.impact.x(), ion.impact.y());
		vec2f sc_impact = (1./scale.x())*(d_impact)+orig; // setP in nSpectrum rimane su riferimento matrice originale
		lp<<QPoint(sc_impact.x(), sc_impact.y());
		//std::cout<<"got impact at "<<ion.impact<<" (img at "<<sc_impact<<")"<<std::endl;
	}

	
	// update points table in nLine
	// WARNING: this is the ONLY mechanism actually passing energy
	// associated to curvilinear coordinate from tp-plugin to nSpectrum
	my_line->setPoints(lp);
	for (int ii=0; ii<ens.size(); ii++) {
		QTableWidgetItem *en_it = new QTableWidgetItem(QString::number(ens[ii]));
		my_line->my_w.points->setItem(ii, 2, en_it);
	
		QTableWidgetItem *charge_it = new QTableWidgetItem(QString::number(sp.Z));
		my_line->my_w.points->setItem(ii, 3, charge_it);
		
		QTableWidgetItem *mass_it = new QTableWidgetItem(QString::number(sp.A));
		my_line->my_w.points->setItem(ii, 4, mass_it);
	
	}

}

// tracks table management
void
Thomson_Parabola::addTrack()
{
	my_w.trackTable->insertRow(my_w.trackTable->rowCount());
	int rown = my_w.trackTable->rowCount()-1;
	
	my_w.trackTable->setItem(rown, 0, new QTableWidgetItem(QString("%1").arg(my_w.A_sb->value())));
	my_w.trackTable->setItem(rown, 1, new QTableWidgetItem(QString("%1").arg(my_w.Z_sb->value())));

	float initE  = my_w.initE_sb->value(), endE = my_w.endE_sb->value();
	if (my_w.escale_cb->isChecked()) {
		// scale energies to be in the same energy range as protons
		float rescale = my_w.Z_sb->value()/sqrt(my_w.A_sb->value());
		initE *= rescale;
		endE *= rescale;
	}
	my_w.trackTable->setItem(rown, 2, new QTableWidgetItem(QString("%1").arg(initE)));
	my_w.trackTable->setItem(rown, 3, new QTableWidgetItem(QString("%1").arg(endE)));
	my_w.trackTable->setItem(rown, 4, new QTableWidgetItem(QString("%1").arg(my_w.points_sb->value())));
	
	// del button
	my_w.trackTable->setCellWidget(rown, 5, new QPushButton("Del"));
	connect(my_w.trackTable->cellWidget(rown, 5), SIGNAL(clicked()), this, SLOT(removeTrack()));

	updateTracks();
}

void
Thomson_Parabola::removeTrack()
{
	QObject *so = QObject::sender();
	for (int rr=0; rr<my_w.trackTable->rowCount(); rr++) {
		for (int cc=0; cc<my_w.trackTable->columnCount(); cc++) {
			if (so == my_w.trackTable->cellWidget(rr, cc)) {
				my_w.trackTable->removeRow(rr);
				delete my_tracks[rr];
				my_tracks.remove(rr);
			}
		}
	}


	updateTracks();
}


// not elegant, should be merged with tpGUI::updateTracks, somehow
void
Thomson_Parabola::updateSingleTrack(int cr, int cc)
{
	// if row being edited is not in my_tracks (check length!) it is being added, hence not to be taken in account
	if ((cr+1) > my_tracks.size()) {
		std::cerr<<"NOT updating single track "<<cr<<std::endl;
	} else {
		sp.A = my_w.trackTable->item(cr, 0)->data(0).toInt();
		sp.Z = my_w.trackTable->item(cr, 1)->data(0).toInt();
		sp.iE_mev = my_w.trackTable->item(cr, 2)->data(0).toFloat();
		sp.eE_mev = my_w.trackTable->item(cr, 3)->data(0).toFloat();
		sp.n_points = my_w.trackTable->item(cr, 4)->data(0).toInt();

		if (my_tracks[cr] == NULL) {
			DEBUG("allocate nsp");
            my_tracks[cr] = new nSpectrum(this);
		}
		sp.nsp = my_tracks[cr];
        sp.nsp->my_w.name->setText(QString("A: %1, Z: %2").arg(sp.A).arg(sp.Z));
		run_simulation();
	}
	
}

void
Thomson_Parabola::updateTracks(void)
{
    DEBUG(5, "update tracks called");
	my_tracks.resize(my_w.trackTable->rowCount());
	for (int rr=0; rr < my_w.trackTable->rowCount(); rr++) {
		sp.A = my_w.trackTable->item(rr, 0)->data(0).toInt();
		sp.Z = my_w.trackTable->item(rr, 1)->data(0).toInt();
		sp.iE_mev = my_w.trackTable->item(rr, 2)->data(0).toFloat();
		sp.eE_mev = my_w.trackTable->item(rr, 3)->data(0).toFloat();
		sp.n_points = my_w.trackTable->item(rr, 4)->data(0).toInt();

		if (my_tracks[rr] == NULL) {
			DEBUG("allocate nsp");
            my_tracks[rr] = new nSpectrum(this);
		}
		sp.nsp = my_tracks[rr];
        sp.nsp->my_w.name->setText(QString("A: %1, Z: %2").arg(sp.A).arg(sp.Z));
		run_simulation();
	}
}

