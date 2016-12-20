#include "nPhysPyWrapper.h"
#include "nPhysWave.h"
#include "nPhysFormats.h"
//#include "numpy/arrayobject.h"
//#include "numpy/npy_common.h"

/**
 nPhysD "creator" set a new nPhysD from array and shape data, usually to push data from Python
 */
QList<nPhysD*> nPhysPyWrapper::static_nPhysD_open(QString fname){
    QList<nPhysD*> my_list;
    if (fname.isEmpty()) {
        QString formats("");
        formats+="Neutrino Images (*.txt *.neu *.neus *.tif *.tiff *.hdf *.png *.pgm *.ppm *.sif *.b16 *.spe *.pcoraw *.img *.raw *.fits *.inf *.gz);;";
        formats+="Images (";
        foreach (QByteArray format, QImageReader::supportedImageFormats() ) {
            formats+="*."+format+" ";
        }
        formats.chop(1);
        formats+=");;";
        formats+=("Any files (*)");

        fname = QFileDialog::getOpenFileName(NULL,tr("Open Image(s)"),property("fileOpen").toString(),formats);
    }
    if (!fname.isEmpty()) {
        std::vector<nPhysD*> my_vec=phys_open(fname.toUtf8().constData());
        for(auto it : my_vec) {
            if (it->getSurf()>0) {
                my_list << it;
            } else {
                delete it;
            }
        }
    }
    return my_list;
}

nPhysD* nPhysPyWrapper::new_nPhysD(QVector<double> tempData, QPair<int,int> my_shape){
    DEBUG("here");
	nPhysD *phys=new nPhysD();
    int h=my_shape.first; // row major: first rows(yw or height) than columns(xw or widthv)
    int w=my_shape.second;
    
    if (tempData.size()!=0  && tempData.size()==w*h) {
        phys->resize(w, h);
        for (int i=0;i<h*w;i++) {
            phys->set(i,tempData[i]);
        }
        phys->TscanBrightness();
	}
	return phys;
}

/**
 nPhysD creates an nPhysD of size x filled with val and name name
 */
nPhysD* nPhysPyWrapper::new_nPhysD(int width, int height, double val, QString name){
    DEBUG("here");
	nPhysD *phys=new nPhysD (width,height,val,name.toStdString());
	return phys;
}

/**
 nPhysD copy constructor
 */
nPhysD* nPhysPyWrapper::new_nPhysD(nPhysD* phys) {
    DEBUG("here");
	nPhysD *ret_phys = new nPhysD(*phys);
	return ret_phys;
}

/**
 nPhysD Destructor
 */
void nPhysPyWrapper::delete_nPhysD(nPhysD* obj) {
    delete obj;
}

QString nPhysPyWrapper::getName(nPhysD* phys) {
	return QString::fromStdString(phys->getName());
}

void nPhysPyWrapper::setName(nPhysD* phys, QString name) {
	phys->setName(name.toStdString());
}

double nPhysPyWrapper::get(nPhysD* phys, double x, double y){
	return phys->getPoint(x,y);
}

double nPhysPyWrapper::get(nPhysD* phys, QPointF p){
	return get(phys, p.x(), p.y());
}

QVector<double> nPhysPyWrapper::getMinMax(nPhysD* phys){
	QVector<double> vec;
	vec << phys->get_min() <<  phys->get_max();
	return vec;
}

double nPhysPyWrapper::getMin(nPhysD* phys){
	return phys->get_min();
}

double nPhysPyWrapper::getMax(nPhysD* phys){
	return phys->get_max();
}

QPointF nPhysPyWrapper::getOrigin(nPhysD*phys) {
	return QPointF(phys->get_origin().x(), phys->get_origin().y());
}

void nPhysPyWrapper::setOrigin(nPhysD*phys, QPointF p) {
	phys->set_origin(p.x(),p.y());
}

void nPhysPyWrapper::setOrigin(nPhysD*phys, double x, double y){
	phys->set_origin(x,y);
}

QPointF nPhysPyWrapper::getScale(nPhysD*phys) {
	return QPointF(phys->get_scale().x(), phys->get_scale().y());
}

void nPhysPyWrapper::setScale(nPhysD*phys, QPointF p) {
	phys->set_scale(p.x(),p.y());
}

void nPhysPyWrapper::setScale(nPhysD*phys, QVariant varX, QVariant varY){
	if (!varY.isValid()) {
		varY=varX;
	}
	bool ok1,ok2;
	double x=varX.toDouble(&ok1);
	double y=varY.toDouble(&ok2);
	if (ok1 && ok2)	phys->set_scale(x,y);
}

QPair<int, int> nPhysPyWrapper::getShape(nPhysD *phys){
    if (phys)
        return qMakePair((int)phys->getH(), (int)phys->getW());
        
    return qMakePair(0, 0);
}


int nPhysPyWrapper::getSurf(nPhysD* phys){	
    return phys?phys->getSurf():0;
}

QStringList nPhysPyWrapper::properties(nPhysD* phys){
    QStringList retval;
    for(anymap::iterator it = phys->property.begin(); it != phys->property.end(); ++it) {
        retval << QString::fromStdString(it->first);
    }
    return retval;
}

QVariant nPhysPyWrapper::getProperty(nPhysD* phys, QString my_name){
    return toVariant(phys->property[my_name.toStdString()]);
}

void nPhysPyWrapper::setProperty(nPhysD* phys, QString prop_name, QVariant prop_val) {
    anydata pippo=toAnydata(prop_val);
    phys->property[prop_name.toStdString()]=pippo;
}

QVector<double> nPhysPyWrapper::getData(nPhysD* phys){
	QVector<double> tempData;
	tempData.clear();

	if (phys) {
		tempData.resize(phys->getSurf());
		for (size_t i=0; i<phys->getSurf(); i++) {
			tempData[i]=phys->point(i);
		}
	}
	return tempData;
}
