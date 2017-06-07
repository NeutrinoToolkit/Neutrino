#include "nPhysPyWrapper.h"
#include "nPhysWave.h"
#include "nPhysFormats.h"
#include "nApp.h"

/**
 nPhysD "creator" set a new nPhysD from array and shape data, usually to push data from Python
 */
QList<nPhysD*> nPhysPyWrapper::static_nPhysD_open(QString fname){
    QList<nPhysD*> my_list;
    if (fname.isEmpty()) {
        QString formats("");
        formats+="Neutrino Images (*.txt *.neu *.neus *.tif *.tiff *.hdf *.png *.sif *.b16 *.spe *.pcoraw *.img *.raw *.fits *.inf *.gz);;";
        formats+="Images (";
        foreach (QByteArray format, QImageReader::supportedImageFormats() ) {
            formats+="*."+format+" ";
        }
        formats.chop(1);
        formats+=");;";
        formats+=("Any files (*)");

        fname = QFileDialog::getOpenFileName(NULL,tr("Open Image(s)"),property("fileOpen").toString(),formats);
    }
    if (!fname.isEmpty( )) {
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
    DEBUG("new_nPhysD new_nPhysD new_nPhysD new_nPhysD new_nPhysD ");
    nPhysD *ret_phys = new nPhysD(*phys);
    return ret_phys;
}

#ifdef HAVE_NUMPY


#define __map_numpy(__arr,__my_phys,__numpy_type,__cplusplus_type) case __numpy_type : {__cplusplus_type *data = (__cplusplus_type*) PyArray_DATA(__arr); for (npy_intp i=0; i<(npy_intp) __my_phys->getSurf(); i++) {__my_phys->set(i,(double)data[i]);} break;}

nPhysD* nPhysPyWrapper::new_nPhysD(PyObject* my_py_obj){
    DEBUG("here " << my_py_obj);
    init_numpy();
    if (PyArray_Check(my_py_obj)) {
        PyArrayObject * arr = (PyArrayObject *)my_py_obj;
        DEBUG("here " << arr);
        if (arr && PyArray_NDIM(arr)==2){
            auto dims=PyArray_DIMS(arr);
            DEBUG(dims[0] << " x " << dims[1] << " " << PyArray_TYPE(arr));
            PyObject* objectsRepresentation = PyObject_Repr(my_py_obj);
            std::string name(PyString_AsString(objectsRepresentation));
            Py_DECREF(objectsRepresentation);
            nPhysD *my_phys = new nPhysD(dims[1], dims[0],std::numeric_limits<double>::quiet_NaN(),name);
            my_phys->setShortName("numpy");

            switch (PyArray_TYPE(arr)) {
                __map_numpy(arr,my_phys,NPY_BOOL        , bool                  );
                __map_numpy(arr,my_phys,NPY_UBYTE       , char                  );
                __map_numpy(arr,my_phys,NPY_SHORT       , short                 );
                __map_numpy(arr,my_phys,NPY_USHORT      , unsigned short        );
                __map_numpy(arr,my_phys,NPY_INT         , int                   );
                __map_numpy(arr,my_phys,NPY_UINT        , unsigned int          );
                __map_numpy(arr,my_phys,NPY_LONG        , long int              );
                __map_numpy(arr,my_phys,NPY_ULONG       , unsigned long int     );
                __map_numpy(arr,my_phys,NPY_LONGLONG    , long long int         );
                __map_numpy(arr,my_phys,NPY_ULONGLONG   , unsigned long long int);
                __map_numpy(arr,my_phys,NPY_FLOAT       , float                 );
                __map_numpy(arr,my_phys,NPY_DOUBLE      , double                );
                __map_numpy(arr,my_phys,NPY_LONGDOUBLE  , long double           );
                default:
                    DEBUG("it's a trap!")
                    break;
            }

            my_phys->TscanBrightness();
            Py_INCREF(my_py_obj);
            return my_phys;
        }
    }
    DEBUG("expected sequence");
    return nullptr;
}

#endif

/**
 nPhysD Destructor
 */
void nPhysPyWrapper::delete_nPhysD(nPhysD* phys) {
    DEBUG("delete_nPhysD delete_nPhysD delete_nPhysD delete_nPhysD delete_nPhysD");
    nApp* my_app=qobject_cast<nApp*> (qApp);
    bool found=false;
    if (my_app) {
        DEBUG("and here");
        for(auto& neu: my_app->neus()) {
            if (neu->getBufferList().contains(phys)) {
                neu->removePhys(phys);
                found=true;
            }
        }
    }
    if (!found) {
        delete phys;
    }
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

QPair<double, double> nPhysPyWrapper::getMinMax(nPhysD* phys){
    QPair<double, double> vec;
    return qMakePair(phys->get_min(), phys->get_max());
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


