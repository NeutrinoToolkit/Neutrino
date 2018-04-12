#include "nPhysPyWrapper.h"
#include "nPhysWave.h"
#include "nPhysFormats.h"
#include "nApp.h"
#include "neutrino.h"
#ifdef HAVE_NUMPY
#include <numpy/arrayobject.h>
#endif


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
        std::vector<nPhysD*> my_vec=physFormat::phys_open(fname.toUtf8().constData());
        for(auto it : my_vec) {
            if (it->getSurf()>0) {
                it->property["keep_phys_alive"]=42;
                my_list << it;
            } else {
                delete it;
            }
        }
    }
    return my_list;
}

nPhysD* nPhysPyWrapper::new_nPhysD() {
    nPhysD *my_phys= new nPhysD();
    my_phys->setType(PHYS_DYN);
    my_phys->property["keep_phys_alive"]=42;
    return my_phys;
}

nPhysD* nPhysPyWrapper::new_nPhysD(QVector<double> tempData, QPair<int,int> my_shape){
    DEBUG("here");
    nPhysD *phys=new nPhysD(PHYS_DYN);
    phys->setType(PHYS_DYN);
    int h=my_shape.first; // row major: first rows(yw or height) than columns(xw or widthv)
    int w=my_shape.second;
    
    if (tempData.size()!=0  && tempData.size()==w*h) {
        phys->resize(w, h);
        for (int i=0;i<h*w;i++) {
            phys->set(i,tempData[i]);
        }
        phys->TscanBrightness();
    }
    phys->property["keep_phys_alive"]=42;
    return phys;
}

/**
 nPhysD creates an nPhysD of size x filled with val and name name
 */
nPhysD* nPhysPyWrapper::new_nPhysD(int width, int height, double val, QString name){
    DEBUG("here");
    nPhysD *phys=new nPhysD (width,height,val,name.toStdString());
    phys->setType(PHYS_DYN);
    phys->property["keep_phys_alive"]=42;
    return phys;
}

/**
 nPhysD copy constructor
 */
nPhysD* nPhysPyWrapper::new_nPhysD(nPhysD* phys) {
    DEBUG("new_nPhysD new_nPhysD new_nPhysD new_nPhysD new_nPhysD ");
    nPhysD *ret_phys = new nPhysD(*phys);
    ret_phys->property["keep_phys_alive"]=42;
    return ret_phys;
}

#ifdef HAVE_NUMPY


#if PY_MAJOR_VERSION >= 3
int neutrino_init_numpy()
{
import_array();
}
#else
void neutrino_init_numpy()
{
import_array();
}
#endif

PyObject* nPhysPyWrapper::toArray(nPhysD* my_phys) {
    neutrino_init_numpy();
    std::vector<npy_intp> dims={(npy_intp)my_phys->getH(),(npy_intp)my_phys->getW()};
    return (PyObject*) PyArray_SimpleNewFromData(2, &dims[0], NPY_DOUBLE, my_phys->Timg_buffer);
}

nPhysD* nPhysPyWrapper::new_nPhysD(PyObject* my_py_obj){
    neutrino_init_numpy();
    if (PyArray_Check(my_py_obj)) {
        PyArrayObject * my_arr = (PyArrayObject *)my_py_obj;
        if (my_arr && PyArray_NDIM(my_arr)==2 && PyArray_ISONESEGMENT(my_arr)){
            auto dims=PyArray_DIMS(my_arr);
            if (PyArray_ISFORTRAN(my_arr)) {
                std::swap(dims[0],dims[1]);
            }
            DEBUG("Contiguous: " << PyArray_ISCONTIGUOUS(my_arr) << " " << dims[0] << " x " << dims[1] << " " << PyArray_TYPE(my_arr));

            std::string name;
            PyObject* objectsRepresentation = PyObject_Repr(my_py_obj);
            if (objectsRepresentation) {
                name = PyString_AsString(objectsRepresentation);
            }
            DEBUG("name ------------------>" << name);
            Py_DECREF(objectsRepresentation);
            nPhysD *my_phys = new nPhysD(dims[1], dims[0],std::numeric_limits<double>::quiet_NaN(),name);
            my_phys->setType(PHYS_DYN);
            my_phys->setShortName("ndarray");

#define __map_numpy(__arr,__my_phys,__cpp_type)  {__cpp_type *data = (__cpp_type*) PyArray_DATA(__arr); for (npy_intp i=0; i<(npy_intp) __my_phys->getSurf(); i++) {__my_phys->set(i,(double)data[i]);} break;}

            switch (PyArray_TYPE(my_arr)) {
                case NPY_BOOL        : __map_numpy(my_arr,my_phys, bool                  );
                case NPY_UBYTE       : __map_numpy(my_arr,my_phys, char                  );
                case NPY_SHORT       : __map_numpy(my_arr,my_phys, short                 );
                case NPY_USHORT      : __map_numpy(my_arr,my_phys, unsigned short        );
                case NPY_INT         : __map_numpy(my_arr,my_phys, int                   );
                case NPY_UINT        : __map_numpy(my_arr,my_phys, unsigned int          );
                case NPY_LONG        : __map_numpy(my_arr,my_phys, long int              );
                case NPY_ULONG       : __map_numpy(my_arr,my_phys, unsigned long int     );
                case NPY_LONGLONG    : __map_numpy(my_arr,my_phys, long long int         );
                case NPY_ULONGLONG   : __map_numpy(my_arr,my_phys, unsigned long long int);
                case NPY_FLOAT       : __map_numpy(my_arr,my_phys, float                 );
                case NPY_DOUBLE      : __map_numpy(my_arr,my_phys, double                );
                case NPY_LONGDOUBLE  : __map_numpy(my_arr,my_phys, long double           );
                default:
                    DEBUG("it's a trap!")
                            break;
            }

            if (PyArray_ISFORTRAN(my_arr)) {
                physMath::phys_transpose(*my_phys);
            }

            my_phys->TscanBrightness();
            my_phys->property["keep_phys_alive"]=42;
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
void nPhysPyWrapper::delete_nPhysD(nPhysD* my_phys) {
    DEBUG("delete_nPhysD delete_nPhysD delete_nPhysD delete_nPhysD delete_nPhysD");
    nApp* my_app=qobject_cast<nApp*> (qApp);
    if (my_app) {
        my_app->processEvents();
        DEBUG("and here");
        for(auto& neu: my_app->neus()) {
            if (neu->getBufferList().contains(my_phys)) {
                neu->removePhys(my_phys);
            }
        }
    }
    if (my_phys->property["keep_phys_alive"].get_i()==42) {
        delete my_phys;
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


