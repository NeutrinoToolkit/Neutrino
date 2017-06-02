#ifndef __nphys_py_wrapper
#define __nphys_py_wrapper

#include <cmath>

#ifdef HAVE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#endif

#include "PythonQt.h"


#include <QtGui>
#include <QWidget>

#include "neutrino.h"


//! Wrapper for images nPhysD
/*!
 You can create or get nPhysD as well applyin some filter
*/
Q_DECLARE_METATYPE(nPhysD*);

class nPhysPyWrapper : public QObject {
    Q_OBJECT

public slots:
    nPhysD* new_nPhysD() {return new nPhysD();};

    QList<nPhysD*> static_nPhysD_open(QString=QString());

    nPhysD* new_nPhysD(QVector<double>, QPair<int,int>);

    nPhysD* new_nPhysD(int, int, double val=0.0, QString name=QString("Python"));

    nPhysD* new_nPhysD(nPhysD*);

    void delete_nPhysD(nPhysD*); // python destructor

    QString getName(nPhysD*);

    void setName(nPhysD*, QString);

    double get(nPhysD*, double, double);

    double get(nPhysD*, QPointF);

    QPointF getOrigin(nPhysD*);

    void setOrigin(nPhysD*, QPointF);

    void setOrigin(nPhysD*, double, double);

    QPointF getScale(nPhysD*);

    void setScale(nPhysD*, QPointF);

    void setScale(nPhysD*, QVariant, QVariant=QVariant());

    QPair<double, double> getMinMax(nPhysD*);

    double getMin(nPhysD*);

    double getMax(nPhysD*);

    QPair<int,int> getShape(nPhysD *);

    int getSurf(nPhysD *);

    QStringList properties(nPhysD*);

    QVariant getProperty(nPhysD *, QString);
    
    void setProperty(nPhysD *, QString, QVariant);
    
    QVector<double> getData(nPhysD*); // geta data in row major order

#ifdef HAVE_NUMPY
    PyObject* toArray(nPhysD* my_phys) {
        DEBUG("here");
        init_numpy();
        DEBUG("here");
        std::vector<npy_intp> dims={(npy_intp)my_phys->getW(),(npy_intp)my_phys->getH()};
//        nPhysD *my_copy=new nPhysD();
//        *my_copy=my_phys->copy();
//        return (PyObject*) PyArray_SimpleNewFromData(2, &dims[0], NPY_DOUBLE, my_copy->Timg_buffer);
        return (PyObject*) PyArray_SimpleNewFromData(2, &dims[0], NPY_DOUBLE, my_phys->Timg_buffer);
    }

    nPhysD* new_nPhysD(PyObject* my_py_obj){
        DEBUG("here");
        if (PyArray_Check(my_py_obj)) {
            PyArrayObject * arr = (PyArrayObject *)my_py_obj;
            if (PyArray_NDIM(arr)==2){
                auto dims=PyArray_DIMS(arr);
                double *data = reinterpret_cast<double*>(PyArray_DATA(arr));
                DEBUG(dims[0] << " x " << dims[1]);
                PyObject* objectsRepresentation = PyObject_Repr(my_py_obj);
                std::string name(PyString_AsString(objectsRepresentation));
                Py_DECREF(objectsRepresentation);
                nPhysD *my_phys = new nPhysD(dims[0], dims[1],0,name);
                my_phys->setShortName("numpy");
                for (npy_intp i=0; i<(npy_intp) my_phys->getSurf(); i++) {
                    my_phys->set(i,data[i]);
                }
                my_phys->TscanBrightness();
                Py_INCREF(my_py_obj);
                return my_phys;
            }
        }
        DEBUG("expected sequence");
        return nullptr;
    }

private:
#if PY_MAJOR_VERSION >= 3
    int
#else
    void
#endif
    init_numpy()
    {
        import_array();
    }
#endif

};

#endif
