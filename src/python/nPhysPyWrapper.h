#ifndef __nphys_py_wrapper
#define __nphys_py_wrapper

#include <cmath>
#include "PythonQt.h"

#include <QtGui>
#include <QWidget>

#include "neutrino.h"

//! Wrapper for images nPhysD
/*!
 You can create or get nPhysD as well applyin some filter
*/
class nPhysPyWrapper : public QObject {
	Q_OBJECT	

public slots:
	//! return a new empty nPhysD
	/*! */
	nPhysD* new_nPhysD() {return new nPhysD();};

    QList<nPhysD*> static_nPhysD_open(QString=QString());

	//! return a new nPhysD starting with a vector(row major order) and usually used to set data from python
	/*! */
	nPhysD* new_nPhysD(QVector<double>, QPair<int,int>); 
	
	//! return a new nPhysD of dimensions w and h filled with value val
	/*! */
	nPhysD* new_nPhysD(int, int, double val=0.0, QString name=QString("Python")); 
	
	//! copy contructor (contruct a fresh nPhysD * from an existing one)
	/*! */
	nPhysD* new_nPhysD(nPhysD*);

	//! this is the distructor of the object 
	/*! */
	void delete_nPhysD(nPhysD*); // python destructor
	
	//! returns the name of the nPhysD 
	/*! */
	QString getName(nPhysD*);

	//! set the name of the nPhysD 
	/*! */
	void setName(nPhysD*, QString);

	//! returns the value of the image at point x, y 
	/*! */
	double get(nPhysD*, double, double);
	//! returns the value of the image at point given via QPoint 
	/*! */
	double get(nPhysD*, QPointF);

	//! returns the Origin
	/*! */
	QPointF getOrigin(nPhysD*);
	
	//! set the Origin
	/*! */
	void setOrigin(nPhysD*, QPointF);
	
	//! set the Origin
	/*! */
	void setOrigin(nPhysD*, double, double);
	
	//! returns the scale of the matrix
	/*! */
	QPointF getScale(nPhysD*);
	
	//! set the scale of te matrix
	/*! */
	void setScale(nPhysD*, QPointF);
	
	//! set the scale of te matrix
	/*! */
	void setScale(nPhysD*, QVariant, QVariant=QVariant());
	
	//! returns the min and max
	/*! */
	QVector<double> getMinMax(nPhysD*);
	
	//! returns the min
	/*! */
	double getMin(nPhysD*);
	
	//! returns the max
	/*! */
	double getMax(nPhysD*);
	
	//! returns the shape of the matrix (# of rows and # of columns) 
	/*! */
    QPair<int,int> getShape(nPhysD *);

	//! returns the surface of the matrix (rows * columns) 
	/*! */
	int getSurf(nPhysD *); // get the matrix shape (2 values: height and width)


    QStringList properties(nPhysD*);

	//! returns the property
    QVariant getProperty(nPhysD *, QString);
    
	//! set the property
    void setProperty(nPhysD *, QString, QVariant);
    
	//! returns the data as 1D vector (row major order) 
	/*! */
	QVector<double> getData(nPhysD*); // geta data in row major order
	

};

#endif
