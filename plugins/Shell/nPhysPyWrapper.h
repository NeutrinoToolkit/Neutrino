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
	

};

#endif
