%Module PyNeutrino
%Import QtGui/QtGuimod.sip
%Import QtCore/QtCoremod.sip

%MappedType QList<QList<qreal> >
{
%TypeHeaderCode
#include <QList>
%End

%ConvertFromTypeCode
  // Create the list.
  PyObject *l;
  if ((l = PyList_New(sipCpp->size())) == NULL)
    return NULL;

  const sipMappedType* qlist_type = sipFindMappedType("QList<qreal>");
  // Set the list elements.
  for (int i = 0; i < sipCpp->size(); ++i) {
    QList<qreal>* t = new QList<qreal>(sipCpp->at(i));
    PyObject *tobj;

    if ((tobj = sipConvertFromMappedType(t, qlist_type, sipTransferObj)) == NULL) {
      Py_DECREF(l);
      delete t;
      return NULL;
    }
    PyList_SET_ITEM(l, i, tobj);
  }

  return l;
%End

%ConvertToTypeCode
  const sipMappedType* qlist_type = sipFindMappedType("QList<qreal>");

  if (sipIsErr == NULL) {
    if (!PyList_Check(sipPy))
      return 0;

    for (int i = 0; i < PyList_GET_SIZE(sipPy); ++i)
      if (!sipCanConvertToMappedType(PyList_GET_ITEM(sipPy, i), qlist_type, SIP_NOT_NONE))
        return 0;

    return 1;
  }
  
  QList<QList<qreal> > *ql = new QList<QList<qreal> >;

  for (int i = 0; i < PyList_GET_SIZE(sipPy); ++i) {
    int state;
    QList<qreal> * t = reinterpret_cast< QList<qreal> * >(sipConvertToMappedType(PyList_GET_ITEM(sipPy, i), qlist_type, sipTransferObj, SIP_NOT_NONE, &state, sipIsErr));
    if (*sipIsErr) {
      sipReleaseMappedType(t, qlist_type, state);
      delete ql;
      return 0;
    }
    ql->append(*t);
    sipReleaseMappedType(t, qlist_type, state);
  }

  *sipCppPtr = ql;
  return sipGetState(sipTransferObj);
%End
};

class neutrino : QMainWindow {
%TypeHeaderCode
#include "../src/neutrino.h"
#include <QtGlobal>
%End
public:
	QList<QList<qreal> > getData(int=-1);
	bool setData(QList<QList<qreal> >,int=-1);
	nLine* line(QString);
};

class nGenericPan : QMainWindow {
%TypeHeaderCode
#include "../src/nGenericPan.h"
%End
public:
	nGenericPan(neutrino *, QString);
};

class nLine : QGraphicsObject {
%TypeHeaderCode
#include "../src/graphics/nLine.h"
%End
public:
	nLine(neutrino*);
	QPolygonF getPoints();
	void setPoints(QPolygonF);
};


