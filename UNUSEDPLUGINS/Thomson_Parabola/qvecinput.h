#include <iostream>

#include <QLineEdit>

#include <QObject>
#include <QDebug>
#include <QRegularExpression>
#include <QRegularExpressionValidator>

#include "tridimvec.h"

#ifndef __qvecinput
#define __qvecinput

typedef tridimvec<double> f3point;

class QVecInput : public QLineEdit {
	Q_OBJECT

public:

	QVecInput(QWidget *parent = 0)
		: QLineEdit(parent)
	{
        QRegularExpression rx("[(]{1}[0-9.eE-]+:[0-9.eE-]+:[0-9.eE-]+[)]{1}");
        v = new QRegularExpressionValidator(rx, 0);
		setValidator(v);

		connect(this, SIGNAL(editingFinished()), SLOT(editingFinished()));
		
		setPlaceholderText(tr("(0.0:0.0:0.0)"));
	}


	~QVecInput() {
		delete v;
	}

signals:
	void vecInput(f3point);

protected:
    QRegularExpressionValidator *v;

protected slots:
	void editingFinished()
	{
		emit vecInput(f3point(text().toStdString().c_str()));
	}
};

#endif
