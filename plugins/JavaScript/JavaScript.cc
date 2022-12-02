/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 *
 *    This file is part of neutrino.
 *
 *    Neutrino is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    Neutrino is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *    Contact Information:
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */

#include "JavaScript.h"
#include "nPhysImageF.h"

Q_DECLARE_METATYPE(QList<nGenericPan*>);
Q_DECLARE_METATYPE(QList<neutrino*>);

JavaScript::JavaScript(neutrino *my_nparent) : nGenericPan(my_nparent),
my_eng(my_nparent)
{
    setupUi(this);
    my_eng.globalObject().setProperty("neu", my_eng.newQObject(nparent));
    my_eng.globalObject().setProperty("nApp", my_eng.newQObject(qApp));
    my_eng.setObjectOwnership(nparent, QJSEngine::CppOwnership);
    my_eng.setObjectOwnership(qApp, QJSEngine::CppOwnership);

    my_eng.installExtensions(QJSEngine::ConsoleExtension);

    qRegisterMetaType<nGenericPan*>("nGenericPan*");
    qRegisterMetaType<nPhysD*>("nPhysD*");
    qRegisterMetaType<QRectF*>("QRectF*");

    splitter->setStretchFactor(0, 10);
    splitter->setStretchFactor(1, 1);

    QKeySequence key_seq=QKeySequence(Qt::CTRL | Qt::Key_Return);
    command->setToolTip("Press "+key_seq.toString(QKeySequence::NativeText)+" to execute"+command->toolTip());
    QShortcut* my_shortcut = new QShortcut(key_seq, command);
    connect(my_shortcut, SIGNAL(activated()), this, SLOT(runIt()));
    show();
}

void JavaScript::runIt() {
    saveDefaults();
    QJSValue retval;
    output->clear();
    QString mytext=command->toPlainText();
    if(QFileInfo::exists(mytext)) {
        QFile t(mytext);
        t.open(QIODevice::ReadOnly| QIODevice::Text);
        QTextStream out(&t);
        QString toRun=out.readAll();
        t.close();
        retval = my_eng.evaluate(toRun, mytext);
    } else {
        qDebug() << mytext;
        retval = my_eng.evaluate(mytext);
    }
    output->setPlainText(retval.toString());
}

void JavaScript::on_actionOpen_File_triggered() {
    QString fname=QFileDialog::getOpenFileName(this,tr("Open JS file"),property("NeuSave-fileJS").toString(),tr("JS file")+QString(" (*.js);;")+tr("Any files")+QString(" (*)"));
    if (!fname.isEmpty()) {
        setProperty("NeuSave-fileJS", fname);
        QFile f(fname);
        if (f.open(QFile::ReadOnly | QFile::Text)) {
            QTextStream in(&f);
            command->setPlainText(in.readAll());
        }
    }
}

void JavaScript::on_actionSave_File_triggered() {
    QString fname=QFileDialog::getSaveFileName(this,tr("Save JS file"),property("NeuSave-fileJS").toString(),tr("JS file")+QString(" (*.js);;")+tr("Any files")+QString(" (*)"));
    if (!fname.isEmpty()) {
        setProperty("NeuSave-fileJS", fname);
        QFile f(fname);
        if (f.open(QFile::ReadWrite | QFile::Text)) {
            QTextStream in(&f);
            in << command->toPlainText();
        }
        command->setPlainText(fname);
    }
}
