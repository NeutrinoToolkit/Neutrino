#include "holderGUI.h"
#include <QImageReader>
#include <QFileDialog>

holderGUI::holderGUI() : QMainWindow() {
    DEBUG("HERE");
    setupUi(this);
    show();
}


void holderGUI::on_actionOpen_triggered() {
    QString formats("");
	formats+="Neutrino Images (*.txt *.neu *.neus *.tif *.tiff *.hdf *.sif *.b16 *.spe *.pcoraw *.img *.raw *.fits *.inf *.gz);;";
    formats+="Images (";
    foreach (QByteArray format, QImageReader::supportedImageFormats() ) {
        formats+="*."+format+" ";
    }
    formats.chop(1);
    formats+=");;";
    formats+=("Any files (*)");

    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Image(s)"),property("NeuSave-fileOpen").toString(),formats);
	openFiles(fnames);
}

void holderGUI::openFiles(QStringList fnames) {
	setProperty("NeuSave-fileOpen", fnames);
	foreach (QString fname, fnames) {
		std::vector<nPhysD> retlist = nHolder::getInstance().fileOpen(fname.toStdString());
		for (auto& img: retlist) {
			addPhys(img);
		}
	}
}

void holderGUI::addPhys(nPhysD* my_phys) {
	graphicsView->showPhys(my_phys);
	QListWidgetItem *pippo= new QListWidgetItem(listPhys);
	pippo->setText(QString::fromStdString(my_phys->getName()));
	pippo->setData(1,QVariant::fromValue(qobject_cast<QObject*>(my_phys)));
	connect(my_phys, SIGNAL(destroyed(QObject*)), this, SLOT(delPhys(QObject*)));
}

void holderGUI::delPhys(QObject* my_obj) {
	if (my_obj) {
		for (int i=0; i< listPhys->count(); i++ ) {
			qDebug() << listPhys->item(i);
			qDebug() << my_obj;
			qDebug() << listPhys->item(i)->data(1).value<QObject*>();
			if (listPhys->item(i)->data(1).value<QObject*>() == my_obj) {
				delete listPhys->takeItem(i);
			}
		}
		qDebug() << my_obj << " : " << sender();
	}
}
