#include "holderGUI.h"
#include <QImageReader>
#include <QFileDialog>
#include <QListWidgetItem>

holderGUI::holderGUI() : QMainWindow() {
	setAttribute(Qt::WA_DeleteOnClose);
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
		std::vector<nPhysD*> retlist = nHolder::getInstance().fileOpen(fname.toStdString());
		for (auto& img: retlist) {
			addPhys(img);
		}
	}
}

void holderGUI::addPhys(nPhysD* my_phys) {
	graphicsView->showPhys(my_phys);
	QListWidgetItem *my_item= new QListWidgetItem(listPhys);
	my_item->setText(QString::fromStdString(my_phys->getName()));
	QVariant my_var=QVariant::fromValue(qobject_cast<nPhysD*>(my_phys));
	qDebug() << my_phys << my_var;
	my_item->setData(1,my_var);
	connect(my_phys, SIGNAL(destroyed(QObject*)), this, SLOT(delPhys(QObject*)));
}

void holderGUI::delPhys(QObject* my_obj) {
	if (my_obj) {
		for (int i=0; i< listPhys->count(); i++ ) {
			if (listPhys->item(i)->data(1) == QVariant::fromValue(static_cast<nPhysD*>(my_obj))) {
				delete listPhys->takeItem(i);
			}
		}
	}
}
