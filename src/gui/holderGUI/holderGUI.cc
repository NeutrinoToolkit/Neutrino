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
    formats+="Neutrino Images (*.txt *.neu *.neus *.tif *.tiff *.hdf *.png *.pgm *.ppm *.sif *.b16 *.spe *.pcoraw *.img *.raw *.fits *.inf *.gz);;";
    formats+="Images (";
    foreach (QByteArray format, QImageReader::supportedImageFormats() ) {
        formats+="*."+format+" ";
    }
    formats.chop(1);
    formats+=");;";
    formats+=("Any files (*)");

    QStringList fnames = QFileDialog::getOpenFileNames(this,tr("Open Image(s)"),property("NeuSave-fileOpen").toString(),formats);

    setProperty("NeuSave-fileOpen", fnames);
    foreach (QString fname, fnames) {
        std::vector<nPhysD*> retlist = nHolder::getInstance().fileOpen(fname.toStdString());
        for (auto& img: retlist) {
            graphicsView->showPhys(img);
        }
    }

}
