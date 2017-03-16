#include <QtCore>
#include <string>
#include <iostream>
#include <iomanip>
#include <vector>

int main(int , char **) {
	QFile fileout("../../src/gui/graphics/neutrinoPalettes.cc");
	fileout.open(QIODevice::WriteOnly | QIODevice::Text);
	QTextStream outs(&fileout);
    outs << "#include \"nView.h\"\nvoid nView::build_colormap() \{\n";
	QDir::setCurrent("cmaps");
    QStringList allFiles = QDir().entryList(QDir::NoDotAndDotDot | QDir::System | QDir::Hidden  | QDir::AllDirs | QDir::Files, QDir::DirsFirst);
	std::vector<unsigned char> palette(256*3);
    foreach (QString paletteFile, allFiles) {
		QFile filein(paletteFile);
		if (filein.open(QIODevice::ReadOnly | QIODevice::Text)) {
			QString paletteName=QFileInfo(paletteFile).baseName();
			bool allOk=true;
			int i=0;
			while (!filein.atEnd() && allOk) {
				QString line = QString(filein.readLine()).trimmed();
				if (line.startsWith("#")) {
					paletteName=line.remove(0,1).trimmed();
				} else {
					QStringList colorsToSplit=line.split(QRegExp("\\s+"),QString::SkipEmptyParts);
					if (colorsToSplit.size()==3) {
						bool ok0,ok1,ok2;
						unsigned int redInt=colorsToSplit.at(0).toUInt(&ok0);
						unsigned int greenInt=colorsToSplit.at(1).toUInt(&ok1);
						unsigned int blueInt=colorsToSplit.at(2).toUInt(&ok2);
						if (ok0 && ok1 && ok2 && redInt<256 && greenInt<256 && blueInt<256) {
							if (i<256) {
								palette[3*i+0]=redInt;
								palette[3*i+1]=greenInt;
								palette[3*i+2]=blueInt;
								i++;
							} else {
								allOk=false;
							}
						}
					} else {
						allOk=false;
					}
				}
			}
			if (i==256 && allOk) {
				qDebug() << paletteFile << paletteName;
				outs << "\tnPalettes[\"" << paletteName << "\"] = {" << (int)palette[0];
				for (int i=1; i<768; i++) {
					outs << "," << (int)palette[i];
				}
				outs << "};" << endl;
			}
			outs.setFieldWidth(0);			
		}   
		filein.close();
	}
	outs << "}" << endl;
}
