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
#ifndef nCore_H_
#define nCore_H_

#include <QObject>
#include <QWidget>
#include <QMap>
#include <QPointer>
#include <vector>

class neutrino;
namespace Ui {
class nLogWin;
}

// what the Neutrino windows share, whatever the QApplication is
// (nApp in the executable, a plain QApplication when Neutrino is used as a library, e.g. from PySide).
// It is created at the first call of nCore::self(), which needs an existing QApplication
class nCore : public QObject {
    Q_OBJECT

public:
    static nCore* self();

    // the time (ms) of the messages of the status bars: all of them, the code shows its messages without a timeout
    static constexpr int statusMessageTime=5000;

    QMap<QString, std::vector<unsigned char>> nPalettes;

    // the log window (created again if it was deleted at the quit of the application)
    QWidget* logWin();
    Ui::nLogWin* logUi();

    // shows the log window with an exception (from the places that catch them: there is no QApplication::notify
    // to catch them when Neutrino is a library)
    void showException(const char *what);

    // opens in a Neutrino window the files that macOS gives to the application (Finder, Dock):
    // enabled by the executable only, a library user (e.g. PySide) keeps them
    void handleFileOpen(bool enable);

protected:
    bool eventFilter(QObject *obj, QEvent *ev) override;

public slots:

    // closes the Neutrino windows only (not the other windows of the application)
    void closeAllWindows();

    QList<neutrino*> neus();

    static void changeThreads(int);

    bool addPaletteFile(QString);
    void addDefaultPalettes();

    void clearLog();
    void copyLog();
    void saveLog();
    void findLogText();

    void forceDecimalDot(bool);
    int checkUpdates();

    void saveLogSettings();

    // deletes the log window (saving its settings): before the QApplication is destroyed, e.g. at the exit of python
    // (it is also done when the application quits)
    void deleteLogWin();

private:
    // never deleted: views keep a reference to nPalettes
    nCore();

    // light or dark icons (resources icons.qrc / icons_dark.qrc), chosen once, before the first window
    // returns true for the dark ones
    bool selectIconTheme();

    void createLogWin();

    static void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg);

    QPointer<QWidget> log_win;
    Ui::nLogWin *log_win_ui;
};

#endif
