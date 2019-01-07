#include "nApp.h"
#include "neutrino.h"
#include <QtNetwork>
#include "ui_nLogWin.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QScrollBar>

nApp::nApp( int &argc, char **argv ) : QApplication(argc, argv),
    log_win(nullptr,Qt::Tool),
    log_win_ui(new Ui::nLogWin)
#ifndef __clang__
  ,
#ifdef __phys_debug
    qerr(std::cerr),
#endif
    qout(std::cout)
#endif
{

}

int nApp::exec() {

    log_win_ui->setupUi(&log_win);

#ifndef __phys_debug
    qInstallMessageHandler(nApp::myMessageOutput);
#endif

    addDefaultPalettes();

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");

    QCoreApplication::setOrganizationName("polytechnique");
    QCoreApplication::setOrganizationDomain("edu");
    QCoreApplication::setApplicationName("Neutrino");
    QCoreApplication::setApplicationVersion(__VER);

#if defined(Q_OS_MAC)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QCoreApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

#if QT_VERSION >= QT_VERSION_CHECK(5, 4, 0)
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
#endif
    QCoreApplication::addLibraryPath(QCoreApplication::applicationDirPath()+QString("/plugins"));

    changeLocale(my_set.value("locale",QLocale()).toLocale());
    changeThreads(my_set.value("threads",1).toInt());
    if (!my_set.contains("checkUpdates")) {
        my_set.setValue("checkUpdates",QMessageBox::Yes == QMessageBox::question(nullptr,tr("Attention"),tr("Do you want to check for updates?"), QMessageBox::Yes | QMessageBox::No));
    }
    if (my_set.value("checkUpdates",true).toBool()) {
        checkUpdates();
    }

    my_set.endGroup();

    connect(this, SIGNAL(lastWindowClosed()), this, SLOT(quit()));

    QObject::connect(log_win_ui->clearLog,&QPushButton::released,log_win_ui->logger,&QTextEdit::clear);
    QObject::connect(log_win_ui->copyLog,&QPushButton::released,this,&nApp::copyLog);
    QObject::connect(log_win_ui->saveLog,&QPushButton::released,this,&nApp::saveLog);
    QObject::connect(log_win_ui->buttonFind,&QPushButton::released,this,&nApp::findLogText);
    QObject::connect(log_win_ui->lineFind,&QLineEdit::returnPressed,this,&nApp::findLogText);

//    QSettings my_set("neutrino","");
//    my_set.beginGroup("nPreferences");

    log_win_ui->levelLog->setCurrentIndex(my_set.value("log_level",0).toInt());
    log_win_ui->followLog->setChecked(my_set.value("log_follow",1).toInt());
    log_win.setVisible(my_set.value("log_winVisible",false).toBool());
    setProperty("NeuSave-fileTxt",my_set.value("NeuSave-fileTxt","log.txt").toString());
    log_win_ui->logger->setWordWrapMode(QTextOption::WrapAnywhere);

    my_set.endGroup();

    QStringList args=arguments();
    args.removeFirst();

    neutrino *ny_neu = new neutrino();
    for (auto &arg : args)
        ny_neu->fileOpen(arg);


    return QApplication::exec();
}

void nApp::findLogText() {
    QString searchString = log_win_ui->lineFind->text();
    if (!searchString.isEmpty()) {
       bool found = log_win_ui->logger->find(searchString, QTextDocument::FindWholeWords);
       if (!found) {
           QTextCursor cursor(log_win_ui->logger->textCursor());
           cursor.movePosition(QTextCursor::Start);
           log_win_ui->logger->setTextCursor(cursor);
           found = log_win_ui->logger->find(searchString, QTextDocument::FindWholeWords);
       }
       if (found) {
           log_win_ui->logger->ensureCursorVisible();
       }
    }
}

void nApp::myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
    nApp *napp(qobject_cast<nApp*> (qApp));
    if (napp /*&& napp->log_win.isVisible()*/) {
        if (napp->log_win_ui->levelLog->currentIndex() > type) return;
        QByteArray localMsg = msg.toLocal8Bit();
        QString outstr;
        switch (type) {
            case QtDebugMsg:
                outstr += "D: <font color=\"#A9A9A9\">" + QString("Debug (") + context.file + QString(":")+ QLocale().toString(context.line) +QString(") ") + " :</font><font color=\"black\">";
                break;
            case QtInfoMsg:
                outstr+="I: <font color=\"black\">";
                break;
            case QtWarningMsg:
                outstr+="W: <font color=\"#C71585\">";
                break;
            case QtCriticalMsg:
                outstr+="C: <font color=\"#9932CC\">";
                break;
            case QtFatalMsg:
                outstr+="F: <font color=\"#FF0000\">";
                abort();
        }
        outstr +=  msg + "</font>";

        napp->log_win_ui->logger->append(outstr);
        if (napp->log_win_ui->followLog->isChecked())
            napp->log_win_ui->logger->verticalScrollBar()->setValue(napp->log_win_ui->logger->verticalScrollBar()->maximum());

    }
}

void nApp::copyLog() {
    QApplication::clipboard()->setText(log_win_ui->logger->toPlainText());
}

void nApp::saveLog(){
    QString fnametmp=QFileDialog::getSaveFileName(&log_win,tr("Save Log"),property("NeuSave-fileTxt").toString(),tr("Text files (*.txt *.csv);;Any files (*)"));
    if (!fnametmp.isEmpty()) {
        setProperty("NeuSave-fileTxt",fnametmp);
        QFile t(fnametmp);
        t.open(QIODevice::WriteOnly| QIODevice::Text);
        QTextStream out(&t);
        out << log_win_ui->logger->toPlainText();
        t.close();
    }

}
void nApp::addDefaultPalettes() {
    qDebug() << "reset Palettes";
    nPalettes.clear();
    QDirIterator it(":cmaps/", QDirIterator::Subdirectories);
    while (it.hasNext()) {
        QString pal=it.next();
        addPaletteFile(pal);
    }

    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList userPalettes=my_set.value("userPalettes","").toStringList();
    my_set.endGroup();
    for (auto &pal : userPalettes) {
        addPaletteFile(pal);
    }


    if (nPalettes.size()==0) {
        QMessageBox::warning(nullptr,tr("Attention"),tr("No colorscales present!"), QMessageBox::Ok);
    }
}

void nApp::addPaletteFile(QString cmapfile) {
    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList hiddenPalettes=my_set.value("hiddenPalettes","").toStringList();
    my_set.endGroup();
    if (hiddenPalettes.size()) {
        qInfo() << hiddenPalettes;
    }
    if (QFileInfo(cmapfile).exists() && (! hiddenPalettes.contains(cmapfile))) {
        QFile inputFile(cmapfile);
        if (inputFile.open(QIODevice::ReadOnly)) {
            QTextStream in(&inputFile);
            nPalettes[cmapfile]= std::vector<unsigned char>(256*3);
            unsigned int iter=0;
            while (!in.atEnd()) {
                QStringList line = in.readLine().split(" ",QString::SkipEmptyParts);
                for(auto &strnum : line) {
                    if (iter < nPalettes[cmapfile].size()) {
                        nPalettes[cmapfile][iter] = strnum.toInt();
                    }
                    iter++;
                }
            }            
            qInfo() << "Adding colormap" << cmapfile;
            inputFile.close();
        }
    }
}

void nApp::closeAllWindows() {
    qDebug() << "here" << sender();
    for(auto &neu : neus()) {
        neu->close();
    }
    processEvents();
    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    my_set.setValue("log_winVisible",log_win.isVisible());
    my_set.setValue("log_level",log_win_ui->levelLog->currentIndex());
    my_set.setValue("log_follow",log_win_ui->followLog->isChecked());
    my_set.setValue("NeuSave-fileTxt",property("NeuSave-fileTxt"));

    my_set.endGroup();
    //    QApplication::closeAllWindows();
};


void nApp::checkUpdates() {
    QNetworkProxyFactory::setUseSystemConfiguration(true) ;
    QNetworkAccessManager manager;
    QNetworkReply *response = manager.get(QNetworkRequest(QUrl("https://api.github.com/repos/NeutrinoToolkit/Neutrino/commits/master")));
    QEventLoop event;
    QObject::connect(response,&QNetworkReply::finished,&event,&QEventLoop::quit);
    event.exec();
    QString html = response->readAll(); // Source should be stored here

    QJsonDocument json = QJsonDocument::fromJson(html.toUtf8());

    QJsonObject responseObject = json.object();
#ifdef __phys_debug
    for(const auto &key : responseObject.keys() ) {
        QJsonValue value = responseObject.value(key);
        qDebug() << "Key = " << key << ", Value = " << value.toString();
    }
    QUrl commenturl(responseObject.value("comments_url").toString());
    response = manager.get(QNetworkRequest(commenturl));
    QObject::connect(response,&QNetworkReply::finished,&event,&QEventLoop::quit);
    event.exec();
    qDebug() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";
    qDebug() << response->readAll();
    qDebug() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";
#endif
    if (responseObject.contains("sha") && responseObject.value("sha").isString()) {
        QString compileSHA = QString(__VER_SHA);
        QString onlineSHA=responseObject.value("sha").toString();
        qDebug() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";
        qDebug() << compileSHA;
        qDebug() << onlineSHA;
        qDebug() << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<";

        if (compileSHA != onlineSHA) {

            QMessageBox msgBox;
            QString text=tr("A newer version is available");
            msgBox.setText(text);
            msgBox.addButton(tr("Go get it now"), QMessageBox::YesRole);
            msgBox.addButton(tr("Not now"), QMessageBox::RejectRole);
            msgBox.addButton(tr("Never"), QMessageBox::NoRole);
            //            msgBox.setIconPixmap(QPixmap(":icons/icon").scaledToWidth(100));
            int ret = msgBox.exec();

            qDebug() << ret << " : " << msgBox.result();
            switch ( msgBox.buttonRole(msgBox.clickedButton())) {
                case QMessageBox::YesRole:
                    QDesktopServices::openUrl(QUrl("https://github.com/NeutrinoToolkit/Neutrino/releases"));
                    break;
                case QMessageBox::NoRole: {
                        QSettings my_set("neutrino","");
                        my_set.beginGroup("nPreferences");
                        my_set.setValue("checkUpdates",false);
                        my_set.endGroup();
                    }
                    break;
                default:
                    break;
            }

        }

    }

}

QList<neutrino*> nApp::neus() {
    QList<neutrino*> retList;
    foreach (QWidget *widget, QApplication::topLevelWidgets()) {
        neutrino *my_neu=qobject_cast<neutrino *>(widget);
        if (my_neu) retList<< my_neu;
    }
    return retList;
}

bool nApp::notify(QObject *rec, QEvent *ev)
{
    bool retval=false;
    try {
        retval = QApplication::notify(rec, ev);
    } catch (std::exception &e) {
        log_win.show();
        qCritical() << e.what();
    }
    return retval;
}


bool nApp::event(QEvent *ev) {
    qDebug() << ev;
    if (ev->type() == QEvent::FileOpen) {
        QWidget *widget = QApplication::activeWindow();
        neutrino *neu=qobject_cast<neutrino *>(widget);
        if (neu == NULL) {
            nGenericPan *pan=qobject_cast<nGenericPan *>(widget);
            if (pan) neu = pan->nparent;
        }
        if (neu == NULL) neu = new neutrino();
        neu->fileOpen(static_cast<QFileOpenEvent *>(ev)->file());
        ev->accept();
    } else {
        return QApplication::event(ev);
    }
    return true;
}

QString nApp::localeToString(const QLocale &locale) {
    return QLocale::languageToString(locale.language())+ " " +QLocale::countryToString(locale.country())+ " " +QLocale::scriptToString(locale.script())+ " " +QString(locale.decimalPoint());
}

bool nApp::localeLessThan(const QLocale &loc1, const QLocale &loc2)
{
    return localeToString(loc1) < localeToString(loc2);
}


void nApp::changeLocale(QLocale locale) {
    QKeySequence key_seq=QKeySequence(Qt::CTRL);

    qDebug() << "here" << key_seq.toString();
    if (locale!=QLocale()) {

        qDebug() << QLocale::languageToString(locale.language()) <<
                    QLocale::scriptToString(locale.script()) <<
                    QLocale::countryToString(locale.country()) <<
                    locale.bcp47Name() <<
                    locale.country() <<
                    locale.name() <<
                    locale.decimalPoint();

        QLocale().setDefault(locale);
        QSettings settings("neutrino","");
        settings.beginGroup("nPreferences");
        settings.setValue("locale",locale);
        settings.endGroup();

        QString fileTransl(":translations/neutrino_"+locale.name()+".qm");
        if(QFileInfo(fileTransl).exists()) {
            QPointer<QTranslator> translator(new QTranslator(qApp));
            if (translator->load(fileTransl)) {
                qApp->installTranslator(translator);
                qDebug() << "installing translator" << fileTransl;
            } else {
                delete translator;
            }
        }
    }
}


void nApp::changeThreads(int num) {
    if (num<=1) {
        fftw_cleanup_threads();
    } else {
        fftw_init_threads();
        fftw_plan_with_nthreads(num);
    }
#ifdef HAVE_OPENMP
    omp_set_num_threads(num);
#endif
    DEBUG("THREADS " << num);
}


