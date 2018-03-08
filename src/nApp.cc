#include "nApp.h"
#include "neutrino.h"
#include <QtNetwork>

nApp::nApp( int &argc, char **argv ) : QApplication(argc, argv) {

    qInstallMessageHandler(nApp::myMessageOutput);

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

    QSettings my_set("neutrino","");
    my_set.beginGroup("nPreferences");
    changeLocale(my_set.value("locale",QLocale()).toLocale());
    changeThreads(my_set.value("threads",1).toInt());
    if (!my_set.contains("checkUpdates")) {
        my_set.setValue("checkUpdates",QMessageBox::Yes == QMessageBox::question(nullptr,tr("Attention"),tr("Do you want to check for updates?"), QMessageBox::Yes | QMessageBox::No));
    }
    if (my_set.value("checkUpdates",true).toBool()) {
        checkUpdates();
    }

    log_win.setCentralWidget(&logger);
    log_win.setWindowTitle("Log");
    log_win.setWindowIcon(QIcon(":icons/icon.png"));
    logger.setReadOnly(true);
    logger.setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
    logger.setLineWrapMode(QPlainTextEdit::NoWrap);

    log_win.setVisible(my_set.value("log_winVisible",false).toBool());

    addDefaultPalettes();

    my_set.endGroup();
}

void nApp::myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg) {
    nApp *napp(qobject_cast<nApp*> (qApp));
    if (napp && napp->log_win.isVisible()) {
        QByteArray localMsg = msg.toLocal8Bit();
        QString outstr(QDateTime::currentDateTime().toString());
        switch (type) {
            case QtDebugMsg:
                outstr += " Debug";
                break;
            case QtInfoMsg:
                outstr += " Info";
                break;
            case QtWarningMsg:
                outstr += " Warn";
                break;
            case QtCriticalMsg:
                outstr += " Critical";
                break;
            case QtFatalMsg:
                outstr += " Fatal";
                abort();
        }
        outstr+= QString(" (") + context.file + QString(":")+ QString::number(context.line) +QString(") ") + context.function + " : " + msg;

#ifdef __phys_debug
        std::cerr << "* " << outstr.toStdString();
#endif
        napp->logger.appendPlainText(outstr);
    }


}

void nApp::addDefaultPalettes() {
    qDebug() << "reset Palettes";
    nPalettes.clear();

    QSettings my_set("neutrino","");
    my_set.beginGroup("Palettes");
    QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();
    paletteFiles.removeDuplicates();
    for(auto &my_str : paletteFiles) {
        addPaletteFile(my_str);
    }
    my_set.setValue("paletteFiles",paletteFiles);
    if (nPalettes.size()==0) {
        QDirIterator it(":cmaps/", QDirIterator::Subdirectories);
        while (it.hasNext()) {
            addPaletteFile(it.next());
        }
    }
    if (nPalettes.size()==0) {
        QMessageBox::warning(nullptr,tr("Attention"),tr("No colorscales present!"), QMessageBox::Ok);
    }
}

void nApp::addPaletteFile(QString cmapfile) {
    if (QFileInfo(cmapfile).exists()) {
        QFile inputFile(cmapfile);
        if (inputFile.open(QIODevice::ReadOnly)) {
            QTextStream in(&inputFile);
            nPalettes[cmapfile]= std::vector<unsigned char>(256*3);
            unsigned int iter=0;
            while (!in.atEnd()) {
                QStringList line = in.readLine().split(" ",QString::SkipEmptyParts);
                for(auto &strnum : line) {
                    nPalettes[cmapfile].at(iter) = strnum.toInt();
                    iter++;
                }
            }
            QSettings my_set("neutrino","");
            my_set.beginGroup("Palettes");
            QStringList paletteFiles=my_set.value("paletteFiles","").toStringList();
            paletteFiles << cmapfile;
            paletteFiles.removeDuplicates();
            paletteFiles.sort(Qt::CaseInsensitive);
            my_set.setValue("paletteFiles",paletteFiles);
            my_set.endGroup();

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
    my_set.endGroup();

    log_win.close();
    //    QApplication::closeAllWindows();
};


void nApp::checkUpdates() {
    QNetworkProxyFactory::setUseSystemConfiguration(true) ;
    QNetworkAccessManager manager;
    QNetworkReply *response = manager.get(QNetworkRequest(QUrl("https://api.github.com/repos/NeutrinoToolkit/Neutrino/commits/master")));
    QEventLoop event;
    connect(response,SIGNAL(finished()),&event,SLOT(quit()));
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
    connect(response,SIGNAL(finished()),&event,SLOT(quit()));
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
            QString text=tr("A newer version is available available");
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
    try {
        return QApplication::notify(rec, ev);
    } catch (std::exception &e) {
        QMessageBox dlg(QMessageBox::Critical, tr("Exception"), e.what());
        dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
        dlg.exec();
    }

    return false;
}


bool nApp::event(QEvent *ev) {
    //    qDebug() << ev;
    if (ev->type() == QEvent::FileOpen) {
        QWidget *widget = QApplication::activeWindow();
        neutrino *neu=qobject_cast<neutrino *>(widget);
        if (neu == NULL) {
            nGenericPan *pan=qobject_cast<nGenericPan *>(widget);
            if (pan) neu = pan->nparent;
        }
        if (neu == NULL) neu = new neutrino();
        neu->fileOpen(static_cast<QFileOpenEvent *>(ev)->file());
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


