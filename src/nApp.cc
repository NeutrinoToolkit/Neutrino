#include "nApp.h"
#include "neutrino.h"

#ifdef __neutrino_key
#include "nHash.h"
#endif

#ifdef HAVE_PYTHONQT
#include "PythonQt_QtBindings.h"
#include "nPhysPyWrapper.h"
#include "nPython.h"
#endif


#ifdef HAVE_LIBTIFF
#define int32 tiff_int32
#define uint32 tiff_uint32
extern "C" {
#include <tiffio.h>
}
#undef int32
#undef uint32

static void s_TiffWarningHandler(const char* module, const char* fmt, va_list args)
{
    QString msg;
    msg += QString(module);
    qWarning() << msg ;
}

#endif


#ifdef USE_QT5
#include <QtDebug>
#include <codeeditor.h>

QPointer<QMainWindow> logWin;
QPointer<CodeEditor> logText;

void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    QString my_msg;
    switch (type) {
    case QtDebugMsg:
        my_msg="Debug";
        break;
    case QtInfoMsg:
        my_msg="Info";
        break;
    case QtWarningMsg:
        my_msg="Warning";
        break;
    case QtCriticalMsg:
        my_msg="Critical";
        break;
    case QtFatalMsg:
        my_msg="Fatal";
        break;
    }
    my_msg+=" "+msg;
#ifdef  __phys_debug
    my_msg+=" in " + QString(context.file) +
            ":" + QString::number(context.line) +
            " <" + QString(context.function)+">";
#endif

    if (logWin.isNull()) {
#ifdef USE_QT5
        QFont my_font = QFontDatabase::systemFont(QFontDatabase::FixedFont);
#else
        QFont my_font;
        my_font.setStyleHint(QFont::TypeWriter);
#endif
        my_font.setPointSize(10);

        logWin=new QMainWindow();
        logWin->setWindowTitle("Neutrino log");
        logText=new CodeEditor(logWin);
        logText->setFont(my_font);
        logWin->setCentralWidget(logText);
        logText->setReadOnly(true);
    }

    logText->moveCursor(QTextCursor::End);
    logText->insertPlainText(my_msg+"\n");
}
#endif




NApplication::NApplication( int &argc, char **argv ) :
    QApplication(argc, argv) {

    addLibraryPath(applicationDirPath()+QString("/plugins"));

#ifdef __neutrino_key
    std::string hh = getNHash();
    std::cerr<<"got nHash: "<<hh<<std::endl;
    setProperty("nHash", hh.c_str());
#endif

#ifdef USE_QT5
    qInstallMessageHandler(myMessageOutput);
    setAttribute(Qt::AA_UseHighDpiPixmaps);
#endif

#ifdef HAVE_LIBTIFF
    TIFFSetWarningHandler(&s_TiffWarningHandler);
#endif


    setOrganizationName("ParisTech");
    setOrganizationDomain("edu");
    setApplicationName("Neutrino");
    setApplicationVersion(__VER);

    QStringList args=QCoreApplication::arguments();
    args.removeFirst();

#ifdef HAVE_PYTHONQT

    PythonQt::init(PythonQt::IgnoreSiteModule|PythonQt::RedirectStdOut);

    //PythonQt_QtAll::init();
    PythonQt_init_QtBindings();

    PythonQt::self()->addDecorators(new nPhysPyWrapper());
    PythonQt::self()->registerCPPClass("nPhysD",NULL,"neutrino");

    PythonQt::self()->addDecorators(new nPanPyWrapper());
    PythonQt::self()->registerClass(& nGenericPan::staticMetaObject, "nPan", PythonQtCreateObject<nPanPyWrapper>);

    PythonQt::self()->addDecorators(new nPyWrapper());
    PythonQt::self()->registerClass(& neutrino::staticMetaObject, "neutrino", PythonQtCreateObject<nPyWrapper>);

    QSettings settings("neutrino","");
    settings.beginGroup("Python");
    foreach (QString spath, settings.value("siteFolder").toString().split(QRegExp("\\s*:\\s*"))) {
        if (QFileInfo(spath).isDir()) PythonQt::self()->addSysPath(spath);
    }
    PythonQt::self()->getMainModule().evalScript(settings.value("initScript").toString());
    settings.endGroup();

    PythonQt::self()->getMainModule().addObject("nApp", this);
    foreach (QString filename, args) {
        QFileInfo my_file(filename);
        if (my_file.exists() && my_file.suffix()=="py") {
            QFile t(filename);
            t.open(QIODevice::ReadOnly| QIODevice::Text);
            PythonQt::self()->getMainModule().evalScript(QTextStream(&t).readAll());
            t.close();
        }
    }
#endif

    if (neus().size()==0) {
        neutrino *neu = new neutrino();
        neu->fileOpen(args);
    }

}



QList<neutrino*> NApplication::neus() {
    QList<neutrino*> retList;
    foreach (QWidget *widget, QApplication::topLevelWidgets()) {
        neutrino *my_neu=qobject_cast<neutrino *>(widget);
        if (my_neu) retList<< my_neu;
    }
    return retList;
}

bool NApplication::notify(QObject *rec, QEvent *ev)
{
    try {
        return QApplication::notify(rec, ev);
    }
    catch (std::exception &e) {
        qCritical() << e.what();
//        QMessageBox dlg(QMessageBox::Critical, tr("Exception"), e.what());
//        dlg.setWindowFlags(dlg.windowFlags() | Qt::WindowStaysOnTopHint);
//        dlg.exec();
    }

    return false;
}

#ifdef USE_QT5
void NApplication::toggleLog() {
    logWin->setWindowState(logWin->windowState() & (~Qt::WindowMinimized | Qt::WindowActive));
    logWin->raise();  // for MacOS
    logWin->activateWindow(); // for Windows
    logWin->show();
}
#endif

bool NApplication::event(QEvent *ev) {
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

