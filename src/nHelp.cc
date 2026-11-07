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
#include "nHelp.h"
#include "nGenericPan.h"
#include "nPanPlug.h"
#include "ui_nPanHelp.h"
#include "neutrino.h"
#include <QMenu>

#include <QSplitter>
#include <QTreeWidget>
#include <QLineEdit>
#include <QVBoxLayout>
#include <QDirIterator>
#include <QTextDocumentFragment>
#include <QSettings>
#include <QCloseEvent>

QPointer<nHelp> nHelp::instance;
QPointer<neutrino> nHelp::window_shortcuts;

static const QString generalPage(":/NeutrinoREADME.html");
// made by the program, not a resource (it was the plugin Shortcuts)
static const QString shortcutsPage(":/help/keyboard_shortcuts.html");

// the shortcuts of all the actions of neutrino and of the plugins, by menu;
// a shortcut used by more than one action, or also by the keys of the objects, is shown in red
// (Qt runs none of two actions; a menu shortcut is run instead of the key of the selected object)
static QString menuShortcuts(neutrino *nparent) {
    QMap<QString, QList<QPair<QString,QString>>> groups; // menu -> (action, keys)
    QMap<QString, QStringList> byKey;                     // key -> actions
    foreach (QAction *action, nparent->findChildren<QAction*>()) {
        if (action->shortcut().isEmpty()) continue;
        // not the actions of the analysis windows (children of nparent): e.g. their gear menu, one per open window,
        // active only in its window (listed in the html)
        bool inPan=false;
        for (QObject *obj=action->parent(); obj && obj!=nparent; obj=obj->parent()) {
            if (qobject_cast<nGenericPan*>(obj)) { inPan=true; break; }
        }
        if (inPan) continue;
        // only the actions in a menu/toolbar: the others have no active shortcut
        if (action->associatedObjects().isEmpty()) continue;
        QString menuName=QObject::tr("Other");
        for (QObject *obj : action->associatedObjects()) {
            QMenu *menu=qobject_cast<QMenu*>(obj);
            if (menu) {
                QStringList path;
                for (QMenu *m=menu; m; m=qobject_cast<QMenu*>(m->parentWidget())) {
                    path.prepend(m->title().remove('&'));
                }
                menuName=path.join(" / ");
                break;
            }
        }
        QStringList keys;
        for (auto &seq : action->shortcuts()) {
            keys << seq.toString(QKeySequence::NativeText);
            byKey[seq.toString(QKeySequence::PortableText)] << action->text().remove('&');
        }
        groups[menuName] << qMakePair(action->text().remove('&'), keys.join(", "));
    }
    QString html;
    for (auto it=groups.begin(); it!=groups.end(); ++it) {
        auto list=it.value();
        std::sort(list.begin(), list.end());
        html+="<B>"+it.key().toHtmlEscaped()+"</B><BR>\n";
        for (auto &entry : list) {
            html+=entry.second.toHtmlEscaped()+" : "+entry.first.toHtmlEscaped()+"<BR>\n";
        }
        html+="<HR>\n";
    }
    // the keys of the selected objects (nLine, nObject, nPoint, keyPressEvent) and of the image (nView):
    // the menu shortcuts are run before them, an object never gets a key that is also a menu shortcut
    const QList<QPair<QString,QString>> objectKeys={
        {"Up", QObject::tr("move (selected object, mouse cursor)")}, {"Down", QObject::tr("move (selected object, mouse cursor)")},
        {"Left", QObject::tr("move (selected object, mouse cursor)")}, {"Right", QObject::tr("move (selected object, mouse cursor)")},
        {"Shift+Up", QObject::tr("move faster")}, {"Shift+Down", QObject::tr("move faster")},
        {"Shift+Left", QObject::tr("move faster")}, {"Shift+Right", QObject::tr("move faster")},
        {"Return", QObject::tr("stop adding points (line), click (image)")}, {"Esc", QObject::tr("stop adding points (line, rectangle)")},
        {"Backspace", QObject::tr("delete the selected object")},
        {"W", QObject::tr("properties window (line, rectangle, point)")}, {"B", QObject::tr("Bezier (line)")},
        {"C", QObject::tr("closed line, center the rectangle")}, {"Shift+C", QObject::tr("center the line, copy the rectangle points")},
        {"S", QObject::tr("antialias (line), submatrix (rectangle)")}, {"A", QObject::tr("append point (line)")},
        {"P", QObject::tr("prepend point (line)")}, {"D", QObject::tr("delete point (line)")},
        {"X", QObject::tr("horizontal (line), expand width (rectangle)")}, {"Y", QObject::tr("vertical (line), expand height (rectangle)")},
        {"R", QObject::tr("rectangle (line)")}, {"Shift+R", QObject::tr("expand to the image (line)")},
        {"E", QObject::tr("expand (rectangle)")}, {"I", QObject::tr("intersection with the image (rectangle)")},
        {"Shift+V", QObject::tr("paste the points (rectangle)")}};
    for (auto &key : objectKeys) {
        if (byKey.contains(key.first)) byKey[key.first] << QObject::tr("key of the objects: ")+key.second;
    }
    QStringList conflicts;
    for (auto it=byKey.begin(); it!=byKey.end(); ++it) {
        if (it.value().size()>1) {
            conflicts << QKeySequence(it.key()).toString(QKeySequence::NativeText).toHtmlEscaped()+" : "+it.value().join(", ").toHtmlEscaped();
        }
    }
    if (!conflicts.isEmpty()) {
        html.prepend("<FONT COLOR=\"red\"><B>"+QObject::tr("Shortcuts used more than once (only the menu one works, two menu ones: none):")+"</B><BR>\n"+conflicts.join("<BR>\n")+"</FONT><HR>\n");
    }
    return html;
}

// the keyboard shortcuts of a window (its menus and the plugins), from resources/help/Shortcuts.html
static QString shortcutsHtml(neutrino *nparent) {
    if (!nparent) return QString();
    QFile lic(":/help/Shortcuts.html");
    if (lic.open(QFile::ReadOnly | QFile::Text)) {
        // QString htmltext=QTextStream(&lic).readAll().replace("%CTRL%",QKeySequence(Qt::CTRL).toString(QKeySequence::NativeText)).replace("%OPTION%",QKeySequence(Qt::META).toString(QKeySequence::NativeText)).replace("%COMMAND%",QKeySequence(Qt::ALT).toString(QKeySequence::NativeText));
        // the menu shortcuts from the actions themselves (the fixed list was out of date)
        // QString htmltext=QTextStream(&lic).readAll().replace("%MENUS%",menuShortcuts(nparent)).replace("%CTRL%",QKeySequence(Qt::CTRL).toString(QKeySequence::NativeText)).replace("%OPTION%",QKeySequence(Qt::META).toString(QKeySequence::NativeText)).replace("%COMMAND%",QKeySequence(Qt::ALT).toString(QKeySequence::NativeText));
        // the keys of the gear menu of the pans (nGenericPan) as Qt writes them (the modifiers in the order of the
        // platform), and Alt for the reset (on macOS %OPTION% was Qt::META, i.e. Control)
        auto panKey=[](Qt::Key key) {
            return QKeySequence(Qt::CTRL | Qt::ALT | Qt::META | key).toString(QKeySequence::NativeText).toHtmlEscaped();
        };
        QString alt=QKeySequence(Qt::ALT | Qt::Key_A).toString(QKeySequence::NativeText);
        alt.chop(1); // (the modifier alone)
        if (alt.endsWith('+')) alt.chop(1);
        QString htmltext=QTextStream(&lic).readAll().replace("%MENUS%",menuShortcuts(nparent)).replace("%PAN_S%",panKey(Qt::Key_S)).replace("%PAN_J%",panKey(Qt::Key_J)).replace("%PAN_N%",panKey(Qt::Key_N)).replace("%ALT%",alt.toHtmlEscaped()).replace("%HELP%",QKeySequence(QKeySequence::HelpContents).toString(QKeySequence::NativeText).toHtmlEscaped());
        return htmltext;
    }
    return QString();
}

QString nHelp::pageFile(const QString &page) {
    if (page=="Shortcuts") return shortcutsPage;
    QString fname= page.isEmpty() ? generalPage : (page.startsWith(":/") ? page : ":/"+page+"README.html");
    return QFileInfo::exists(fname) ? fname : QString();
}

nHelp* nHelp::showPage(const QString &page, neutrino *window) {
    if (window) window_shortcuts=window;
    if (instance.isNull()) instance=new nHelp();
    QString fname=pageFile(page);
    if (fname.isEmpty()) fname=generalPage;
    instance->browser->setSource(QUrl("qrc"+fname));
    instance->setWindowState(instance->windowState() & ~Qt::WindowMinimized);
    instance->show();
    instance->raise();          // for MacOS
    instance->activateWindow(); // for Windows
    return instance;
}

nHelp::nHelp() : QMainWindow(nullptr) {
    setAttribute(Qt::WA_DeleteOnClose);
    Ui::PanHelp my_help; // the toolbar and the browser of the help of the pans
    my_help.setupUi(this);
    setWindowTitle(tr("Neutrino help"));
    browser=my_help.help;

    QWidget *left=new QWidget(this);
    QVBoxLayout *leftLayout=new QVBoxLayout(left);
    leftLayout->setContentsMargins(0,0,0,0);
    leftLayout->setSpacing(2);
    search=new QLineEdit(left);
    search->setPlaceholderText(tr("Search"));
    search->setClearButtonEnabled(true);
    search->setToolTip(tr("Shows only the pages containing this text"));
    tree=new QTreeWidget(left);
    tree->setHeaderHidden(true);
    tree->setRootIsDecorated(true);
    leftLayout->addWidget(search);
    leftLayout->addWidget(tree);

    QSplitter *splitter=new QSplitter(this);
    splitter->addWidget(left);
    splitter->addWidget(browser); // (out of the central widget of the ui, replaced below)
    splitter->setStretchFactor(0,0);
    splitter->setStretchFactor(1,1);
    splitter->setSizes({180,620});
    setCentralWidget(splitter);

    connect(my_help.actionHome, &QAction::triggered, this, [this]() { browser->setSource(QUrl("qrc"+generalPage)); });
    connect(my_help.actionBack, &QAction::triggered, browser, &nHelpTextBrowser::backward);
    connect(my_help.actionForward, &QAction::triggered, browser, &nHelpTextBrowser::forward);
    connect(my_help.actionPrint, &QAction::triggered, browser, &nHelpTextBrowser::print);
    connect(my_help.actionIncreaseFont, &QAction::triggered, browser, [this]() { browser->zoomIn(); });
    connect(my_help.actionDecreaseFont, &QAction::triggered, browser, [this]() { browser->zoomOut(); });

    connect(browser, &QTextBrowser::sourceChanged, this, &nHelp::sourceChanged);
    connect(tree, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem *item) { itemChosen(item); });
    connect(tree, &QTreeWidget::itemClicked, this, [this](QTreeWidgetItem *item) { itemChosen(item); });
    connect(search, &QLineEdit::textChanged, this, &nHelp::filter);

    browser->generated=[](const QUrl &url) {
        return url==QUrl("qrc"+shortcutsPage) ? shortcutsHtml(shortcutsWindow()) : QString();
    };

    fillPages();

    resize(820,640);
    QSettings my_set("neutrino","");
    if (my_set.contains("helpGeometry")) restoreGeometry(my_set.value("helpGeometry").toByteArray());
}

void nHelp::closeEvent(QCloseEvent *e) {
    QSettings("neutrino","").setValue("helpGeometry",saveGeometry());
    QMainWindow::closeEvent(e);
}

// the general page, then the plugins with a help under their menu (e.g. Analysis > Lineout), then the helps of the
// resources that are not of a plugin (e.g. a directory with several plugins)
void nHelp::fillPages() {
    auto addPage=[this](QTreeWidgetItem *item, const QString &title, const QString &fname) {
        item->setText(0,title);
        item->setData(0,Qt::UserRole,fname);
        item->setToolTip(0,title);
        items[fname]=item;
        QFile f(fname);
        if (fname==shortcutsPage) texts[fname]=QTextDocumentFragment::fromHtml(shortcutsHtml(shortcutsWindow())).toPlainText();
        else if (f.open(QIODevice::ReadOnly)) texts[fname]=QTextDocumentFragment::fromHtml(QString::fromUtf8(f.readAll())).toPlainText();
    };

    if (QFileInfo::exists(generalPage)) addPage(new QTreeWidgetItem(tree), "Neutrino", generalPage);
    addPage(new QTreeWidgetItem(tree), tr("Keyboard shortcuts"), shortcutsPage);

    QMap<QString, QTreeWidgetItem*> menus;
    auto menuItem=[this, &menus](QString entry) {
        if (entry.isEmpty()) entry="Analysis";
        QTreeWidgetItem *parent=nullptr;
        QString path;
        for (const QString &name : entry.split(";")) {
            path+=";"+name;
            if (!menus.contains(path)) {
                QTreeWidgetItem *item = parent ? new QTreeWidgetItem(parent) : new QTreeWidgetItem(tree);
                item->setText(0,name);
                item->setFlags(Qt::ItemIsEnabled);
                QFont font=item->font(0);
                font.setBold(true);
                item->setFont(0,font);
                item->setExpanded(true);
                menus[path]=item;
            }
            parent=menus[path];
        }
        return parent;
    };

    for (auto &plug : nPanPlug::all()) { // sorted by name
        QString fname=pageFile(plug.name);
        if (!fname.isEmpty()) addPage(new QTreeWidgetItem(menuItem(plug.menuEntry)), QString(plug.name).replace("_"," "), fname);
    }

    QDirIterator it(":/", {"*README.html"}, QDir::Files);
    QStringList others;
    while (it.hasNext()) {
        QString fname=it.next();
        if (!items.contains(fname)) others << fname;
    }
    others.sort();
    for (auto &fname : others) {
        QString title=QFileInfo(fname).fileName().chopped(QString("README.html").size()).replace("_"," ");
        addPage(new QTreeWidgetItem(menuItem(tr("Other"))), title, fname);
    }
    // the groups expanded once they have their children
    for (auto &item : menus) item->setExpanded(true);
}

// shows only the pages with the text in their title or their content (and the groups with such pages)
void nHelp::filter(const QString &text) {
    const QString t=text.trimmed();
    std::function<bool(QTreeWidgetItem*)> apply=[&](QTreeWidgetItem *item) {
        bool visible=false;
        QString fname=item->data(0,Qt::UserRole).toString();
        if (!fname.isEmpty()) {
            visible = t.isEmpty() || item->text(0).contains(t,Qt::CaseInsensitive) || texts.value(fname).contains(t,Qt::CaseInsensitive);
        }
        for (int i=0; i<item->childCount(); i++) visible = apply(item->child(i)) || visible;
        item->setHidden(!visible);
        return visible;
    };
    for (int i=0; i<tree->topLevelItemCount(); i++) apply(tree->topLevelItem(i));
    if (!t.isEmpty()) {
        browser->moveCursor(QTextCursor::Start);
        browser->find(t);
    }
}

void nHelp::itemChosen(QTreeWidgetItem *item) {
    if (!item) return;
    QString fname=item->data(0,Qt::UserRole).toString();
    if (fname.isEmpty()) return;
    if (browser->source()!=QUrl("qrc"+fname)) browser->setSource(QUrl("qrc"+fname));
    const QString t=search->text().trimmed();
    if (!t.isEmpty()) { // the first occurrence of the search
        browser->moveCursor(QTextCursor::Start);
        browser->find(t);
    }
}

// the page in the list follows the browser (links, back, forward)
void nHelp::sourceChanged(const QUrl &url) {
    QString fname=url.toString(QUrl::RemoveFragment|QUrl::RemoveQuery);
    if (fname.startsWith("qrc:")) fname=fname.mid(3);
    QTreeWidgetItem *item=items.value(fname);
    if (item && tree->currentItem()!=item) {
        QSignalBlocker block(tree);
        tree->setCurrentItem(item);
    }
    if (item) setWindowTitle(tr("Neutrino help")+" - "+item->text(0));
}

// the window whose shortcuts are shown: the last one that opened the help, or any Neutrino window
neutrino* nHelp::shortcutsWindow() {
    if (window_shortcuts) return window_shortcuts;
    for (QWidget *w : QApplication::topLevelWidgets()) {
        if (neutrino *neu=qobject_cast<neutrino*>(w)) return neu;
    }
    return nullptr;
}
