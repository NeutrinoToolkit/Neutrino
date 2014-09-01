#ifndef nScene_h
 #define nScene_h

 #include <QGraphicsScene>

 class neutrino;

 class nScene : public QGraphicsScene
 {
     Q_OBJECT

 public:
     nScene(neutrino* const);
 };

 #endif