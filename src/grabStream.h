#ifndef grabStream_H
#define grabStream_H

#include <streambuf>
#include <QDebug>

class grabStream : public QObject, public std::basic_streambuf<char> {
    Q_OBJECT

public:
    grabStream(std::ostream &stream) : my_stream(stream) {
        my_old_stream=stream.rdbuf() ;
        stream.rdbuf(this);
    }
    ~grabStream() {
        if (!my_string.empty()) {
            warn(my_string);
        }
        my_stream.rdbuf(my_old_stream);
    }

protected:
    virtual std::streambuf::int_type overflow(std::streambuf::int_type v) {
        if (v == '\n') {
            warn(my_string);
            my_string.clear();
        } else {
            my_string += v;
        }
        return v;
    }

    virtual std::streamsize xsputn(const char *p, std::streamsize n) {
        my_string.append(p, p + n);
        long pos = 0;
        while (pos != static_cast<long>(std::string::npos)) {
            pos = static_cast<long>(my_string.find('\n'));
            if (pos != static_cast<long>(std::string::npos)) {
                std::string tmp_string = my_string.substr(0,pos);
                warn(tmp_string);
                my_string.erase(my_string.begin(), my_string.begin() + pos + 1);
            }
        }
        return n;
    }

private:
    void warn(std::string& msg) {
        qWarning().noquote() << QString::fromStdString(msg);
    }
    std::ostream &my_stream;
    std::streambuf *my_old_stream;
    std::string my_string;
};

#endif
