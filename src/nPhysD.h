#ifndef nPhysD_H
#define nPhysD_H

#include <QDebug>
#include <QRectF>
#include "nPhysImageF.h"

class nPhysD : public physD {

public:

    using physD::physD;

//    nPhysD(physD in=physD()): physD(in) {
    nPhysD(physD in=physD()): physD(std::move(in)) {
        qDebug() << "------------------------------------------------------------ CREATE "<< copies() << " " << this << QString::fromStdString(this->getName());
    }

    ~nPhysD(){
        qDebug() << "------------------------------------------------------------ DELETE "<< copies() << " " << this << QString::fromStdString(this->getName());
    }

    nPhysD & operator= (const nPhysD &rhs)
    {
        physD *tmpimg=static_cast<physD*>(this);
        *tmpimg=static_cast<const physD &>(rhs);
        display_prop = rhs.display_prop;
        uchar_buf = rhs.uchar_buf;
        return *this;
    }

    nPhysD & operator= (const physD &rhs)
    {
        physD *tmpimg=static_cast<physD*>(this);
        *tmpimg=rhs;
        reset_display(); // new pixels: the cached uchar_buf is stale
        return *this;
    }

    // move versions: the user-declared copy operator= above suppresses the implicit ones,
    // so without these a = temporary would deep copy the pixels
    nPhysD(const nPhysD &) = default;
    nPhysD(nPhysD &&) noexcept = default;

    nPhysD & operator= (nPhysD &&rhs) noexcept
    {
        physD::operator=(std::move(static_cast<physD &>(rhs)));
        display_prop = std::move(rhs.display_prop);
        uchar_buf = std::move(rhs.uchar_buf);
        return *this;
    }

    nPhysD & operator= (physD &&rhs)
    {
        physD::operator=(std::move(rhs));
        reset_display();
        return *this;
    }

    // display range without side effects: prop["display_range"] would insert an empty entry if missing
    vec2f displayRange() {
        if (prop.have("display_range") && prop["display_range"].is_vec()) return prop["display_range"];
        return get_min_max();
    }

    const unsigned char *to_uchar_palette(std::vector<unsigned char>  &palette, std::string palette_name) {
        if (getSurf()>0 && palette.size()==768) {
            if (uchar_buf.size() == getSurf()*3 &&
                    display_prop.have("display_range") &&
                    display_prop.have("palette_name") &&
                    display_prop.have("gamma") &&
                    display_prop["palette_name"].get_str()==palette_name &&
                    display_prop["gamma"].get_i()==prop["gamma"].get_i()) {

                vec2f old_display_range=display_prop["display_range"];
                vec2f new_display_range=displayRange();

                if (old_display_range==new_display_range) {
                    DEBUG("reusing old uchar_buf");
                    return &uchar_buf[0];
                }
            }

            vec2f minmax=displayRange();
            double mini=minmax.first();
            double maxi=minmax.second();
            double my_gamma=gamma();

            uchar_buf.assign(getSurf()*3,255);
#pragma omp parallel for
            for (size_t i=0; i<getSurf(); i++) {
                //int val = mult*(Timg_buffer[i]-lower_cut);
                if (std::isfinite(Timg_buffer[i])) {
                    // fraction of the display range, clamped first: pow of a negative number is NaN
                    // (inverted range) and min==max would divide by zero
                    double frac = (maxi!=mini) ? (Timg_buffer[i]-mini)/(maxi-mini)
                                               : (Timg_buffer[i]>mini ? 1.0 : (Timg_buffer[i]<mini ? 0.0 : 0.5));
                    frac = std::min(1.0,std::max(0.0,frac));
                    unsigned char val = static_cast<unsigned char>(std::max(0,std::min(255,static_cast<int>(255.0*pow(frac,my_gamma)))));
                    std::copy ( palette.begin()+val*3, palette.begin()+val*3+3, uchar_buf.begin()+3*i);
                }
            }
            display_prop["palette_name"]=palette_name;
            display_prop["gamma"]=prop["gamma"].get_i();
            display_prop["display_range"]=prop["display_range"]=vec2f(mini,maxi);

            return &uchar_buf[0];
        }
        return nullptr;
    }

    phys_properties display_prop;
    std::vector<unsigned char> uchar_buf;

    inline void reset_display() {
        display_prop.clear();
        uchar_buf.clear();
        DEBUG(uchar_buf.capacity());
    }

    void TscanBrightness() override {
        reset_display();
        physD::TscanBrightness();
    }

    nPhysD sub(QRectF geom2, unsigned int pad=0) {
        return physD::sub(geom2.left(),geom2.top(),geom2.width(),geom2.height(), pad);
    }
};

//Q_DECLARE_METATYPE(nPhysD*);

#endif
