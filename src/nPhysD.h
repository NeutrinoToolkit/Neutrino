#ifndef __nPhysD
#define __nPhysD

#include <QDebug>
#include "nPhysImageF.h"

class nPhysD : public physD {

  public:

    using physD::nPhysImageF;

    nPhysD(physD in): physD(in) {}

    ~nPhysD(){
        qDebug() << "------------------------------------------------------------ DELETE "<< copies();
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
        return *this;
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
                vec2f new_display_range=prop["display_range"];

                if (old_display_range==new_display_range) {
                    DEBUG("reusing old uchar_buf");
                    return nullptr;
                }
            }

            vec2f minmax=prop.have("display_range") ? prop["display_range"] : get_min_max();
            double mini=minmax.first();
            double maxi=minmax.second();
            double my_gamma=gamma();

            uchar_buf.assign(getSurf()*3,255);
#pragma omp parallel for
            for (size_t i=0; i<getSurf(); i++) {
                //int val = mult*(Timg_buffer[i]-lower_cut);
                if (std::isfinite(Timg_buffer[i])) {
                    unsigned char val = static_cast<unsigned char>(std::max(0,std::min(255,static_cast<int>(255.0*pow((Timg_buffer[i]-mini)/(maxi-mini),my_gamma)))));
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

//    nPhysD & operator= (const nPhysD &rhs)
//    {
//        physD::operator=(rhs);
//        display_property=rhs.display_property;
//        return *this;
//    }

//    nPhysD & operator= (const physD &rhs)
//    {
//        physD::operator=(rhs);
//        return *this;

//    }

    phys_properties display_prop;

    inline void reset_display() {
        display_prop.clear();
        uchar_buf.clear();
        DEBUG(uchar_buf.capacity());
    }

    void TscanBrightness() {
        reset_display();
        physD::TscanBrightness();
    }
    std::vector<unsigned char> uchar_buf;

};


#endif
