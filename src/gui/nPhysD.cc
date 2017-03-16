#include "nPhysD.h"


void nPhysD::addParent(nPhysD* my_phys) {
    physParents.push_back(my_phys);
    connect(my_phys, SIGNAL(destroyed), this, SLOT(removeParent));
}

void nPhysD::addChildren(nPhysD* my_phys) {
    physChildren.push_back(my_phys);
    connect(my_phys, SIGNAL(destroyed), this, SLOT(removeChildren));
}

void nPhysD::removeParent(nPhysD* my_phys) {
    physParents.erase( std::remove( physParents.begin(), physParents.end(), my_phys ), physParents.end() );
}

void nPhysD::removeChildren(nPhysD* my_phys) {
    physChildren.erase( std::remove( physChildren.begin(), physChildren.end(), my_phys ), physChildren.end() );
}

const int nPhysD::childCount() {
    return physChildren.size();
}

const int nPhysD::parentCount() {
    return physParents.size();
}

nPhysD* nPhysD::parentN(unsigned int num) {
    return (num<physParents.size() ? physParents[num] : nullptr);
}

nPhysD* nPhysD::childN(unsigned int num) {
    return (num<physChildren.size() ? physChildren[num] : nullptr);
}

void nPhysD::TscanBrightness() {
    nPhysImageF<double>::TscanBrightness();
    emit physChanged(this);
}

double nPhysD::gamma() {
    if (!property.have("gamma")) {
        property["gamma"]=(int)1;
    }
    int gamma_int= property["gamma"].get_i();
    return gamma_int < 1 ? -1.0/(gamma_int-2) : gamma_int;
}


const unsigned char* nPhysD::to_uchar_palette(std::vector<unsigned char>  &palette, std::string palette_name) {
    bidimvec<double> minmax=property.have("display_range") ? property["display_range"] : get_min_max();
    double mini=minmax.first();
    double maxi=minmax.second();

    if (!property.have("gamma")) {
        property["gamma"]=(int)1;
    }
    double my_gamma=gamma();

    if (getSurf()>0 && palette.size()==768) {

        if (uchar_buf.size() == getSurf()*3 &&
                display_property.have("display_range") &&
                display_property.have("palette_name") &&
                display_property["palette_name"].get_str()==palette_name &&
                display_property.have("gamma") &&
                display_property["gamma"].get_i()==property["gamma"].get_i()) {

            vec2f old_display_range=display_property["display_range"];
            vec2f new_display_range=property["display_range"];

            if (old_display_range==new_display_range) {
                DEBUG("reusing old uchar_buf");
                return &uchar_buf[0];
            }
        }

        DEBUG(6,"8bit ["<<get_min()<<":"<<get_max() << "] from [" << mini << ":" << maxi<<"]");
        uchar_buf.resize(getSurf()*3);
#pragma omp parallel for
        for (size_t i=0; i<getSurf(); i++) {
            //int val = mult*(Timg_buffer[i]-lower_cut);
            if (std::isfinite(point(i))) {
                unsigned char val = std::max(0,std::min(255,(int) (255.0*pow((point(i)-mini)/(maxi-mini),my_gamma))));
                uchar_buf[i*3+0] = palette[3*val+0];
                uchar_buf[i*3+1] = palette[3*val+1];
                uchar_buf[i*3+2] = palette[3*val+2];
            } else {
                uchar_buf[i*3+0] = 255;
                uchar_buf[i*3+1] = 255;
                uchar_buf[i*3+2] = 255;
            }
        }
        display_property["palette_name"]=palette_name;
        display_property["gamma"]=property["gamma"].get_i();
        display_property["display_range"]=property["display_range"];

        return &uchar_buf[0];
    }
    WARNING("asking for uchar buffer of empty image");

    return NULL;
}
