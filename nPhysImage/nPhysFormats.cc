/*
 *
 *	Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 *
 *	This file is part of nPhysImage library.
 *
 *	nPhysImage is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU Lesser General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	nPhysImage is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU Lesser General Public License for more details.
 *
 *	You should have received a copy of the GNU Lesser General Public License
 *	along with neutrino.  If not, see <http://www.gnu.org/licenses/>.
 *
 *	Contact Information:
 *	Alessandro Flacco <alessandro.flacco@polytechnique.edu>
 *	Tommaso Vinci <tommaso.vinci@polytechnique.edu>
 *
 */
#include "nPhysFormats.h"
#include "NaNStream.h"
#include "bidimvec.h"
#include "nPhysMaths.h"
#include <time.h>
#include <zlib.h>
#include <regex>

#ifdef HAVE_LIBCFITSIO
#include "fitsio.h"
#endif

#define CHUNK 0x4000
#define windowBits 15
#define ENABLE_ZLIB_GZIP 32
#define GZIP_ENCODING 16

physFormat::physDouble_txt::physDouble_txt(const char *ifilename)
    : physD(std::string(ifilename), PHYS_FILE)
{
    std::ifstream ifile(ifilename);
    // 1. get image statistics
    std::string tline;
    int nlines = 0, ncols = 0;

    while (getline (ifile, tline)) {
        if ((tline.find('#') == std::string::npos) && (tline.find_first_not_of(' ') != tline.find_last_not_of(' ')))
            nlines++;
    }

    ifile.clear();
    ifile.seekg(0,std::ios::beg);
    std::string comment("");
    do {
        getline (ifile, tline);
        comment.append(tline);
    } while ((tline.find('#') != std::string::npos) || (tline.find_first_not_of(' ') == tline.find_last_not_of(' ')));

    std::stringstream ss(tline);
    std::string word;
    while( ss >> word ) ++ncols;

    DEBUG(5,"file has "<<nlines<<" lines and "<<ncols<<" cols");

    //physImage mycc(ncols, nlines);
    resize(ncols, nlines);

    // 2. read and save five
    ifile.clear();
    ifile.seekg(0, std::ios::beg);

    int row = 0, col = 0;
    int w = getW();
    while (getline(ifile, tline) && row<nlines) {
        col=0;
        std::stringstream sline(tline);
        while( sline >> word && col<ncols) {
            set(row*w+col, strtod(word.c_str(),nullptr));
            col++;
        }
        row++;
    }

    TscanBrightness();
}


physFormat::physDouble_asc::physDouble_asc(const char *ifilename)
    : physD(std::string(ifilename), PHYS_FILE)
{
    std::ifstream ifile(ifilename);
    // 1. get image statistics
    std::string tline;
    int nlines = 0, ncols = 0;
    while (getline (ifile, tline))
        nlines++;
    ifile.clear();
    ifile.seekg(0,std::ios::beg);

    getline (ifile, tline);
    std::stringstream ss(tline);

    while ((ss.tellg() < (int) ss.str().length()) && (ss.tellg() > -1)) {
        double readv;
        char ch;
        ss>>readv>>ch;
        ncols++;
    }

    DEBUG(5,"file has "<<nlines<<" lines and "<<ncols<<" cols");

    resize(ncols, nlines);

    // 2. read and save five
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    int row = 0, col = 0;
    char ch;
    while (getline(ifile, tline)) {
        std::stringstream sline(tline);
        while ((sline.tellg() < (int) sline.str().length()) && (sline.tellg() > -1)) {
            sline >> Timg_buffer[row*getW()+col] >> ch;
            col++;
        }
        col = 0;
        row++;
    }
    TscanBrightness();
}

physFormat::physInt_sif::physInt_sif(std::string ifilename)
    : nPhysImageF<int>(ifilename, PHYS_FILE)
{
    // Andor camera .sif file

    std::string temp_string;
    std::stringstream ss;
    int skiplines=0;

    std::ifstream ifile(ifilename.c_str(), std::ios::in | std::ios::binary);
    getline(ifile, temp_string);
    if ( temp_string.substr(0,5)!=std::string("Andor")) {
        WARNING("Does not start with Andor "<<ifilename);
        return;
    }

    // matrix informations on line 5
    for (size_t i=0; i<3; i++) {
        getline(ifile, temp_string);
        ss.str(""); ss.clear(); ss << std::setw(2) << std::setfill('0') << skiplines++;
        prop["sif-a-"+ss.str()]=temp_string;
    }
    getline(ifile, temp_string);
    ss.str(""); ss.clear(); ss << std::setw(2) << std::setfill('0') << skiplines++;
    prop["sif-b-"+ss.str()]=temp_string;

    int w, h;
    ss.str(temp_string);
    ss >> w;
    ss >> h;
    resize(w, h);

    getline(ifile, temp_string);
    ss.str(""); ss.clear(); ss << std::setw(2) << std::setfill('0') << skiplines++;
    prop["sif-c-"+ss.str()]=temp_string;

    getline(ifile, temp_string);
    ss.str(temp_string);

    int binary_header=0,useless=0;
    ss >> useless >> binary_header;
    DEBUG("unused value " << useless);
    std::vector<char> buf(binary_header);
    ifile.read(&buf[0], buf.size());


    /*
     * brought to you by some braindead @Andor's!
     *
     * 1. look for "Pixel number" (the first occurrence)
     * 2. look for a line with *a single* number on it (no indent)
     * 3. read the value and jump by the amount of lines!
     *
     * (thank you, Andor, thank you, I love this!)
     */

    temp_string.clear();
    std::string control_string="Pixel number";
    DEBUG(control_string.size());
    while ((!ifile.eof()) && (temp_string.substr(0,control_string.size()) != control_string)) {
        getline(ifile, temp_string);
        ss.str(""); ss.clear(); ss << std::setw(2) << std::setfill('0') << skiplines++;
        prop["sif-d-"+ss.str()]=temp_string;
    }

    temp_string.clear();
    unsigned int magic_number = 0; // usually 3 (lol)
    while (!ifile.eof()) {
        long int test_position = ifile.tellg();
        getline(ifile, temp_string);

        if (temp_string.size() > 10000) {
            temp_string.clear();
            ifile.seekg(test_position);
            WARNING("breaking sif loop");
            DEBUG("to praise the hindi god of love Kamadeva");
            break;
        }
        std::istringstream iss(temp_string);

        ss.str(""); ss.clear(); ss << std::setw(2) << std::setfill('0') << skiplines++;
        prop["sif-e-"+ss.str()]=temp_string;

        DEBUG(ss.str() << " " << temp_string.size());

        // most readable ever
        if ( !(iss >> std::noskipws >> magic_number).fail() && iss.eof() ) {
            prop["sif-magic_number"]=(int)magic_number;
            break;
        }
    }

    DEBUG("We are at byte "<< ifile.tellg());

    // jump magic lines
    DEBUG(5, "jump "<<magic_number<<" lines for the glory of Ra");
    for (size_t i=0; i<magic_number; i++) {
        getline(ifile, temp_string);
        std::istringstream iss(temp_string);
        ss.str(""); ss.clear(); ss << std::setw(2) << std::setfill('0') << skiplines++;
        prop["sif-f-"+ss.str()]=temp_string;
    }

    // consistency check

    int init_matrix = ifile.tellg();
    ifile.seekg(0, ifile.end);
    long int datacheck = (long int) ifile.tellg()-init_matrix-getSurf()*sizeof(float);

    if (ifile.eof() || ifile.fail()) {
        throw phys_fileerror("SIF: header parsing reached end of file "+ifilename);
    }

    if (datacheck < 0) {
        std::stringstream oss;
        oss<<ifilename << "\nFailed consistency check before SIF matrix read\n";
        oss<<"init_matrix: "<<init_matrix<<"\n";
        oss<<"end_file: "<<ifile.tellg()<<"\n";
        oss<<"matrix surface: "<<getSurf()<<"\n";
        oss<<"matrix size: "<<getSurf()*sizeof(float)<<"\n";

        //        property.dumper(std::cerr);

        throw phys_fileerror(oss.str());

    } else {
        // get data
        ifile.seekg(init_matrix);
        DEBUG(5,"size : "<<getW()<< " x " <<getH() << " + " << ifile.tellg() );
        ss.str(""); ss.clear();
        ss << getW()<< " x " <<getH() << " +" << init_matrix << " bytes";
        prop["sif-header"]=ss.str();
        std::vector<float> readb(getSurf());

        ifile.read((char*)(&readb[0]),getSurf()*sizeof(float));
        DEBUG(ifile.gcount());
        ifile.close();
        for (size_t i=0; i<getSurf(); i++) set(i,(int) readb[i]);

        TscanBrightness();
        DEBUG(get_min() << " " << get_max());

    }

}

physFormat::physShort_b16::physShort_b16(const char *ifilename)
    : nPhysImageF<short>(std::string(ifilename), PHYS_FILE)
{

    char *ptr = new char[3], tempch;
    char *readb=nullptr;
    char revision;

    int header_size=0; // cambia con la revision del file


    std::ifstream ifile(ifilename, std::ios::in | std::ios::binary);
    assert(ifile);

    ifile.read(ptr,3);

    //if ( string(ptr) != string("PCO") ) {
    if ( strncmp(ptr,"PCO",3) != 0 ) {
        WARNING("not a PCO file ["<<ptr<<"]");
        return;
    }

    // alla camWare succhiano i cazzi
    ifile.seekg(0x05,std::ios::beg);
    ifile.read(&revision,1);
    if (revision=='a') header_size=384;
    else if (revision=='b') header_size=512;


    // bpp (idiozia del formato: e' un carattere a 0x09)
    // altra idiozia: non capisco la specificazione di bpp in revision a: fisso a 2
    int bpp = 0;
    if (revision=='b') {
        ifile.seekg(0x09,std::ios::beg);
        ifile.read(&tempch,1);
        bpp = (int)tempch;
    } else if (revision=='a') bpp=2;

    int w, h;

    // width
    ifile.seekg(0x0c,std::ios::beg);
    ifile.read((char *)&w,4);


    // height
    ifile.seekg(0x10,std::ios::beg);
    ifile.read((char *)&h,4);

    resize(w, h);


    DEBUG(5,"width: "<<getW());
    DEBUG(5,"height: "<<getH());


    ifile.seekg(header_size,std::ios::beg);
    assert (readb = new char [w*bpp]);
    if (readb) {
        for (int i=0; i<h; i++) {
            memset(readb,0,w*bpp);
            ifile.read(readb,w*bpp);
            memcpy(Timg_matrix[i], readb, w*sizeof(short));
        }
    }
    ifile.close();
    delete [] readb;
    delete [] ptr;
}



std::string splitHamamatsuComments(std::string line) {
    std::string res;
    const char *mystart=line.c_str();
    bool instring{false};
    for (const char* p=mystart; *p; p++) {
        if (*p=='"')
            instring = !instring;
        else if (*p==',' && !instring) {
            res+=std::string(mystart,p-mystart)+"\n";
            mystart=p+1;
        }
    }
    res+=std::string(mystart);
    return res;
}

physFormat::physDouble_img::physDouble_img(std::string ifilename)
    : physD(ifilename, PHYS_FILE) {

    unsigned short buffer;
    std::ifstream ifile(ifilename.c_str(), std::ios::in | std::ios::binary);
    
    int my_width=0;
    int my_height=0;
    int skipbyte=0;
    int kind=-1;
    
    bool endian=false;
    
    ifile.read((char *)&buffer,sizeof(unsigned short));

    if (std::string((char *)&buffer,sizeof(unsigned short)) == "IM") { // Hamamatsu
        ifile.read((char *)&buffer,sizeof(unsigned short));
        skipbyte=buffer;
        ifile.read((char *)&buffer,sizeof(unsigned short));
        my_width=buffer;
        ifile.read((char *)&buffer,sizeof(unsigned short));
        my_height=buffer;
        
        for (int i=0;i<2;i++) {
            ifile.read((char *)&buffer,sizeof(unsigned short));
            DEBUG(">>>>>>>>>>> " <<  i << " " << buffer);
            if (buffer != 0)
                throw phys_fileerror("This file is detected as Hamamatsu ut it cannot be opened, please contact developpers");
        }

        ifile.read((char *)&buffer,sizeof(unsigned short));
        kind=buffer;
        
        for (int i=0;i<25;i++) {
            ifile.read((char *)&buffer,sizeof(unsigned short));
            DEBUG(i << " " << buffer);
            if (buffer != 0)
                throw phys_fileerror("This file is detected as Hamamatsu ut it cannot be opened, please contact developpers");
        }

        std::string buffer2;
        buffer2.resize(skipbyte);
        ifile.read((char *)&buffer2[0],skipbyte);
        
        buffer2.erase(std::remove(buffer2.begin(), buffer2.end(), '\t'), buffer2.end());
        buffer2.erase(std::remove(buffer2.begin(), buffer2.end(), '\r'), buffer2.end());
        prop["Hamamatsu"]=std::string(buffer2);

        std::vector<std::string> strings;

        std::string::size_type pos = 0;
        std::string::size_type prev = 0;
        while ((pos = buffer2.find("\n", prev)) != std::string::npos) {
            strings.push_back(buffer2.substr(prev, pos - prev));
            prev = pos + 1;
        }
        DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        DEBUG(buffer2);
        DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        strings.push_back(buffer2.substr(prev));
        for(unsigned int i=0; i<strings.size(); i++) {

            std::stringstream ss;
            ss << std::setw(log10(strings.size())+1) << std::setfill('0') << i;

            DEBUG( ss.str() << " <> " << strings[i]);
//            prop["Hamamatsu_"+ss.str()]=strings[i];

            std::regex my_regex(".*\\[(.*?)\\],(.*?)");

            std::smatch m;
            if(regex_match(strings[i],m,my_regex) &&m.size()==3) {
                DEBUG(m[0]);
                DEBUG(m[1]);
                DEBUG(m[2]);
                std::string res = splitHamamatsuComments(m[2]);
                prop["Hamamatsu_"+ss.str()+"_"+std::string(m[1])]=res;
            } else {
                prop["Hamamatsu_"+ss.str()]=strings[i];
            }
        }
        DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        
        switch (kind) {
            case 2: // unsigned short int
                kind=2;
                break;
            case 3: // unsigned int
                kind=4;
                break;
            default:
                break;
        }
        
    } else if (buffer == 512) { // ARP blue ccd camera w optic fibers...
        ifile.read((char *)&buffer,sizeof(unsigned short));
        if (buffer==7) {
            ifile.read((char *)&buffer,sizeof(unsigned short));
            skipbyte=buffer;
            ifile.seekg(skipbyte,std::ios_base::beg);
            ifile.read((char *)&buffer,sizeof(unsigned short));
            my_width=buffer;
            ifile.read((char *)&buffer,sizeof(unsigned short));
            my_height=buffer;
            kind=2;
            ifile.read((char *)&buffer,sizeof(unsigned short));
        }
    } else { // LIL images
        ifile.seekg(std::ios_base::beg);
        std::vector<unsigned int>lil_header (4);
        ifile.read((char *)&lil_header[0],lil_header.size()*sizeof(unsigned int));
        if (lil_header[0]==2 && lil_header[3]==1) {
            // lil_header[0] = dimension of the matrix
            // lil_header[3] = kind of data (1=unisgned short, 2=long, 3= float, 4=double)
            my_width=lil_header[1];
            my_height=lil_header[2];
            kind=2;
        }
    }
    
    if (kind!=-1) {
        DEBUG("->>>>>>>>>>>>");
        DEBUG("->>>>>>>>>>>>" << this <<  " " <<  my_width <<  " " << my_height);
        resize(my_width, my_height);
        skipbyte=ifile.tellg();
        ifile.close();
        prop["kind"]=kind;
        prop["skip bytes"]=skipbyte;

        phys_open_RAW(this,kind,skipbyte,endian);
    }
    DEBUG("exit physDouble_img");
}

physFormat::physUint_imd::physUint_imd(std::string ifilename)
    : physD(ifilename, PHYS_FILE)
{
    // Optronics luli

    unsigned short buffer_header;
    std::vector<std::string> exts = {"imd", "IMD", ifilename.substr(ifilename.size()-3,3)};
    std::string ifilenamebase=ifilename;
    ifilenamebase.resize(ifilenamebase.size()-3);

    for (auto &ext : exts) {
        ifilename=ifilenamebase+ext;
        DEBUG(ifilename.substr(ifilename.size()-3,3));
        std::ifstream ifile(ifilename.c_str(), std::ios::in | std::ios::binary);
        if (ifile) {
            unsigned short w=0;
            unsigned short h=0;

            ifile.read((char *)&buffer_header,sizeof(unsigned short));
            prop["imd-version"]=buffer_header;
            ifile.read((char *)&buffer_header,sizeof(unsigned short));
            w=buffer_header;
            ifile.read((char *)&buffer_header,sizeof(unsigned short));
            h=buffer_header;

            resize(w, h);
            std::vector<unsigned int> buf(w*h);
            ifile.read((char *)(&buf[0]),sizeof(unsigned int)*w*h);
#pragma omp parallel for
            for (size_t ii=0; ii<w*h; ii++) {
                set(ii, buf[ii]/1000.);
            }

            ifile.close();

            std::vector<std::string> extensions = {"imi", "IMI"};
            for (auto &ext : extensions) {
                ifilenamebase = ifilenamebase+ext;
                std::ifstream ifileimg(ifilenamebase.c_str(), std::ios::in);
                if (ifileimg) {
                    std::string comment(""),temp_line;
                    while (!ifileimg.eof()) {
                        getline(ifileimg, temp_line);
                        comment.append(temp_line);
                    }
                    ifileimg.close();
                    prop["imi-info"]=comment;
                    break;
                }
            }
            TscanBrightness();
            break;
        }
    }
}


// dump out for state save
void physFormat::phys_dump_binary(physD *my_phys, const char *fname) {
    std::ofstream ofile(fname, std::ios::out | std::ios::binary);
    physFormat::phys_dump_binary(my_phys,ofile);
}

void physFormat::phys_dump_binary(physD *my_phys, std::ofstream &ofile) {

    if (!ofile.fail() && my_phys != nullptr) {

        //int pos = ofile.tellg();
        //int written_data = 0;

        //ofile<<my_phys->property<<"\n";
        my_phys->prop.dumper(ofile);

        DEBUG("Starting binary dump...");


        //ofile<<my_phys->getW()<<"\n";
        //ofile<<my_phys->getH()<<"\n";

        // Compress data using zlib
        int buffer_size=my_phys->getSurf()*sizeof(double);
        std::vector<unsigned char> out(buffer_size);
        z_stream strm;

        strm.zalloc = Z_NULL;
        strm.zfree  = Z_NULL;
        strm.opaque = Z_NULL;
        int status;
        status=deflateInit2 (&strm, Z_DEFAULT_COMPRESSION, Z_DEFLATED, windowBits | GZIP_ENCODING, 8, Z_DEFAULT_STRATEGY);
        if (status < 0) {
            // .alex. -- questa non compila (e non sono sicuro sia a causa di anymap)
            WARNING("Zlib a bad status of " << status);
            exit (EXIT_FAILURE);
        }

        strm.next_in = (unsigned char *) my_phys->Timg_buffer;
        strm.avail_in = my_phys->getSurf()*sizeof(double);
        strm.avail_out = buffer_size;
        strm.next_out = &out[0];
        status= deflate (& strm, Z_FINISH);

        if (status < 0) {
            WARNING("Zlib a bad status of " << status);
            exit (EXIT_FAILURE);
        }
        int writtendata = buffer_size - strm.avail_out;

        deflateEnd (& strm);
        DEBUG(5,"writtendata " << writtendata);

        int oo = my_phys->getW();
        ofile.write((const char *)&oo, sizeof(int));
        oo = my_phys->getH();
        ofile.write((const char *)&oo, sizeof(int));

        ofile.write((const char *)&writtendata, sizeof (int));
        ofile.write((const char *)(&out[0]), sizeof (char)*writtendata);
        DEBUG("Binary dump finished");

        //	ofile.write((const char *)my_phys->Timg_buffer, my_phys->getSurf()*sizeof(double));
        ofile<<"\n"<<std::flush;

        //written_data = ofile.tellg()-pos;
    } else {
        throw phys_fileerror("ERROR writing binary dump data");
    }
}

void physFormat::phys_dump_ascii(physD *my_phys, std::ofstream &ofile)
{	

    if (ofile.good() && my_phys != nullptr) {

        std::stringstream ss;
        my_phys->prop.dumper(ss);
        std::string str = ss.str(), str2;

        size_t pos=0;
        while((pos = str.find("\n", pos)) != std::string::npos)
        {
            str.insert(pos+1, "# ");
            pos += 1;
        }
        str.insert(0, "# ");

        ofile<<str<<"\n";

        for (size_t i=0; i<my_phys->getH(); i++) {
            for (size_t j=0; j<my_phys->getW()-1; j++)
                ofile<<std::setprecision(8)<<my_phys->getPoint(j,i)<<"\t";
            ofile<<std::setprecision(8)<<my_phys->getPoint(my_phys->getW()-1,i)<<"\n";
        }

    } else {
        throw phys_fileerror("ERROR writing ASCII data");
    }

}

#ifdef HAVE_LIBTIFF

#define TIFFTAG_PHYSPROP  34595
#define TIFFTAG_NEUTRINO  34594

static const TIFFFieldInfo xtiffFieldInfo[] = {
    { TIFFTAG_PHYSPROP,  TIFF_VARIABLE, TIFF_VARIABLE, TIFF_ASCII,  FIELD_CUSTOM, 0, 0, const_cast<char*>("Neutrino") },
    { TIFFTAG_NEUTRINO,  TIFF_VARIABLE, TIFF_VARIABLE, TIFF_ASCII,  FIELD_CUSTOM, 0, 0, const_cast<char*>("Neutrino") }
};

static TIFFExtendProc parent_extender = nullptr;  // In case we want a chain of extensions

static void registerCustomTIFFTags(TIFF *tif)
{
    /* Install the extended Tag field info */
    TIFFMergeFieldInfo(tif, xtiffFieldInfo, sizeof(xtiffFieldInfo)/sizeof(xtiffFieldInfo[0]));
    if (parent_extender) (*parent_extender)(tif);
}

static void augment_libtiff_with_custom_tags() {
    static bool first_time = true;
    if (!first_time) return;
    first_time = false;
    DEBUG("Registering tiff neutrino extension");
    parent_extender = TIFFSetTagExtender(registerCustomTIFFTags);
}
#endif

std::vector <physD> physFormat::phys_open_tiff(std::string ifilename, bool separate_rgb) {
    std::vector <physD> vecReturn;
#ifdef HAVE_LIBTIFF
    TIFFErrorHandler oldhandler = TIFFSetWarningHandler(nullptr);
    augment_libtiff_with_custom_tags();
    TIFF* tif = TIFFOpen(ifilename.c_str(), "r");
    if (tif) {
        DEBUG("Opened");
        unsigned short config;

        do {
            DEBUG(vecReturn.size());
            TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &config);
            if (config==PLANARCONFIG_CONTIG ) {
                phys_properties tiff_prop;


                unsigned short samples=1, compression, format, fillorder, units=0;
                TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples);
                tiff_prop["Tiff_samples"]=samples;

                unsigned int count;
                long *MD_FileTag = nullptr ;
                if (TIFFGetField(tif, 33445, &count, &MD_FileTag)){
                    if (count==1) {
                        tiff_prop["TIFF_MD_FileTag"] = static_cast<int>(MD_FileTag[0]);
                        DEBUG("MD_FileTag " << MD_FileTag[0]);
                    }
                }
                unsigned int *MD_ScalePixel=nullptr;
                if (TIFFGetField(tif, 33446, &count, &MD_ScalePixel)){
                    if (count==1) {
                        tiff_prop["TIFF_MD_ScalePixel"] = vec2i(MD_ScalePixel[0],MD_ScalePixel[1]);
                        DEBUG("MD_ScalePixel " << count << " " << MD_ScalePixel << " " << MD_ScalePixel[0]<< " " << MD_ScalePixel[1] << " " << MD_ScalePixel[0]/MD_ScalePixel[1]);
                    }
                }
                const unsigned short *colortable=nullptr;
                if (TIFFGetField(tif, 33447, &count, &colortable)) {
                    if (count>1) {
                        tiff_prop["TIFF_MD_ColorTable"] = vec2f(colortable[1],colortable[count-2]);
                    }
                    DEBUG(colortable[1] << " " << colortable[count-2]);
                }
                const char *strdata=nullptr;
                if (TIFFGetField(tif, 33448, &count, &strdata)) {
                    tiff_prop["TIFF_MD_LabName"] = std::string(strdata);
                    DEBUG("33448 " << count << " " << strlen(strdata) << " " << std::string(strdata));
                }
                if (TIFFGetField(tif, 33449, &count, &strdata)) {
                    std::string res(strdata);
                    tiff_prop["TIFF_MD_SampleInfo"] = res;

                    std::string myreg(R"(PMT:(\d*)V, L(\d*), (\d*)(.*))");
                    std::regex my_regex(myreg);
                    std::smatch m;
                    DEBUG("33449 " << count << "\n" << std::string(strdata));
                    if(std::regex_search(res,m,my_regex)) {
                        DEBUG("found GEL");
                        tiff_prop["gel_V"]=std::stoi(m.str(1));
                        tiff_prop["gel_L"]=std::stoi(m.str(2));
                        tiff_prop["gel_R"]=std::stoi(m.str(3));
                        tiff_prop["gel_U"]=m.str(4);
                    }
                }
                if (TIFFGetField(tif, 33450, &count, &strdata)) {
                    tiff_prop["TIFF_MD_PrepDate"] = std::string(strdata);
                    DEBUG(count << " " << std::string(strdata));
                }
                if (TIFFGetField(tif, 33451, &count, &strdata)) {
                    tiff_prop["TIFF_MD_PrepTime"] = std::string(strdata);
                    DEBUG(count << " " << std::string(strdata));
                }
                if (TIFFGetField(tif, 33452, &count, &strdata)) {
                    tiff_prop["TIFF_MD_FileUnits"] = std::string(strdata);
                    DEBUG(count << " " << std::string(strdata));
                }

                TIFFGetField(tif, TIFFTAG_COMPRESSION, &compression);

                if (!TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &format)) {
                    format=SAMPLEFORMAT_UINT;
                }

                TIFFGetField(tif, TIFFTAG_FILLORDER, &fillorder);

                float resx=1.0, resy=1.0;
                TIFFGetField(tif, TIFFTAG_XRESOLUTION, &resx);
                TIFFGetField(tif, TIFFTAG_YRESOLUTION, &resy);
                if (resx!=0.0 && resy!=0.0) {
                    tiff_prop["scale"]=vec2f(1.0/resx,1.0/resy);
                }
                DEBUG("Resolution " << resx << " " << resy);
                TIFFGetField(tif, TIFFTAG_RESOLUTIONUNIT, &units);
                switch (units) {
                    case 1:
                        tiff_prop["unitsX"]=tiff_prop["unitsY"]="";
                        break;
                    case 2:
                        tiff_prop["unitsX"]=tiff_prop["unitsY"]="in";
                        break;
                    case 3:
                        tiff_prop["unitsX"]=tiff_prop["unitsY"]="cm";
                        break;
                }

                float posx=0.0, posy=0.0;
                TIFFGetField(tif, TIFFTAG_XPOSITION, &posx);
                TIFFGetField(tif, TIFFTAG_YPOSITION, &posy);
                tiff_prop["tiff_origin"]=vec2f(posx,posy);

                DEBUG("COMPRESSION_NONE " << compression << " " << COMPRESSION_NONE);
                DEBUG("PLANARCONFIG_CONTIG " << config << " " << PLANARCONFIG_CONTIG);
                DEBUG("FORMAT " << format);
                DEBUG("FILLORDER " << fillorder);
                DEBUG("SAMPLES " << samples);
                DEBUG("UNITS " << units);
                unsigned short extra;
                TIFFGetField(tif, TIFFTAG_EXTRASAMPLES, &extra);
                if (extra!=0) {
                    tiff_prop["Tiff_extra_samples"]=extra;
                }
                const char *soft=nullptr;
                if (TIFFGetField(tif, TIFFTAG_SOFTWARE, &soft)) {
                    tiff_prop["TIFF_Software"]=std::string(soft);
                    DEBUG("TIFFTAG_SOFTWARE " << soft);
                }
                const char *copyright=nullptr;
                if (TIFFGetField(tif, TIFFTAG_COPYRIGHT, &copyright)) {
                    tiff_prop["TIFF_copyright"]=std::string(copyright);
                    DEBUG("TIFFTAG_COPYRIGHT " << copyright);
                }
                const char *docname=nullptr;
                if (TIFFGetField(tif, TIFFTAG_DOCUMENTNAME, &docname)) {
                    tiff_prop["TIFF_docname"]=std::string(docname);
                    tiff_prop["phys_name"]=docname;
                    DEBUG(docname);
                }
                const char *neu_prop=nullptr;
                if (TIFFGetField(tif, TIFFTAG_PHYSPROP, &neu_prop)) {
                    std::string str_desc=std::string(neu_prop);
                    DEBUG(str_desc.size() << "\n" << str_desc);
                    std::stringstream ss(str_desc);
                    tiff_prop.loader(ss);
                }
                char *comment=nullptr;
                if (TIFFGetField(tif, TIFFTAG_NEUTRINO, &comment)) {
                    tiff_prop["neutrino"]=std::string(comment);
                    DEBUG(std::string(comment));
                }
                char *desc=nullptr;
                if (TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &desc)) {
                    std::string str_desc(desc);
                    tiff_prop["TIFF_description"]=str_desc;
                    DEBUG(str_desc.size() << "\n" << str_desc);
                    std::stringstream ss(str_desc);
                    vec2f display_range(0,0);
                    std::string my_line;
                    while (!(ss  >> my_line).fail()) {
                        auto index = my_line.find('=');
                        std::pair<std::string,std::string> keyVal;
                        if (index != std::string::npos) {
                            // Split around ':' character
                            std::string left=my_line.substr(0,index);
                            std::string right=my_line.substr(index+1);
                            if (left=="ImageJ") {
                                tiff_prop["ImageJ-version"]=right;
                            } else if (left=="min") {
                                std::stringstream rightss(right);
                                double val=0;
                                rightss >> val;
                                display_range.set_first(val);
                            } else if (left=="max") {
                                std::stringstream rightss(right);
                                double val=0;
                                rightss >> val;
                                display_range.set_second(val);
                            }
                        }
                    }
                    if (display_range != vec2f(0,0)) {
                        tiff_prop["display_range"] =display_range;
                    }
                }
                tiff_prop["phys_from_name"]=ifilename;
                unsigned short bytesperpixel=0;
                TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bytesperpixel);
                bytesperpixel/=8;
                unsigned int w=0, h=0;
                TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
                TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
                DEBUG("Width " << w << " Height " << h << " BYTESperPIXEL " << bytesperpixel);

                if (w*h>0 && bytesperpixel>0 ) {
                    tsize_t scanlineSize=TIFFScanlineSize(tif);
                    tdata_t buf = _TIFFmalloc(scanlineSize);
                    DEBUG("Buf allocated " << scanlineSize);
                    physD my_phys(w,h,0.0,"");
                    DEBUG("here");

                    for (int k=0;k<samples;k++) {
                        std::stringstream ss(ifilename);
                        if(docname) {
                            ss << std::string(docname);
                        } else {
                            ss << ifilename;
                            if (vecReturn.size()>0) {
                                ss << " " << vecReturn.size();
                            }
                        }
                        if (separate_rgb) {
                            ss << " c" << samples;
                        }
                        my_phys.setName(ss.str());
                        for (auto &pro : tiff_prop) {
                            my_phys.prop[pro.first] =pro.second;
                            DEBUG(pro.first << " : " << pro.second);
                        }
                        my_phys.setType(PHYS_FILE);

                        DEBUG("here");
                        for (unsigned int j = 0; j < h; j++) {
                            TIFFReadScanline(tif, buf, j);
                            for (unsigned int i=0; i<w; i++) {
                                double val=0;
                                if (bytesperpixel == sizeof(char)) {
                                    if (format==SAMPLEFORMAT_UINT) {
                                        val+=((unsigned char*)buf)[i*samples+k];
                                    } else if (format==SAMPLEFORMAT_INT) {
                                        val+=((char*)buf)[i*samples+k];
                                    }
                                } else if (bytesperpixel == sizeof(short)) {
                                    if (format==SAMPLEFORMAT_UINT) {
                                        val+=((unsigned short*)buf)[i*samples+k];
                                    } else if (format==SAMPLEFORMAT_INT) {
                                        val+=((short*)buf)[i*samples+k];
                                    }
                                } else if (bytesperpixel == sizeof(int)) {
                                    if (format==SAMPLEFORMAT_UINT) {
                                        val+=((unsigned int*)buf)[i*samples+k];
                                    } else if (format==SAMPLEFORMAT_INT) {
                                        val+=((int*)buf)[i*samples+k];
                                    } else if (format==SAMPLEFORMAT_IEEEFP) {
                                        val+=((float*)buf)[i*samples+k];
                                    }
                                } else if (bytesperpixel == sizeof(double)) {
                                    if (format==SAMPLEFORMAT_UINT) {
                                        val+=((long unsigned int*)buf)[i*samples+k];
                                    } else if (format==SAMPLEFORMAT_INT) {
                                        val+=((long int*)buf)[i*samples+k];
                                    } else if (format==SAMPLEFORMAT_IEEEFP) {
                                        val+=((double*)buf)[i*samples+k];
                                    }
                                }
                                my_phys.set(i,j,my_phys.point(i,j)+val);
                            }
                        }
                        if (tiff_prop.have("gel_V") && tiff_prop.have("gel_L") && tiff_prop.have("gel_R"), tiff_prop.have("TIFF_MD_FileTag")){
                            DEBUG("here we are");
                            physMath::phys_flip_ud(my_phys);
                            double D=pow(pow(2,8*bytesperpixel)-1,2);
                            double S=pow(10,-15.845+6.861*0.4343*log(tiff_prop["gel_V"].get_i()));
                            double a=(pow(tiff_prop["gel_R"].get_i(),2)/10000)*(4000/S)*pow(10,tiff_prop["gel_L"].get_i()/2.)/D;
                            if (tiff_prop["TIFF_MD_FileTag"].get_i()==2) {
                                physMath::phys_square(my_phys);
                            }
                            physMath::phys_multiply(my_phys,a);
                            my_phys.prop["unitsCB"] = "PSL";
                            my_phys.set_scale(tiff_prop["gel_R"].get_i()/10000.,tiff_prop["gel_R"].get_i()/10000.);
                        }
                        my_phys.TscanBrightness();
                        if (separate_rgb) {
                            vecReturn.push_back(my_phys);
                        }
                    }
                    if (!separate_rgb && samples>0) {
                        physMath::phys_divide(my_phys,samples);
                        vecReturn.push_back(my_phys);
                    }
                    _TIFFfree(buf);
                } else {
                    std::stringstream ss("TIFFTAG_PLANARCONFIG: ");
                    ss << config << " not supported";
                    throw phys_fileerror(ss.str());
                }

            }
        } while (TIFFReadDirectory(tif)==1);
        TIFFClose(tif);
    } else {
        throw phys_fileerror("TIFF: contact Neutrino developers");
    }
    TIFFSetWarningHandler(oldhandler);
#else
    throw phys_fileerror("Neutrino was compiled without TIFF support");
#endif

    return vecReturn;
}

#ifdef HAVE_LIBTIFF
void physFormat::phys_write_one_tiff(physD *my_phys, TIFF* tif) {
    TIFFSetWarningHandler(nullptr);
    TIFFSetField(tif, TIFFTAG_SUBFILETYPE, 0);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, 1);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_DOCUMENTNAME, my_phys->getName().c_str());
    std::stringstream prop_ss;
    my_phys->prop.dumper(prop_ss);
    std::string description=prop_ss.str();

    std::replace( description.begin(), description.end(), '\0', ' ');

    DEBUG(description);

    TIFFSetField(tif, TIFFTAG_PHYSPROP, description.c_str());
    TIFFSetField(tif, TIFFTAG_SOFTWARE, "Neutrino");
    TIFFSetField(tif, TIFFTAG_COPYRIGHT, "http://web.luli.polytchnique.fr/neutrino");

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, my_phys->getW());
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, my_phys->getH());
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, my_phys->getH());
    float scalex=my_phys->get_scale().x();
    TIFFSetField(tif, TIFFTAG_XRESOLUTION, 1.0/scalex);
    float scaley=my_phys->get_scale().y();
    TIFFSetField(tif, TIFFTAG_YRESOLUTION, 1.0/scaley);
//    float origx=my_phys->get_origin().x();
//    TIFFSetField(tif, TIFFTAG_XPOSITION, origx);
//    float origy=my_phys->get_origin().y();
//    TIFFSetField(tif, TIFFTAG_XPOSITION, origy);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8*sizeof(float));
    DEBUG("tiff " << TIFFScanlineSize(tif) << " " << my_phys->getW() <<  " " << my_phys->getH());
    unsigned char *buf = (unsigned char *) _TIFFmalloc(TIFFScanlineSize(tif));
    for (size_t j = 0; j < my_phys->getH(); j++) {
        for (size_t i=0; i<my_phys->getW(); i++) {
            ((float*)buf)[i]=(float)my_phys->point(i,j) ;
        }
        TIFFWriteScanline(tif, buf, j, 0);
    }
    _TIFFfree(buf);
}
#endif

void physFormat::phys_write_tiff(physD *my_phys, std::string ofilename) {
#ifdef HAVE_LIBTIFF
    augment_libtiff_with_custom_tags();
    TIFF* tif = TIFFOpen(ofilename.c_str(), "w");
    if (tif && my_phys) {
        physFormat::phys_write_one_tiff(my_phys, tif);
    } else {
        throw phys_fileerror("Problem writing tiff");
    }
    TIFFClose(tif);
#else
    throw phys_fileerror("Neutrino was not compiled with tiff support");
#endif
}

void physFormat::phys_write_tiff(std::vector <physD *> vecPhys, std::string ofilename) {
#ifdef HAVE_LIBTIFF
    augment_libtiff_with_custom_tags();
    TIFF* tif = TIFFOpen(ofilename.c_str(), "w");
    if (tif) {
        for (unsigned int i=0; i<vecPhys.size(); i++) {
            phys_write_one_tiff(vecPhys[i], tif);
            if(TIFFWriteDirectory(tif)==0) {
                throw phys_fileerror("Problem writing multiple images in same tiff file");
            }
        }
    } else {
        throw phys_fileerror("Problem writing tiff");
    }
    TIFFClose(tif);
#else
    throw phys_fileerror("Neutrino was not compiled with tiff support");
#endif
}


std::vector <physD> physFormat::phys_open_spe(std::string ifilename) {
    std::vector <physD> vecReturn;

    std::ifstream ifile(ifilename.c_str(), std::ios::in);
    std::string header;
    header.resize(4100);
    ifile.read((char *)&header[0],header.size());
    unsigned short type= *((unsigned short *) &header[108]);

    unsigned short height= *((unsigned short *) &header[656]);
    unsigned short width= *((unsigned short *) &header[42]);
    unsigned short noscan= *((unsigned short *) &header[34]);
    unsigned int NumFrames= *((unsigned int *) &header[1446]);

    if (noscan == 65535) {
        int lnoscan = *((int *) &header[664]);
        if (lnoscan != -1) NumFrames = lnoscan / height;
    } else {
        NumFrames = noscan / height;
    }

    DEBUG(type << " " << height  << " " << width << " " << noscan << " " << NumFrames);

    for (unsigned int nf=0;nf<NumFrames;nf++) {
        physD phys(width,height,0.0);
        phys.prop["spe-frame"]=vec2f(nf,NumFrames);
        switch (type) {
            case 0: {
                    phys.prop["spe-type"]="float";
                    std::vector<float> buffer(width*height);
                    ifile.read((char*) &buffer[0],width*height*sizeof(float));
                    for (unsigned int i=0; i<phys.getSurf(); i++) {
                        phys.set(i,buffer[i]);
                    }
                    break;
                }
            case 1: {
                    phys.prop["spe-type"]="int";
                    std::vector<int> buffer(width*height);
                    DEBUG(sizeof(long));
                    ifile.read((char*) &buffer[0],width*height*sizeof(int));
                    for (unsigned int i=0; i<phys.getSurf(); i++) {
                        phys.set(i,buffer[i]);
                    }
                    break;
                }
            case 2: {
                    phys.prop["spe-type"]="short";
                    std::vector<short> buffer(width*height);
                    ifile.read((char*) &buffer[0],width*height*sizeof(short));
                    for (unsigned int i=0; i<phys.getSurf(); i++) {
                        phys.set(i,buffer[i]);
                    }
                    break;
                }
            case 3: {
                    phys.prop["spe-type"]="unsigned short";
                    std::vector<unsigned short> buffer(width*height);
                    ifile.read((char*) &buffer[0],width*height*sizeof(unsigned short));
                    for (unsigned int i=0; i<phys.getSurf(); i++) {
                        phys.set(i,buffer[i]);
                    }
                    break;
                }
            default:
                break;
        }

        phys.TscanBrightness();
        vecReturn.push_back(phys);
    }

    ifile.close();
    return vecReturn;
}

std::vector <physD> physFormat::phys_open_pcoraw(std::string ifilename) {
    std::vector <physD> vecReturn;
    throw phys_fileerror("Neutrino has not yet implemented PCO read");

    //    std::ifstream ifile(ifilename.c_str(), std::ios::in);
    //    string header(2,' ');
    //    ifile.read((char *)&header[0],header.size());
    //    DEBUG("HERE >" << header << "<");
    //    if (header == "II") {
    //        DEBUG("HERE");
    //    } else {
    //        throw phys_fileerror("Neutrino has problem reading PCO " + header);
    //    }

    //    ifile.close();
    return vecReturn;
}

std::vector <physD> physFormat::phys_open_inf(std::string ifilename) {
    std::vector <physD> imagelist;
    std::ifstream ifile(ifilename.c_str(), std::ios::in);
    if (ifile) {
        std::string ifilenameimg=ifilename;
        ifilenameimg.replace(ifilenameimg.size()-3,3,"img");
        //        ifilenameimg.resize(ifilenameimg.size()-3);
        //        ifilenameimg = ifilenameimg+"img";

        std::ifstream imgstream(ifilenameimg);
        if (imgstream.good()) {
            DEBUG(">>>>>>>>>>>>>>>>>>>" <<ifilenameimg);
            std::string line;
            getline(ifile,line);
            if (line.compare(std::string("BAS_IMAGE_FILE"))!=0) {
                throw phys_fileerror("File does not start with BAS_IMAGE_FILE");
                return imagelist;
            }
            getline(ifile,line); //this is the basename
            DEBUG("basename: " << line);
            getline(ifile,line); //this we don't know what it is
            DEBUG("unknown : " << line);
            getline(ifile,line);
            double resx=atof(line.c_str());
            getline(ifile,line);
            double resy=atof(line.c_str());
            getline(ifile,line);
            int bit=atoi(line.c_str());
            getline(ifile,line);
            int w=atoi(line.c_str());
            getline(ifile,line);
            int h=atoi(line.c_str());
            physD linearized(w,h,0.0,ifilename);
            linearized.setType(PHYS_FILE);
            physD original(w,h,0.0,ifilenameimg.c_str());
            original.setType(PHYS_FILE);
            original.setShortName("original");

            switch (bit) {
                case 8:
                    phys_open_RAW(&original,0,0,true);
                    break;
                case 16:
                default:
                    phys_open_RAW(&original,2,0,true);
                    break;
            }

            linearized.prop["inf-resx"] = resx;
            linearized.prop["inf-resy"] = resy;
            linearized.set_scale(resx/1000.,resy/1000.);
            linearized.prop["unitsX"] = "mm";
            linearized.prop["unitsY"] = "mm";
            linearized.prop["unitsCB"] = "PSL";

            getline(ifile,line);
            double sensitivity=atof(line.c_str());
            linearized.prop["inf-sensitivity"] = sensitivity;
            getline(ifile,line);
            double latitude=atof(line.c_str());
            linearized.prop["inf-latitude"] = latitude;
            getline(ifile,line);
            linearized.prop["inf-date"] = line;
            getline(ifile,line);
            linearized.prop["inf-number"] = line;
            getline(ifile,line); //empty line
            getline(ifile,line);
            linearized.prop["inf-scanner"] = line;

            getline(ifile,line); //empty line
            getline(ifile,line);
            if (line.compare(std::string("*** more info ***"))==0) {
                getline(ifile,line);
                int nprop=atoi(line.c_str());
                std::string ss;
                for (int i=0;i<nprop;i++) {
                    getline(ifile,line); //empty line
                    ss += line + "\n" ;
                }
                DEBUG(ss);
                linearized.prop["inf-more-info"] = ss;
            }
            double bitVal=pow(2,bit)-1;
            double sens=4000.0/sensitivity;
            double res2=(resx*resy)/10000.0;
#pragma omp parallel for
            for (size_t i=0;i<original.getSurf();i++) {
                if (original.point(i) != 0) {
                    linearized.set(i, res2 * sens * exp(M_LN10*latitude*(original.point(i)/bitVal-0.5)));
//                    double a=pow(1/(pow(2,bit)-1),2)*(resx/100)*(resy/100)*(4000/sensitivity)*pow(10,latitude/2.0);
//                    DEBUG("a = " << a);
                } else {
                    linearized.set(i,0.0);
                }
            }
            physMath::phys_flip_lr(linearized);
            linearized.TscanBrightness();
            imagelist.push_back(linearized);
#ifdef __phys_debug
            original.TscanBrightness();
            imagelist.push_back(original);
#endif
        }
        ifile.close();
    }

    return imagelist;
}

bool physFormat::fits_check_error (int status) {
#ifdef HAVE_LIBCFITSIO
    if (status) {
        char status_char[FLEN_STATUS], errmsg_char[FLEN_ERRMSG];

        fits_get_errstatus(status, status_char);  /* get the error description */

        std::stringstream ss;
        ss << "FITSIO status = " << status << " " << std::string(status_char) << std::endl;
        while ( fits_read_errmsg(errmsg_char) ) {
            ss << std::string(errmsg_char) << std::endl;
        }
        throw phys_fileerror(ss.str());
        return true;
    }
#endif
    return false;
}

void physFormat::phys_write_fits(physD *phys, const char * fname, float compression) {
#ifdef HAVE_LIBCFITSIO
    fitsfile *fptr;       /* pointer to the FITS file, defined in fitsio.h */
    int status=0;

    fits_create_file(&fptr, fname, &status);
    if (physFormat::fits_check_error(status)) return;

    fits_set_compression_type(fptr, GZIP_1, &status);
    if (physFormat::fits_check_error(status)) return;

    int ndim=2;
    long ndimLong[2]={(long)phys->getW(), (long)phys->getH()};
    fits_set_tile_dim(fptr,ndim,ndimLong,&status);
    if (physFormat::fits_check_error(status)) return;

    fits_set_quantize_level(fptr, 0.0, &status);
    if (physFormat::fits_check_error(status)) return;

    long naxes[2]; naxes[0]= phys->getW(); naxes[1] =phys->getH();

    fits_create_img(fptr,  DOUBLE_IMG, 2, naxes, &status);
    if (physFormat::fits_check_error(status)) return;

    double orig_x=phys->get_origin().x();
    fits_update_key(fptr, TDOUBLE, "ORIGIN_X", &orig_x, "nPhysImage origin x", &status);
    if (physFormat::fits_check_error(status)) return;

    double orig_y=phys->get_origin().y();
    fits_update_key(fptr, TDOUBLE, "ORIGIN_Y", &orig_y, "nPhysImage origin y", &status);
    if (physFormat::fits_check_error(status)) return;
    
    double scale_x=phys->get_scale().x();
    fits_update_key(fptr, TDOUBLE, "SCALE_X", &scale_x, "nPhysImage scale x", &status);
    if (physFormat::fits_check_error(status)) return;

    double scale_y=phys->get_scale().y();
    fits_update_key(fptr, TDOUBLE, "SCALE_Y", &scale_y, "nPhysImage scale y", &status);
    if (physFormat::fits_check_error(status)) return;

    std::string name=phys->getName();
    fits_update_key_longstr(fptr, "NAME", &name[0], "nPhysImage name", &status);
    if (physFormat::fits_check_error(status)) return;

    // write actual data
    fits_write_img(fptr, TDOUBLE, 1, phys->getSurf(), phys->Timg_buffer, &status);
    if (physFormat::fits_check_error(status)) return;

    fits_close_file(fptr, &status);
    if (physFormat::fits_check_error(status)) return;

#else
    throw phys_fileerror("Neutrino compiled without FITS support");
#endif
}

std::vector <physD> physFormat::phys_open_fits(std::string ifilename) {
    std::vector<physD> retVec;
#ifdef HAVE_LIBCFITSIO
    fitsfile *fptr;
    char card[FLEN_CARD];
    int status = 0, ii;

    fits_open_file(&fptr, ifilename.c_str(), READONLY, &status);
    int bitpix;
    int anaxis;

    fits_is_compressed_image(fptr, &status);
    if (physFormat::fits_check_error(status)) return retVec;
    DEBUG("fits compressed " << status);
    
    int hdupos=0;
    fits_get_hdu_num(fptr, &hdupos);
    if (physFormat::fits_check_error(status)) return retVec;

    DEBUG("fits_get_hdu_num " << hdupos);

    for (; !status; hdupos++)  {
        DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");

        physD phys(ifilename,PHYS_FILE);

        DEBUG(phys.getShortName());
        int hdutype;
        fits_get_hdu_type(fptr, &hdutype, &status);
        if (physFormat::fits_check_error(status)) return retVec;

        DEBUG("fits_get_hdu_type " << hdutype);
        
        
        // 		if (hdutype == IMAGE_HDU) {
        // 			long naxes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        // 			for (ii = 0; ii < 9; ii++)
        // 				naxes[ii] = 1;
        // 			  int naxis = 0;
        // 			fits_get_img_param(fptr, 9, &bitpix, &naxis, naxes, &status);
        //
        // 			long totpix = naxes[0] * naxes[1] * naxes[2] * naxes[3] * naxes[4] * naxes[5] * naxes[6] * naxes[7] * naxes[8];
        // // 			DEBUG("totpix " << totpix);
        // 		}
        
        fits_get_img_type(fptr,&bitpix,&status);
        DEBUG(5,"fits_get_img_type " << bitpix);
        
        fits_get_img_dim(fptr,&anaxis,&status);
        DEBUG(5,"fits_get_img_dim " << anaxis);
        vec2f orig=phys.get_origin();
        vec2f scale=phys.get_scale();
        
        int nkeys;
        fits_get_hdrspace(fptr, &nkeys, nullptr, &status);
        DEBUG(5,"nkeys " << nkeys);

        for (ii = 1; ii <= nkeys; ii++)  {
            fits_read_record(fptr, ii, card, &status);
            if (physFormat::fits_check_error(status)) return retVec;

            std::string cardStr=std::string(card);
            DEBUG("key " << ii << " : "  << cardStr);
            // 			transform(cardStr.begin(), cardStr.end(), cardStr.begin(), ::tolower);
            if (fits_get_keyclass(card)==TYP_USER_KEY) {
                std::string ctrl="ORIGIN_X";
                if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
                    char dtype;
                    fits_get_keytype(card, &dtype, &status);
                    if (dtype=='F') {
                        double val;
                        fits_read_key_dbl(fptr, ctrl.c_str(), &val, nullptr, &status );
                        orig.set_first(val);
                    }
                }
                ctrl="ORIGIN_Y";
                if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
                    char dtype;
                    fits_get_keytype(card, &dtype, &status);
                    if (dtype=='F') {
                        double val;
                        fits_read_key_dbl(fptr, ctrl.c_str(), &val, nullptr, &status );
                        orig.set_second(val);
                    }
                }
                ctrl="SCALE_X";
                if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
                    char dtype;
                    fits_get_keytype(card, &dtype, &status);
                    if (dtype=='F') {
                        double val;
                        fits_read_key_dbl(fptr, ctrl.c_str(), &val, nullptr, &status );
                        scale.set_first(val);
                    }
                }
                ctrl="SCALE_Y";
                if (cardStr.compare(0,ctrl.length(),ctrl)==0) {
                    char dtype;
                    fits_get_keytype(card, &dtype, &status);
                    if (dtype=='F') {
                        double val;
                        fits_read_key_dbl(fptr, ctrl.c_str(), &val, nullptr, &status );
                        scale.set_second(val);
                    }
                }
            } else {
                std::stringstream ss; ss << std::setw(log10(nkeys)+1) << std::setfill('0') << ii;
                phys.prop["fits-"+ss.str()]=card;
            }
        }
        phys.set_origin(orig);
        phys.set_scale(scale);
        // 		property.dumper(std::cerr);
        
        DEBUG("here");

        std::vector<long> axissize(anaxis,0),fpixel(anaxis,1);

        fits_get_img_size(fptr,anaxis,&axissize[0],&status);
        if (physFormat::fits_check_error(status)) return retVec;

        long totalsize=1;
        for(int i=0; i<anaxis; i++) {
            totalsize*=axissize[i];
        }
        DEBUG("totalsize " << totalsize);

        if (anaxis==2) {
            phys.resize(axissize[0],axissize[1]);
            
            fits_read_pix(fptr, TDOUBLE, &fpixel[0], totalsize, nullptr, (void *)phys.Timg_buffer, nullptr, &status);
            if (physFormat::fits_check_error(status)) return retVec;

            phys.TscanBrightness();
        }

        if (phys.getSurf()) {
            retVec.push_back(phys);
        }

        fits_movrel_hdu(fptr, 1, nullptr, &status);  /* try to move to next HDU */

        if (status == END_OF_FILE) {
            status=0;
            break;
        }

        if (physFormat::fits_check_error(status)) {
            return retVec;
        }
    }
    fits_close_file(fptr, &status);
    physFormat::fits_check_error(status);
#endif
    DEBUG("out of here");
    return retVec;
}

std::vector <physD> physFormat::phys_resurrect_binary(std::string fname) {

    std::vector <physD> imagelist;
    std::ifstream ifile(fname.c_str(), std::ios::in | std::ios::binary);
    int ret;
    while(ifile.peek()!=-1) {

        physD datamatrix;
        DEBUG("here");

        ret=physFormat::phys_resurrect_binary(datamatrix,ifile);
        if (ret>=0 && datamatrix.getSurf()>0) {
            imagelist.push_back(datamatrix);
        }

    }
    ifile.close();
    return imagelist;
}

int
physFormat::phys_resurrect_binary(physD& my_phys, std::ifstream &ifile) {

    if (ifile.fail() || ifile.eof()) {
        throw phys_fileerror("istream error");
        return -1;
    }

    my_phys.prop.loader(ifile);

#ifdef __phys_debug
    my_phys.prop.dumper(std::cerr);
#endif

    // w/h/size binary read
    int my_w, my_h, buffer_size;
    ifile.read((char *)&my_w, sizeof(int));
    ifile.read((char *)&my_h, sizeof(int));
    ifile.read((char *)&buffer_size, sizeof(int));

    DEBUG("w: "<<my_w);
    DEBUG("h: "<<my_h);
    DEBUG("s: "<<buffer_size);

    if (buffer_size>(int) (my_w*my_h*sizeof(double))) return -1; // this should not happend

    my_phys.resize(my_w,my_h);
    std::vector<unsigned char> in(buffer_size);
    ifile.read((char*)&in[0], buffer_size);


    z_stream strm;

    strm.zalloc = Z_NULL;
    strm.zfree  = Z_NULL;
    strm.opaque = Z_NULL;
    int status;
    status = inflateInit2 (&strm,windowBits | GZIP_ENCODING);
    if (status < 0) {
        std::ostringstream oss;
        oss << "Zlib inflateInit2 : got a bad status of " << status;
        throw phys_fileerror(oss.str());
        return (EXIT_FAILURE);
    }

    strm.next_in = &in[0];
    strm.avail_in = buffer_size;
    strm.next_out = (unsigned char *)my_phys.Timg_buffer;
    strm.avail_out = my_phys.getSurf()*sizeof(double);
    status = inflate (&strm, Z_SYNC_FLUSH);
    if (status < 0) {
        std::ostringstream oss;
        oss << "Zlib inflate : got a bad status of " << status;
        throw phys_fileerror(oss.str());
        return (EXIT_FAILURE);
    }
    inflateEnd (& strm);

    my_phys.setType(PHYS_FILE);

    //	ifile.read((char *)my_phys.Timg_buffer, my_phys.getSurf()*sizeof(double));
    my_phys.TscanBrightness();

    std::string line;
    getline(ifile,line);

    DEBUG(line);
    return 0;
}

void physFormat::phys_open_RAW(physD * my_phys, int kind, int skipbyte, bool endian){
    std::ifstream ifile(my_phys->getName().c_str(), std::ios::in | std::ios::binary);
    if (!ifile.fail()) {
        my_phys->prop["raw-skip"]=skipbyte;
        my_phys->prop["raw-endian"]= endian? 1:0;
        my_phys->prop["raw-kind"]= endian? 1:0;

        if (my_phys!=nullptr && my_phys->getSurf()>0) {

            ifile.seekg(skipbyte);
            if (ifile.fail() || ifile.eof()) throw phys_fileerror("Not enough space on file");
            int bpp=0;

            DEBUG (my_phys->getSurf());
            switch (kind) {
                case 0: bpp=sizeof(unsigned char);      my_phys->prop["raw-kind"]= "unsigned char";     break;
                case 1: bpp=sizeof(signed char);        my_phys->prop["raw-kind"]= "signed char";       break;
                case 2: bpp=sizeof(unsigned short);     my_phys->prop["raw-kind"]= "unsigned short";    break;
                case 3: bpp=sizeof(signed short);       my_phys->prop["raw-kind"]= "signed short";      break;
                case 4: bpp=sizeof(unsigned int);       my_phys->prop["raw-kind"]= "unsigned int";      break;
                case 5: bpp=sizeof(signed int);         my_phys->prop["raw-kind"]= "signed int";        break;
                case 6: bpp=sizeof(float);              my_phys->prop["raw-kind"]= "float";             break;
                case 7: bpp=sizeof(double);             my_phys->prop["raw-kind"]= "double";            break;
                default: kind=-1;
            }

            if (kind!=-1) {

                std::vector<char> buffer(bpp*my_phys->getSurf());
                ifile.read(&buffer[0], buffer.size());
                ifile.close();
                for (unsigned int i=0;i<my_phys->getSurf();i++) {
                    switch (kind) {
                        case 0: {
                                unsigned char buf = *((unsigned char*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<unsigned char>(buf) : buf);
                                break;
                            }
                        case 1: {
                                signed char buf = *((signed char*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<signed char>(buf) : buf);
                                break;
                            }
                        case 2: {
                                unsigned short buf = *((unsigned short*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<unsigned short>(buf) : buf);
                                break;
                            }
                        case 3: {
                                signed short buf = *((signed short*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<signed short>(buf) : buf);
                                break;
                            }
                        case 4: {
                                unsigned int buf = *((unsigned int*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<unsigned int>(buf) : buf);
                                break;
                            }
                        case 5: {
                                signed int buf = *((signed int*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<signed int>(buf) : buf);
                                break;
                            }
                        case 6: {
                                float buf = *((float*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<float>(buf) : buf);
                                break;
                            }
                        case 7: {
                                double buf = *((double*)(&buffer[i*bpp]));
                                my_phys->set(i,endian ? swap_endian<double>(buf) : buf);
                                break;
                            }
                    }
                }

                my_phys->TscanBrightness();
            } else {
                throw phys_fileerror("Unknown kind of data");
            }
        } else {
            throw phys_fileerror("Empty size");
        }
    } else {
        my_phys->resize(0,0);
        throw phys_fileerror("Can't open file "+my_phys->getName());
    }
    DEBUG("exit phys_open_RAW");
}
//! open HDF4 file (works for: Omega LLR  visar images)
std::vector <physD> physFormat::phys_open_HDF4(std::string fname) {
    std::vector <physD> imagelist;
    DEBUG("HERE");
#if defined(HAVE_LIBMFHDF) || defined(HAVE_LIBMFHDFDLL)
    DEBUG("HERE2");
    int32 sd_id, sds_id, n_datasets, n_file_attrs, index,status ;
    int32 dim_sizes[3];
    int32 rank, num_type, attributes;
    int32 i;

    char name[64];

    physD background;
    /* Open the file and initiate the SD interface. */
    sd_id = SDstart(fname.c_str(), DFACC_READ);
    if (sd_id != FAIL) {
        /* Determine the contents of the file. */
        SDfileinfo(sd_id, &n_datasets, &n_file_attrs);
        DEBUG(5,"datasets " << n_datasets);
        for (index = 0; index < n_datasets; index++) {
            DEBUG(5,"Image number " << index);
            sds_id = SDselect(sd_id, index);
            SDgetinfo(sds_id, name, &rank, dim_sizes, &num_type, &attributes);
            DEBUG(5,"	  rank " << rank << " attributes " << attributes << " : " << num_type);
            for (i=0;i<rank;i++) {
                DEBUG(5,"	  " << i << " = " << dim_sizes[i]);
            }

            int surf=1;
            int32 start[rank], edges[rank];
            for (i=0;i<rank;i++) {
                surf*=dim_sizes[i];
                start[i]=0;
                edges[i]=dim_sizes[i];
            }
            std::vector<char> data;
            switch (num_type) {
                case DFNT_CHAR:
                case DFNT_UCHAR:
                case DFNT_INT8:
                case DFNT_UINT8:
                    data.resize(surf*sizeof(char));
                    break;
                case DFNT_FLOAT32:
                    data.resize(surf*sizeof(float));
                    break;
                case DFNT_FLOAT64:
                    data.resize(surf*sizeof(double));
                    break;
                case DFNT_INT16:
                case DFNT_UINT16:
                    data.resize(surf*sizeof(short));
                    break;
                case DFNT_INT32:
                case DFNT_UINT32:
                    data.resize(surf*sizeof(int));
                    break;
            }

            if (data.size()>0) {
                status=SDreaddata(sds_id,start,nullptr,edges,(VOIDP)&data[0]);
                if (status!=FAIL && rank>1) {
                    int numMat=(rank==2?1:dim_sizes[0]);
                    if (dim_sizes[rank-1]*dim_sizes[rank-2]>0) {
                        DEBUG(5,"	  num mat " << numMat << " " << dim_sizes[rank-1]<< " x " << dim_sizes[rank-2]);
                        for (i=0;i<numMat;i++) {
                            DEBUG(5,"	  mat " << i);
                            physD phys(dim_sizes[rank-1],dim_sizes[rank-2],0.0,"hdf4");
                            for (size_t k=0;k<phys.getSurf();k++) {
                                switch (num_type) {
                                    case DFNT_CHAR :
                                        phys.set(k,((char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_UCHAR:
                                        phys.set(k,((unsigned char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_INT8:
                                        phys.set(k,(int)((char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_UINT8:
                                        phys.set(k,(int)((unsigned char *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_FLOAT32:
                                        phys.set(k,(float) ((float *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_FLOAT64:
                                        phys.set(k,((double *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_INT16:
                                        phys.set(k,((short *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_UINT16:
                                        phys.set(k,((unsigned short *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_INT32:
                                        phys.set(k,((int *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                    case DFNT_UINT32:
                                        phys.set(k,((unsigned int *)&data[0])[k+i*dim_sizes[1]*dim_sizes[2]]);
                                        break;
                                }
                            }
                            double origin[2];
                            int attr_origin=SDfindattr(sds_id, "physOrigin");
                            if (attr_origin!=-1) {
                                if (SDreadattr(sds_id,attr_origin,origin)==0) phys.set_origin(origin[0],origin[1]);
                            }
                            double scale[2];
                            int attr_scale=SDfindattr(sds_id, "physScale");
                            if (attr_scale!=-1) {
                                if (SDreadattr(sds_id,attr_scale,scale)==0) phys.set_scale(scale[0],scale[1]);
                            }
                            if (i==1 && background.getSurf()==0) {
                                background=phys;
                            } else {
                                imagelist.push_back(phys);
                            }
                        }

                    }
                }
            }

            SDendaccess(sds_id);
        }

        if (background.getSurf() > 0 ) {
            for (auto& it : imagelist) {
                it = it -background;
            }
        }

        /* Terminate access to the SD interface and close the file. */
        SDend(sd_id);
    }
#endif
    return imagelist;
}


void physFormat::phys_write_HDF4(physD *phys, const char* fname) {
#if defined(HAVE_LIBMFHDF) || defined(HAVE_LIBMFHDFDLL)	
    if (phys) {
        intn istat=0;
        int32 sd_id = SDstart(fname, DFACC_CREATE);
        if (sd_id != FAIL) {
            int32 start[2], dimsizes[2];
            dimsizes[0]=phys->getH();
            dimsizes[1]=phys->getW();
            start[0]=0;
            start[1]=0;
            int32 sds_id=SDcreate(sd_id, phys->getName().c_str(), DFNT_FLOAT64, 2, dimsizes);
            comp_info c_info;
            c_info.deflate.level=6;
            istat+=SDsetcompress(sds_id, COMP_CODE_DEFLATE, &c_info);
            istat+=SDwritedata(sds_id, start, nullptr, dimsizes, (VOIDP)phys->Timg_buffer);
            double data[2];
            data[0]=phys->get_origin().x();
            data[1]=phys->get_origin().y();
            istat+=SDsetattr(sds_id, "physOrigin", DFNT_FLOAT64, 2, data);
            data[0]=phys->get_scale().x();
            data[1]=phys->get_scale().y();
            istat+=SDsetattr(sds_id, "physScale", DFNT_FLOAT64, 2, data);
            istat+=SDendaccess(sds_id);
            istat += SDend(sd_id);
        }

        if (istat!=0) {
            std::stringstream oss;
            oss<<"Error writing Hdf4 file: " << istat;
            throw phys_fileerror(oss.str());
        }
    }
#else
    WARNING("phys_write_HDF4: was not compiled with hdf4 enabled");
#endif
}


int inflate(FILE *source, FILE *dest)
{
    int ret;
    unsigned have;
    z_stream strm;
    std::vector<unsigned char> in(CHUNK), out(CHUNK);
    
    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit2 (&strm,windowBits | GZIP_ENCODING);
    if (ret != Z_OK)
        return ret;
    
    /* decompress until deflate stream ends or end of file */
    do {
        strm.avail_in = fread(&in[0], 1, CHUNK, source);
        if (ferror(source)) {
            (void)inflateEnd(&strm);
            return Z_ERRNO;
        }
        if (strm.avail_in == 0)
            break;
        strm.next_in = &in[0];
        
        /* run inflate() on input until output buffer not full */
        do {
            strm.avail_out = CHUNK;
            strm.next_out = &out[0];
            ret = inflate(&strm, Z_NO_FLUSH);
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            switch (ret) {
                case Z_NEED_DICT:
                    ret = Z_DATA_ERROR;     /* and fall through */
                    break;
                case Z_DATA_ERROR:
                case Z_MEM_ERROR:
                    (void)inflateEnd(&strm);
                    return ret;
            }
            have = CHUNK - strm.avail_out;
            if (fwrite(&out[0], 1, have, dest) != have || ferror(dest)) {
                (void)inflateEnd(&strm);
                return Z_ERRNO;
            }
        } while (strm.avail_out == 0);
        
        /* done when inflate() says it's done */
    } while (ret != Z_STREAM_END);
    
    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

std::string physFormat::gunzip (std::string filezipped) {

    std::string fileunzipped=filezipped;
    size_t last_idx = filezipped.find_last_of(".");
    
    if (std::string::npos == last_idx) return std::string();
    
    fileunzipped.erase(last_idx,filezipped.size());
    
    DEBUG(fileunzipped);
    
    FILE *filein;
    filein = fopen(filezipped.c_str(),"rb");
    if (filein == nullptr) return std::string();
    
    FILE *fileout;
    fileout = fopen(fileunzipped.c_str(),"wb");
    if (fileout == nullptr) return std::string();
    
    if (inflate(filein, fileout) != Z_OK ) {
        unlink(fileunzipped.c_str());
        return std::string();
    }
    fclose(filein);
    fclose(fileout);
    return fileunzipped;
}

std::vector <physD> physFormat::phys_open_shimadzu(std::string fname) {
    std::vector <physD> retPhys;

    unsigned int w=400;
    unsigned int h=250;
    unsigned int z=256;

    std::ifstream ifile(fname.c_str(), std::ios::in | std::ios::binary);

    std::streampos fsize = ifile.tellg();
    ifile.seekg( 0, std::ios::end );
    fsize = ifile.tellg() - fsize;

    unsigned int skip = static_cast<unsigned int>(fsize)-sizeof(unsigned short)*h*w*z;
    ifile.seekg(skip);

    std::vector<unsigned short> buffer(w*h);
    for (unsigned int i=0;i<z;i++){
        physD iimage(400,200,0.0,std::to_string(i));
        ifile.read((char *)&buffer[0],buffer.size()*sizeof(unsigned short));
#pragma omp parallel for
        for (unsigned int ii=0; ii<iimage.getSurf(); ii++)
            iimage.set(ii, buffer[ii]);
        retPhys.push_back(iimage);
    }
    return retPhys;
}

std::vector <physD> physFormat::phys_open(std::string fname, bool separate_rgb) {
    std::vector <physD> retPhys;
    size_t last_idx=0;

    std::string ext=fname;
    last_idx = fname.find_last_of(".");
    if (std::string::npos != last_idx) {
        ext.erase(0,last_idx + 1);
    }

    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << fname << " " << ext);


    std::string name=fname;
    last_idx = fname.find_last_of("\\/");
    if (std::string::npos != last_idx) {
        name.erase(0,last_idx + 1);
    }

    DEBUG(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> " << name << " " << ext);

    if (ext=="txt") {
        retPhys.push_back(physDouble_txt(fname.c_str()));
    } else if (ext.substr(0,3)=="tif"|| ext=="gel") {
        retPhys=physFormat::phys_open_tiff(fname, separate_rgb);
    } else if (ext=="spe") {
        retPhys=physFormat::phys_open_spe(fname);
    } else if (ext=="dat") {
        retPhys=physFormat::phys_open_shimadzu(fname);
    } else if (ext=="pcoraw") {
        retPhys=physFormat::phys_open_pcoraw(fname);
    } else if (ext=="inf") {
        retPhys=physFormat::phys_open_inf(fname);
    } else if (ext=="sif") {
        retPhys.push_back(physFormat::physInt_sif(fname.c_str()));
    } else if (ext=="b16") {
        retPhys.push_back(physFormat::physShort_b16(fname.c_str()));
    } else if (ext=="img") {
        DEBUG("++++++++++ calling physDouble_img");
        retPhys.push_back(physFormat::physDouble_img(fname));
        DEBUG("++++++++++ end calling physDouble_img");
    } else if (ext=="imd" || ext=="imi") {
        retPhys.push_back(physFormat::physUint_imd(fname.c_str()));
    } else if (ext.substr(0,3)=="fit") {
        retPhys=physFormat::phys_open_fits(fname);
    } else if (ext=="hdf") {
        retPhys=physFormat::phys_open_HDF4(fname);
    } else if (ext=="neu") {
        retPhys=physFormat::phys_resurrect_binary(fname);
    } else if (ext=="gz") {
        std::string filenameunzipped = physFormat::gunzip(fname);
        if ((!filenameunzipped.empty()) && (filenameunzipped!=fname)) {
            std::vector <physD> imagelist=physFormat::phys_open(filenameunzipped);
            unlink(filenameunzipped.c_str());
            for (auto& it : imagelist) {
                it.setName("");
                it.setShortName("");
            }
            retPhys.insert(retPhys.end(), imagelist.begin(), imagelist.end());
        }
        DEBUG(filenameunzipped);
    }
    
    for (size_t i=0;i<retPhys.size();i++) {

        std::ostringstream ss;
        if (retPhys.size()>1) {
            ss << i << " ";
        }

        DEBUG( "<" << retPhys[i].getName() << "> <" <<  retPhys[i].getShortName() << ">");

        // if Name and ShortName are set, don't change them
        if (retPhys[i].getName().empty())
            retPhys[i].setName(ss.str()+fname);

        if (retPhys[i].getShortName().empty())
            retPhys[i].setShortName(ss.str()+name);

        DEBUG( "<" << retPhys[i].getName() << "> <" <<  retPhys[i].getShortName() << ">");

        retPhys[i].setFromName(fname);
        retPhys[i].setType(PHYS_FILE);
    }
    DEBUG("<<<<<<<<<<<<<<<<<<<<");
    return retPhys;
}

std::vector<std::string> physFormat::phys_image_formats() {

    std::vector<std::string> retval={"neu", "txt", "spe", "pcoraw", "inf", "sif", "b16", "img", "imd", "imi", "gz", "dat"};

#ifdef HAVE_LIBTIFF
    retval.push_back("tif");
    retval.push_back("tiff");
    retval.push_back("gel");
#endif
#ifdef HAVE_LIBCFITSIO
    retval.push_back("fits");
#endif
#if defined(HAVE_LIBMFHDF) || defined(HAVE_LIBMFHDFDLL)
    retval.push_back("hdf");
#endif

    return retval;
}
