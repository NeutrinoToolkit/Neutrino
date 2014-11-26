/*
 *
 *    Copyright (C) 2013 Alessandro Flacco, Tommaso Vinci All Rights Reserved
 * 
 *    This file is part of nPhysImage library.
 *
 *    nPhysImage is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    nPhysImage is distributed in the hope that it will be useful,
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
#include "nPhysImageF.h"

#ifndef n_phys_formats__
#define n_phys_formats__

#if defined(HAVE_LIBMFHDF) || defined(HAVE_LIBMFHDFDLL)
extern "C" {
#define intf hdf4_intf
#define int8 hdf4_int8
#include "hdf.h"
#include "mfhdf.h"
#undef intf
#undef int8
}
#endif

// .alex. incredibile quanto bordello genera questo...
//using namespace std;

// standard formats
class physDouble_txt : public nPhysImageF<double> {
public:
	physDouble_txt(const char *);
};

class physDouble_tiff : public nPhysImageF<double> {
public:
	physDouble_tiff(const char *);
};

// LULI scanner it opens .inf and the .img associated with 16 bit raw data
std::vector <nPhysImageF<double> *> phys_open_inf(std::string);

class physDouble_img : public nPhysImageF<double> {
public:
	physDouble_img(std::string);
};

int phys_write_fits(nPhysImageF<double> *phys, const char * fname, float compression=0);

std::vector <nPhysImageF<double> *> phys_open_fits(std::string);

class physDouble_asc : public nPhysImageF<double> {
public:
	physDouble_asc(const char *);
};

// external library formats
class physInt_pgm : public nPhysImageF<int> {
public:
	physInt_pgm(const char *);
};

// external library formats
#ifdef HAVE_LIBNETPBM
extern "C" {
#include <pgm.h>
}
class physGray_pgm : public nPhysImageF<gray> {
public:
	physGray_pgm(const char *);
};
#endif

// proprietary formats

// Andor .SIF format
class physInt_sif : public nPhysImageF<int> {
public:
	physInt_sif(std::string);
};

// PCO .B16 format
class physShort_b16 : public nPhysImageF<short> {
public:
	physShort_b16(const char *);
};

// Optronics luli
class physUint_imd : public nPhysImageF<unsigned int> {
public:
	physUint_imd(std::string);
};


// Specialization for 2-byte types.
inline void endian2Swap(char* dest, char const* src)
{
    // Use bit manipulations instead of accessing individual bytes from memory, much faster.
    unsigned short* p_dest = reinterpret_cast< unsigned short* >(dest);
    unsigned short const* const p_src = reinterpret_cast< unsigned short const* >(src);
    *p_dest = (*p_src >> 8) | (*p_src << 8);
}

// Specialization for 4-byte types.
inline void endian4Swap(char* dest, char const* src)
{
    // Use bit manipulations instead of accessing individual bytes from memory, much faster.
    unsigned int* p_dest = reinterpret_cast< unsigned int* >(dest);
    unsigned int const* const p_src = reinterpret_cast< unsigned int const* >(src);
    *p_dest = (*p_src >> 24) | ((*p_src & 0x00ff0000) >> 8) | ((*p_src & 0x0000ff00) << 8) | (*p_src << 24);
}


template <typename T>
T swap_endian(T u)
{
    union
    {
        T u;
        unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
        dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
}


// --- write out ---


// now use anymap::loader(std::istream &) and anymap::dumper(std::ostream &)

//std::ostream &
//operator<< (std::ostream &, phys_properties &);

//std::istream &
//operator>> (std::istream &, phys_properties &);

// dump out for state save
int
phys_dump_binary(nPhysImageF<double> *my_phys, const char *ofile);

int 
phys_dump_binary(nPhysImageF<double> *, std::ofstream &);

int
phys_resurrect_binary(nPhysImageF<double> *, std::ifstream &);

int
phys_resurrect_old_binary(nPhysImageF<double> *, std::ifstream &);

std::vector <nPhysImageF<double> *> phys_resurrect_binary(std::string);

//generic raw open
int
phys_open_RAW(nPhysImageF<double> *, int, int, bool);

//write neutrino tiff files
int 
phys_write_tiff(nPhysImageF<double> *, const char *);


//! HDF stuff
std::vector <nPhysImageF<double> *> phys_open_HDF4(std::string);
int phys_write_HDF4(nPhysImageF<double> *, const char*);


std::vector <nPhysImageF<double> *> phys_open_spe(std::string);


std::vector <nPhysImageF<double> *> phys_open(std::string);

std::string gunzip(std::string);

#endif

