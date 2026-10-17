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
#ifndef nHdf5io_H_
#define nHdf5io_H_

// HDF5 reading/writing for the OpenHdf5 plugin, without Qt (tested on its own)
// works with HDF5 1.10 (Ubuntu 24.04), 1.12 (EL9), 1.14 and 2.x

#include <string>
#include <vector>

#include "hdf5.h"
#include "nPhysImageF.h"

namespace nHdf5 {

// an HDF5 id closed when going out of scope
class Handle {
public:
    Handle() = default;
    Handle(hid_t id, herr_t (*closer)(hid_t)) : my_id(id), my_close(closer) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle(Handle&& o) noexcept : my_id(o.my_id), my_close(o.my_close) { o.my_id = H5I_INVALID_HID; }
    Handle& operator=(Handle&& o) noexcept { if (this != &o) { reset(); my_id = o.my_id; my_close = o.my_close; o.my_id = H5I_INVALID_HID; } return *this; }
    ~Handle() { reset(); }
    void reset() { if (my_id >= 0 && my_close) my_close(my_id); my_id = H5I_INVALID_HID; }
    operator hid_t() const { return my_id; }
    bool valid() const { return my_id >= 0; }
private:
    hid_t my_id = H5I_INVALID_HID;
    herr_t (*my_close)(hid_t) = nullptr;
};

// what can be done with a dataset
enum class Kind {
    none,       // not readable as numbers (string, compound, enum, 4D...)
    data1D,     // numbers: copied as text
    image,      // 2D numbers, or a dataset of 2D arrays (their mean)
    frames      // 3D numbers [frame][y][x]
};

struct Attribute {
    std::string name;
    std::string type;                   // e.g. "int16", "float64", "string"
    std::vector<double> numbers;        // integer and float attributes (converted by HDF5, any size)
    std::vector<std::string> strings;   // string attributes (fixed or variable length)
    std::string text() const;           // values separated by spaces
};

struct Node {
    enum Type { group, dataset, datatype, softLink, externalLink, other } type = other;
    std::string name;                   // link name
    std::string path;                   // full path in the file
    std::string typeText;               // type of the dataset (or link target)
    std::string dataText;               // dimensions, members...
    Kind kind = Kind::none;
    std::vector<hsize_t> dims;          // dataspace dimensions (for image: [h][w], frames: [n][h][w])
    std::vector<Attribute> attributes;
    std::vector<Node> children;
};

// the tree of a file; false (and error) if the file cannot be opened
bool scanFile(const std::string &fname, Node &root, std::string &error);

// read a 2D dataset (or the mean of a dataset of 2D arrays) with its neutrino attributes
bool readImage(const std::string &fname, const std::string &path, physD &image, std::string &error);

// frames of a 3D dataset [n][h][w]: one frame, all the frames, or their mean (NaN skipped)
bool readFrame(const std::string &fname, const std::string &path, hsize_t frame, physD &image, std::string &error);
bool readFrames(const std::string &fname, const std::string &path, std::vector<physD> &images, std::string &error);
bool readFramesMean(const std::string &fname, const std::string &path, physD &image, std::string &error);

// a 1D dataset as numbers
bool readData1D(const std::string &fname, const std::string &path, std::vector<double> &data, std::string &error);

// write an image as /neutrino (float64, chunked, deflate if available) with origin, scale and names
bool writeImage(const physD &image, const std::string &fname, const std::string &version, std::string &error);

// readable name of a datatype (int16, uint8, float32, string, compound...)
std::string typeName(hid_t type);

}

#endif
