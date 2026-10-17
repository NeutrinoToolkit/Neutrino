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

#include <iostream>
#include <string>
#include <fstream>
#include <ostream>
#include <sstream>
#include <map>
#include <regex>
#include <variant>

#include "bidimvec.h"
#include "tools.h"

#ifndef anymap_h_
#define anymap_h_

std::string trim(const std::string&, const std::string&);
std::ostream & operator<< (std::ostream &, struct anydata &);
std::istream & operator>> (std::istream &, struct anydata &);
bool check_vec(const std::string &);

// Tag type that distinguishes a serialised bidimvec string from a plain string.
// Stored as "(x:y)" — same wire format as before.
struct vec_string { std::string s; };

// Generic data holder — backed by std::variant instead of a manual tagged union.
struct anydata {
public:
    anydata() = default;

    // ----- assignment --------------------------------------------------------

    anydata& operator=(int rhs)          { _v = rhs; return *this; }
    anydata& operator=(unsigned int rhs) { _v = static_cast<int>(rhs); return *this; }
    anydata& operator=(double rhs)       { _v = rhs; return *this; }

    anydata& operator=(std::string rhs) {
        if (check_vec(rhs)) {
            DEBUG(10, "Got vector in string operator: " << rhs);
            _v = vec_string{rhs};
        } else {
            _v = std::move(rhs);
        }
        return *this;
    }

    template<class T>
    anydata& operator=(bidimvec<T> rhs) {
        // std::ostringstream ss; ss << rhs;
        // (default precision 6: origin and scale were rounded to 6 significant digits when set)
        std::ostringstream ss; ss.precision(15); ss << rhs;
        _v = vec_string{ss.str()};
        return *this;
    }

    // ----- type predicates ---------------------------------------------------

    bool is_none() const { return std::holds_alternative<std::monostate>(_v); }
    bool is_i()   const { return std::holds_alternative<int>(_v); }
    bool is_d()   const { return std::holds_alternative<double>(_v); }
    bool is_str() const { return std::holds_alternative<std::string>(_v); }
    bool is_vec() const { return std::holds_alternative<vec_string>(_v); }

    // ----- value extraction --------------------------------------------------

    int get_i() const {
        if (is_i()) return std::get<int>(_v);
        if (is_d()) { WARNING("double to int conversion"); return static_cast<int>(std::get<double>(_v)); }
        return 0;
    }

    double get_d() const {
        if (is_d()) return std::get<double>(_v);
        if (is_i()) return static_cast<double>(std::get<int>(_v));
        return 0.0;
    }

    std::string get_str() const {
        return std::visit([](auto&& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, std::monostate>)  return {};
            else if constexpr (std::is_same_v<T, int>)        return std::to_string(v);
            // else if constexpr (std::is_same_v<T, double>)     { std::ostringstream ss; ss << v; return ss.str(); }
            else if constexpr (std::is_same_v<T, double>)     { std::ostringstream ss; ss.precision(15); ss << v; return ss.str(); } // was 6 digits
            else if constexpr (std::is_same_v<T, std::string>) return v;
            else                                               return v.s;  // vec_string
        }, _v);
    }

    template<class T>
    bidimvec<T> get_vec() const {
        return is_vec() ? bidimvec<T>(std::get<vec_string>(_v).s) : bidimvec<T>("(0:0)");
    }

    // ----- implicit conversions (kept for call-site compatibility) -----------

    operator int() const    { return get_i(); }
    operator double() const { return get_d(); }

    operator std::string() const {
        if (is_none()) DEBUG("accessing uninitialized anydata");
        return get_str();
    }

    template<class T>
    operator bidimvec<T>() const {
        if (!is_vec()) WARNING("wrong datatype (vec) required for map member!!");
        return bidimvec<T>(get_str());
    }

private:
    std::variant<std::monostate, int, double, std::string, vec_string> _v;
};


// final data structure
class anymap : public std::map<std::string, anydata> {

public:
    anymap() = default;

    bool have(std::string search_me) const {
        return size() && find(search_me) != end();
    }

    void loader(std::istream &);
    void dumper(std::ostream &);
};

#endif
