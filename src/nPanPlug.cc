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
#include "nPanPlug.h"
#include <algorithm>

// constructed at the first use: the plugins register themselves from the static initialisation of other files
std::vector<nPanPlug>& nPanPlug::registry() {
    static std::vector<nPanPlug> plugs;
    return plugs;
}

const std::vector<nPanPlug>& nPanPlug::all() {
    return registry();
}

const nPanPlug* nPanPlug::find(const QString &name) {
    for (auto &plug : registry()) {
        if (plug.name == name) return &plug;
    }
    return nullptr;
}

bool nPanPlug::add(nPanPlug plug) {
    if (find(plug.name)) return false;
    auto &plugs = registry();
    // sorted by name (the order of the static initialisation is not defined)
    auto pos = std::lower_bound(plugs.begin(), plugs.end(), plug,
                                [](const nPanPlug &a, const nPanPlug &b) { return a.name < b.name; });
    plugs.insert(pos, std::move(plug));
    return true;
}
