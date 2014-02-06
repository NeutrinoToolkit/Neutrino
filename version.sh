#!/bin/sh
git_lasttag=`git tag | tail -1`
git_number=`git rev-list ${git_lasttag}..HEAD --count`
echo ${git_lasttag#v}-r${git_number}
