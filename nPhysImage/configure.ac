#											   -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([nPhysImage], [VERSION], [nphysimage@mellonta.net])
AC_CONFIG_SRCDIR([bidimvec.h])

# if set, creates header file with defs; otherwise exports defs variables to Makefile
AC_CONFIG_HEADERS([config.h])


uname=`uname`

# Checks for programs.
: ${CFLAGS=}

AC_PROG_CXX
AC_PROG_MAKE_SET

if test -d $prefix/include; then 
 CPPFLAGS="$CPPFLAGS -I$prefix/include"
fi 

if test -d $prefix/lib; then 
 LIBS="-L$prefix/lib $LIBS"
fi 

#AC_CANONICAL_HOST
## Check for which host we are on and setup a few things
## specifically based on the host
case $uname in
	*Darwin* )
		# Do something specific for mac
		AC_MSG_NOTICE(OSX : FIX headerpad_max_install_names)
		LDFLAGS+="-headerpad_max_install_names"
		;;
	linux*)
		# Do something specific for linux
	AC_SUBST([LCFLAGS], ["-Wl,-soname,\$@.0"])
		;;
	*BSD*)
		# Do something specific for BSD
		;;
	*)
		#Default Case
#		AC_MSG_ERROR([Your platform is not currently supported])
		;;
esac


# — Macro: AC_PATH_PROG (variable, prog-to-check-for, [value-if-not-found], [path = ‘$PATH’])
#AC_PATH_PROG([NVCC], [nvcc], [])
#m4_ifnblank([NVCC], AC_DEFINE([HAVE_CUDA], [1], [Include CUDA support]))

AC_SUBST(NVCCFLAGS, [])
AC_ARG_ENABLE([64], 
[  --enable-64			 Compile for 64bit], [CFLAGS+=" -m64"; NVCCFLAGS+=" -m64"])

if test "$with_cuda" != no; then
	AC_ARG_WITH([cuda], 
	[  --with-cuda=DIR   Path to nvcc to enable CUDA],,with_cuda="no")

	case $uname in
		*MINGW*)
		AC_MSG_NOTICE([CUDA not supported on MINGW])
			;;
		*)
		if test "$with_cuda" == yes; then
			AC_MSG_NOTICE(CUDA/Enable set to auto configuration)
			AC_PATH_PROG([NVCC], [nvcc], [])
			m4_ifnblank([NVCC], AC_DEFINE([HAVE_CUDA], [1], [Include CUDA support]))
			AC_SUBST(NVCCBASE, ["$prefix"])
		else
			AC_MSG_NOTICE(CUDA/Enable set to manual configuration)
			AC_SUBST(NVCC, ["$with_cuda"/bin/nvcc])
			AC_SUBST(NVCCBASE, ["$with_cuda"])

			# really check
			AC_PATH_PROG([NVCC], [nvcc], [], [path = "$with_cuda"/bin/nvcc])

			if test -n "$NVCC"; then
				AC_DEFINE([HAVE_CUDA], [1], [Include CUDA support])
			fi

			AC_MSG_NOTICE([NVCC: $NVCC])

		fi
		CFLAGS+=" -I$NVCCBASE/include"
		LDFLAGS+=" -L$NVCCBASE/lib"
	#	AC_CHECK_LIB([cudart], [main],, m4_fatal(Cannot find libcudart. Your nvcc installation might be incomplete))
	#	AC_CHECK_LIB([cufft], [main],, m4_fatal(Cannot find libcufft. Your nvcc installation might be incomplete))
			;;
	esac
else
	AC_MSG_NOTICE(CUDA disabled)
fi


# — Macro: AC_CHECK_LIB (library, function, [action-if-found], [action-if-not-found], [other-libraries])

#[LIBS="-lseq $LIBS"]

# Checks for libraries.
AC_CHECK_LIB([cfitsio], [main])
AC_CHECK_LIB([df], [main])

AC_CHECK_LIB([gsl], [main])
AC_CHECK_LIB([gslcblas], [main])


AC_CHECK_LIB([hdf5], [main])
AC_CHECK_LIB([hdf5_hl], [main])
AC_CHECK_LIB([jpeg], [main])
AC_CHECK_LIB([m], [main])
AC_CHECK_LIB([mfhdf], [main])
AC_CHECK_LIB([netpbm], [main])
AC_CHECK_LIB([pthread], [main])
AC_CHECK_LIB([tiff], [main])
AC_CHECK_LIB([z], [main])

case $uname in
	*MINGW*)
	AC_CHECK_LIB([fftw3-3], [main], [], [AC_MSG_ERROR([unable to find FFTW lib])])
		;;
	*)
	AC_CHECK_LIB([fftw3], [main], [], [AC_MSG_ERROR([unable to find FFTW lib])])
		;;
esac

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
# AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T

# Checks for library functions.
AC_FUNC_STRTOD
AC_CHECK_FUNCS([floor memset pow sqrt])

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

