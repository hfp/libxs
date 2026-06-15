%global debug_package %{nil}
%bcond_without tests

Name:           libxs
Version:        0.0.0
Release:        %autorelease
Summary:        Portable C library for numerics, memory operations, and utilities

License:        BSD-3-Clause
URL:            https://github.com/hfp/libxs
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  bash
BuildRequires:  gcc
BuildRequires:  gcc-gfortran
BuildRequires:  make
%if %{with tests}
BuildRequires:  gcc-c++
BuildRequires:  flexiblas-devel
BuildRequires:  gawk
BuildRequires:  ocl-icd-devel
%endif

%description
LIBXS is a portable C library providing building blocks for memory operations,
numerics, synchronization, hashing, random number generation, and related
low-level utilities. It was originally developed as part of LIBXSMM.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description devel
This package contains headers, the Fortran module interface, pkg-config
metadata, CMake package files, the supported header-only source tree, and API
documentation for developing applications that use LIBXS.

%prep
%autosetup

%build
%make_build GNU=1 STATIC=0 \
    POUTDIR=%{_lib} PPKGDIR=%{_lib}/pkgconfig PCMKDIR=%{_lib}/cmake/%{name}

%install
%make_install PREFIX=%{_prefix} CLEAN=0 STATIC=0 \
    POUTDIR=%{_lib} PPKGDIR=%{_lib}/pkgconfig PCMKDIR=%{_lib}/cmake/%{name}

rm -f %{buildroot}%{_datadir}/%{name}/LICENSE.md

%check
%if %{with tests}
%make_build tests GNU=1 STATIC=0 BLASLIB=flexiblas
%endif

%files
%license LICENSE.md
%doc README.md
%{_libdir}/libxs.so.*

%files devel
%license LICENSE.md
%doc %{_datadir}/%{name}
%{_includedir}/%{name}/
%{_libdir}/libxs.so
%{_libdir}/pkgconfig/libxs*.pc
%{_libdir}/cmake/libxs/

%changelog
%autochangelog
