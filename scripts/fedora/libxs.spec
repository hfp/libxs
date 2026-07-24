%bcond tests 1

%global soversion 1

Name:           libxs
Version:        1.0.0
Release:        %autorelease
Summary:        Portable C library for numerics, memory operations, and utilities

License:        BSD-3-Clause
URL:            https://github.com/hfp/libxs
Source0:        https://github.com/hfp/libxs/releases/download/%{version}/%{name}-%{version}.tar.gz

BuildRequires:  bash
BuildRequires:  cmake
BuildRequires:  cmake-rpm-macros
BuildRequires:  gcc
BuildRequires:  gcc-gfortran
%if %{with tests}
BuildRequires:  flexiblas-devel
BuildRequires:  gawk
%endif

%description
LIBXS is a portable C library providing building blocks for memory operations,
numerics, synchronization, hashing, random number generation, and related
low-level utilities. It was originally developed as part of LIBXSMM.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       gcc-gfortran%{?_isa}

%description devel
This package contains headers, the Fortran module interface, pkg-config
metadata, and CMake package files for developing applications that use LIBXS.

%package doc
Summary:        Documentation for %{name}
BuildArch:      noarch

%description doc
This package contains the API and usage documentation for LIBXS.

%prep
%autosetup -p1

%conf
%cmake \
    -DBUILD_TESTING:BOOL=%{with tests} \
    -DCMAKE_INSTALL_Fortran_MODULES:PATH=%{_fmoddir}/%{name} \
    -DLIBXS_FORTRAN:BOOL=ON \
    -DLIBXS_INSTALL_HEADER_ONLY:BOOL=OFF

%build
%cmake_build

%install
%cmake_install

%check
%if %{with tests}
%ctest --output-on-failure
%endif

%files
%license LICENSE.md
%{_libdir}/libxs.so.%{soversion}
%{_libdir}/libxs.so.%{soversion}.*

%files devel
%{_includedir}/%{name}/
%{_fmoddir}/%{name}/
%{_libdir}/libxs.so
%{_libdir}/pkgconfig/libxs*.pc
%{_libdir}/cmake/libxs/
%{_datadir}/%{name}/

%files doc
%dir %{_docdir}/%{name}
%license %{_docdir}/%{name}/LICENSE.md
%doc %{_docdir}/%{name}/README.md
%doc %{_docdir}/%{name}/index.md
%doc %{_docdir}/%{name}/libxs_*.md
%doc %{_docdir}/%{name}/*.pdf
%doc %{_docdir}/%{name}/ozaki/
%doc %{_docdir}/%{name}/predict/
%doc %{_docdir}/%{name}/samples/

%changelog
%autochangelog
