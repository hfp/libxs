%bcond_without tests

Name:           libxs
Version:        1.0.0
Release:        %autorelease
Summary:        Portable C library for numerics, memory operations, and utilities

License:        BSD-3-Clause
URL:            https://github.com/hfp/libxs
Source0:        https://github.com/hfp/libxs/releases/download/%{version}/%{name}-%{version}.tar.gz

BuildRequires:  bash
BuildRequires:  gcc
BuildRequires:  gcc-gfortran
BuildRequires:  cmake
BuildRequires:  cmake-rpm-macros
BuildRequires:  ninja-build
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

%description devel
This package contains headers, the Fortran module interface, pkg-config
metadata, CMake package files, the supported header-only source tree, and API
documentation for developing applications that use LIBXS.

%prep
%autosetup
%build
%cmake \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_TESTING=%{libxs_build_testing} \
    -DLIBXS_FORTRAN=ON
%cmake_build

%install
%cmake_install

# The license is already installed via %%license below; avoid a duplicate copy
# in the API documentation directory.
rm -f %{buildroot}%{_datadir}/%{name}/LICENSE.md

%check
%if %{with tests}
%ctest --output-on-failure --parallel %{_smp_build_ncpus}
%endif

%files
%license LICENSE.md
%doc README.md
%{_libdir}/libxs.so.1{,.*}

%files devel
%{_datadir}/%{name}/
%{_includedir}/%{name}/
%{_libdir}/libxs.so
%{_libdir}/pkgconfig/libxs*.pc
%{_libdir}/cmake/libxs/

%changelog
%autochangelog
