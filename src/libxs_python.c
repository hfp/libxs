/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXS library.                                     *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxs/                              *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
#if defined(__PYTHON) && defined(LIBXS_BUILD) && !defined(__STATIC)
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXS_OFFLOAD_TARGET))
#endif
#include <Python.h> /* must be included first */
#if defined(LIBXS_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#endif
#include <libxs.h>


#if defined(__PYTHON) && defined(LIBXS_BUILD) && !defined(__STATIC)

LIBXS_API PyObject* libxspy_get_target_arch(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_get_target_arch(PyObject* self, PyObject* args)
{
  LIBXS_UNUSED(self); LIBXS_UNUSED(args);
  return PyString_InternFromString(libxs_get_target_arch());
}

LIBXS_API PyObject* libxspy_set_target_arch(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_set_target_arch(PyObject* self, PyObject* args)
{
  int ivalue = LIBXS_TARGET_ARCH_UNKNOWN;
  char* svalue = NULL;
  LIBXS_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "s", &svalue)) {
    libxs_set_target_arch(svalue);
  }
  else if (0 != PyArg_ParseTuple(args, "i", &ivalue)) {
    libxs_set_target_archid(ivalue);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


LIBXS_API PyObject* libxspy_get_target_archid(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_get_target_archid(PyObject* self, PyObject* args)
{
  LIBXS_UNUSED(self); LIBXS_UNUSED(args);
  return Py_BuildValue("i", libxs_get_target_archid());
}

LIBXS_API PyObject* libxspy_set_target_archid(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_set_target_archid(PyObject* self, PyObject* args)
{
  int value = LIBXS_TARGET_ARCH_UNKNOWN;
  LIBXS_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "i", &value)) {
    libxs_set_target_archid(value);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


LIBXS_API PyObject* libxspy_get_verbosity(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_get_verbosity(PyObject* self, PyObject* args)
{
  LIBXS_UNUSED(self); LIBXS_UNUSED(args);
  return Py_BuildValue("i", libxs_get_verbosity());
}

LIBXS_API PyObject* libxspy_set_verbosity(PyObject* self, PyObject* args);
LIBXS_API PyObject* libxspy_set_verbosity(PyObject* self, PyObject* args)
{
  int value = 0;
  LIBXS_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "i", &value)) {
    libxs_set_verbosity(value);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


LIBXS_API PyMODINIT_FUNC initlibxs(void);
LIBXS_API PyMODINIT_FUNC initlibxs(void)
{
  static PyMethodDef pymethod_def[] = {
    { "GetTargetArch", libxspy_get_target_arch, METH_NOARGS,
      PyDoc_STR("Get the name of the code path.") },
    { "SetTargetArch", libxspy_set_target_arch, METH_VARARGS,
      PyDoc_STR("Set the name of the code path.") },
    { "GetTargetArchId", libxspy_get_target_archid, METH_NOARGS,
      PyDoc_STR("Get the id of the code path.") },
    { "SetTargetArchId", libxspy_set_target_archid, METH_VARARGS,
      PyDoc_STR("Set the id of the code path.") },
    { "GetVerbosity", libxspy_get_verbosity, METH_NOARGS,
      PyDoc_STR("Get the verbosity level.") },
    { "SetVerbosity", libxspy_set_verbosity, METH_VARARGS,
      PyDoc_STR("Set the verbosity level.") },
    { NULL, NULL, 0, NULL } /* end of table */
  };
  PyObject *const pymod = Py_InitModule3("libxs", pymethod_def, PyDoc_STR(
    "Library targeting Intel Architecture for small, dense or "
    "sparse matrix multiplications, and small convolutions."));
  PyModule_AddIntConstant(pymod, "VERSION_API", LIBXS_VERSION2(LIBXS_VERSION_MAJOR, LIBXS_VERSION_MINOR));
  PyModule_AddIntConstant(pymod, "VERSION_ALL", LIBXS_VERSION);
  PyModule_AddIntConstant(pymod, "VERSION_MAJOR", LIBXS_VERSION_MAJOR);
  PyModule_AddIntConstant(pymod, "VERSION_MINOR", LIBXS_VERSION_MINOR);
  PyModule_AddIntConstant(pymod, "VERSION_UPDATE", LIBXS_VERSION_UPDATE);
  PyModule_AddIntConstant(pymod, "VERSION_PATCH", LIBXS_VERSION_PATCH);
  PyModule_AddStringConstant(pymod, "VERSION", LIBXS_VERSION);
  PyModule_AddStringConstant(pymod, "BRANCH", LIBXS_BRANCH);
  PyModule_AddIntConstant(pymod, "TARGET_ARCH_UNKNOWN", LIBXS_TARGET_ARCH_UNKNOWN);
  PyModule_AddIntConstant(pymod, "TARGET_ARCH_GENERIC", LIBXS_TARGET_ARCH_GENERIC);
  PyModule_AddIntConstant(pymod, "X86_GENERIC", LIBXS_X86_GENERIC);
  PyModule_AddIntConstant(pymod, "X86_SSE3", LIBXS_X86_SSE3);
  PyModule_AddIntConstant(pymod, "X86_SSE42", LIBXS_X86_SSE42);
  PyModule_AddIntConstant(pymod, "X86_AVX", LIBXS_X86_AVX);
  PyModule_AddIntConstant(pymod, "X86_AVX2", LIBXS_X86_AVX2);
  PyModule_AddIntConstant(pymod, "X86_AVX512", LIBXS_X86_AVX512);
  PyModule_AddIntConstant(pymod, "X86_AVX512_MIC", LIBXS_X86_AVX512_MIC);
  PyModule_AddIntConstant(pymod, "X86_AVX512_KNM", LIBXS_X86_AVX512_KNM);
  PyModule_AddIntConstant(pymod, "X86_AVX512_CORE", LIBXS_X86_AVX512_CORE);
  PyModule_AddIntConstant(pymod, "X86_AVX512_CLX", LIBXS_X86_AVX512_CLX);
  PyModule_AddIntConstant(pymod, "X86_AVX512_CPX", LIBXS_X86_AVX512_CPX);
  libxs_init(); /* initialize LIBXS */
}

#endif /*defined(__PYTHON) && defined(LIBXS_BUILD) && !defined(__STATIC)*/
